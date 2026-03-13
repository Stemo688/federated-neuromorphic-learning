#!/usr/bin/env python3
"""Node worker — runs on each Raspberry Pi with Akida AKD1000.

Two-stage architecture:
  Stage 1: DS-CNN feature extractor (Akida v2) — forward() produces int8 features
  Stage 2: FullyConnected edge learning model (Akida v1) — STDP on binary features

The feature extractor runs in software (v2 model can't map to v2 device for
inference-only forward), while the edge model maps to hardware for on-chip learning.

Lifecycle:
  1. Build feature extractor (convert pretrained DS-CNN) + edge learning model
  2. Run few-shot edge learning on local data
  3. Listen for orchestrator commands on TCP socket
  4. Exchange weights with peer / apply federation strategies
  5. Evaluate and report metrics

Usage:
  python node_worker.py --node-id claudio --data-dir ./data --model-dir ./models
"""

import argparse
import json
import logging
import pickle
import socket
import struct
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (inlined — no relative imports on the Pi)
# ---------------------------------------------------------------------------
EXCHANGE_PORT = 9999
COMMAND_PORT = 9998
INPUT_SHAPE = (49, 10, 1)
EDGE_LAYER_NAME = "edge_layer"
NUM_CLASSES = 3           # Total novel classes across all nodes
NEURONS_PER_CLASS = 50
NUM_FEATURES = 64         # DS-CNN pw_separable_4 output features (default, auto-detected)
NUM_WEIGHTS = 20          # STDP selectivity: must be << NUM_FEATURES
NOVEL_CLASSES = ["backward", "follow", "forward"]  # Global label ordering
BINARIZATION_METHODS = ["mean", "median", "entropy"]


# ---------------------------------------------------------------------------
# Two-stage model building
# ---------------------------------------------------------------------------

def build_models(model_dir: Path, params: dict | None = None) -> tuple:
    """Build the two-stage pipeline: feature extractor + edge learning model.

    Args:
        model_dir: Directory for cached model files.
        params: Optional dict with edge model hyperparameters:
            - num_weights: STDP selectivity (default: NUM_WEIGHTS)
            - neurons_per_class: Neurons per class (default: NEURONS_PER_CLASS)
            - learning_competition: STDP competition (default: 0.1)
            - feat_extractor_name: Filename of extractor (default: "feat_extractor.fbz")

    Returns:
        (feat_model, edge_model, num_features) tuple
    """
    import akida

    if params is None:
        params = {}

    feat_name = params.get("feat_extractor_name", "feat_extractor.fbz")
    feat_path = model_dir / feat_name

    # --- Feature extractor (Akida v2, DS-CNN without classification head) ---
    if feat_path.exists():
        log.info("Loading cached feature extractor from %s", feat_path)
        feat_model = akida.Model(str(feat_path))
    else:
        log.info("Building feature extractor from pretrained ds_cnn_kws ...")
        from akida_models import ds_cnn_kws_pretrained
        from cnn2snn import convert

        model_keras = ds_cnn_kws_pretrained()
        feat_model = convert(model_keras)
        feat_model.pop_layer()  # dequantizer
        feat_model.pop_layer()  # dense_5
        # Now last layer is pw_separable_4: output (1, 1, 64), int8
        feat_model.save(str(feat_path))
        log.info("Feature extractor saved to %s", feat_path)

    # Check for wide projection weights (saved alongside wide extractors)
    proj_stem = feat_name.replace(".fbz", "")
    proj_W_path = model_dir / f"{proj_stem}_proj_W.npy"
    proj_b_path = model_dir / f"{proj_stem}_proj_b.npy"
    wide_projection = None
    if proj_W_path.exists() and proj_b_path.exists():
        proj_W = np.load(str(proj_W_path))
        proj_b = np.load(str(proj_b_path))
        wide_projection = (proj_W, proj_b)
        num_features = proj_W.shape[1]  # e.g. 128 or 256
        log.info("Loaded wide projection: %s -> %d features", proj_W.shape, num_features)
    else:
        # Auto-detect feature dimensions from extractor output
        num_features = feat_model.output_shape[-1]

    log.info("Feature extractor ready: input=%s, output=%s, num_features=%d",
             feat_model.input_shape, feat_model.output_shape, num_features)

    # --- Edge learning model (Akida v1, standalone FullyConnected) ---
    npc = params.get("neurons_per_class", NEURONS_PER_CLASS)
    nw = params.get("num_weights", NUM_WEIGHTS)
    lc = params.get("learning_competition", 0.1)

    edge_model = _build_edge_model(
        NUM_CLASSES * npc, NUM_CLASSES,
        num_features=num_features,
        num_weights=nw, learning_competition=lc,
    )
    log.info("Edge model ready: %d units (%d npc), %d classes, nw=%d, lc=%.2f, feat=%d",
             NUM_CLASSES * npc, npc, NUM_CLASSES, nw, lc, num_features)

    return feat_model, edge_model, num_features, wide_projection


def _build_edge_model(units: int, num_classes: int,
                      num_features: int = NUM_FEATURES,
                      num_weights: int = NUM_WEIGHTS,
                      learning_competition: float = 0.1):
    """Create and compile a standalone edge learning model."""
    import akida

    edge_model = akida.Model([
        akida.InputData(
            name="input",
            input_shape=(1, 1, num_features),
            input_bits=1,
        ),
        akida.FullyConnected(
            units=units,
            name=EDGE_LAYER_NAME,
            activation=False,
        ),
    ])
    edge_model.compile(
        optimizer=akida.AkidaUnsupervised(
            num_weights=num_weights,
            num_classes=num_classes,
            learning_competition=learning_competition,
        )
    )

    # Map to hardware if available
    devices = akida.devices()
    if devices:
        try:
            edge_model.map(devices[0])
            log.info("Edge model mapped to HW: %s", devices[0].desc)
        except Exception as e:
            log.warning("Edge model SW mode: %s", e)

    return edge_model


# ---------------------------------------------------------------------------
# Feature extraction + binarization
# ---------------------------------------------------------------------------

def extract_features(feat_model, X: np.ndarray, num_features: int = NUM_FEATURES,
                     wide_projection: tuple | None = None) -> np.ndarray:
    """Run feature extractor, return raw features (N, num_features).

    If wide_projection is provided, applies linear projection after Akida extraction.
    """
    features = feat_model.forward(X)  # (N, 1, 1, backbone_dim), int8
    backbone_dim = feat_model.output_shape[-1]
    features = features.reshape(len(X), backbone_dim).astype(np.float32)
    if wide_projection is not None:
        proj_W, proj_b = wide_projection
        features = features @ proj_W + proj_b  # (N, wide_dim)
    return features


def binarize(features_2d: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Binarize features using per-feature thresholds.

    Args:
        features_2d: shape (N, D), float
        thresholds: shape (D,), per-feature threshold

    Returns:
        Binary features, shape (N, 1, 1, D), dtype uint8, values {0, 1}
    """
    num_features = features_2d.shape[1]
    binary = (features_2d > thresholds).astype(np.uint8)
    return binary.reshape(len(features_2d), 1, 1, num_features)


def compute_thresholds(features_2d: np.ndarray, labels: np.ndarray = None,
                       method: str = "mean") -> np.ndarray:
    """Compute per-feature binarization thresholds from training data.

    Args:
        features_2d: shape (N, D), float features
        labels: shape (N,), int labels (required for "entropy" method)
        method: "mean", "median", or "entropy"

    Returns:
        Thresholds array of shape (D,).
    """
    if method == "mean":
        return features_2d.mean(axis=0)
    elif method == "median":
        return np.median(features_2d, axis=0)
    elif method == "entropy":
        if labels is None:
            log.warning("Entropy binarization requires labels, falling back to mean")
            return features_2d.mean(axis=0)
        return _entropy_thresholds(features_2d, labels)
    else:
        raise ValueError(f"Unknown binarization method: {method}")


def _mutual_information(binary: np.ndarray, labels: np.ndarray) -> float:
    """Compute mutual information between a binary variable and labels."""
    n = len(labels)
    unique_labels = np.unique(labels)

    # Joint and marginal probabilities
    mi = 0.0
    for b_val in (0, 1):
        b_mask = binary == b_val
        p_b = b_mask.sum() / n
        if p_b == 0:
            continue
        for lbl in unique_labels:
            l_mask = labels == lbl
            p_l = l_mask.sum() / n
            p_bl = (b_mask & l_mask).sum() / n
            if p_bl > 0 and p_l > 0:
                mi += p_bl * np.log2(p_bl / (p_b * p_l))
    return mi


def _entropy_thresholds(features_2d: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-feature thresholds maximizing mutual information with labels."""
    num_features = features_2d.shape[1]
    thresholds = np.zeros(num_features, dtype=np.float32)

    for d in range(num_features):
        col = features_2d[:, d]
        best_mi = -1.0
        best_t = col.mean()
        for q in np.linspace(0.1, 0.9, 17):
            t = np.quantile(col, q)
            binary = (col > t).astype(int)
            mi = _mutual_information(binary, labels)
            if mi > best_mi:
                best_mi = mi
                best_t = t
        thresholds[d] = best_t

    return thresholds


# ---------------------------------------------------------------------------
# Edge learning and evaluation
# ---------------------------------------------------------------------------

def run_edge_learning(
    feat_model,
    edge_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_features: int = NUM_FEATURES,
    binarization_method: str = "mean",
    wide_projection: tuple | None = None,
) -> tuple[dict, np.ndarray]:
    """Run on-chip few-shot edge learning via STDP.

    Returns:
        (metrics_dict, binarization_thresholds)
    """
    log.info("Starting edge learning: %d samples, labels=%s, bin=%s",
             len(X_train), np.unique(y_train).tolist(), binarization_method)

    # Stage 1: Feature extraction
    t0 = time.perf_counter()
    features_raw = extract_features(feat_model, X_train, num_features, wide_projection)
    feat_ms = (time.perf_counter() - t0) * 1000

    # Compute and apply binarization thresholds from training data
    thresholds = compute_thresholds(features_raw, y_train, method=binarization_method)
    features_bin = binarize(features_raw, thresholds)
    ones_ratio = features_bin.mean()

    # Stage 2: STDP learning on binary features
    t0 = time.perf_counter()
    edge_model.fit(features_bin, y_train)
    learn_ms = (time.perf_counter() - t0) * 1000

    log.info("Feature extraction: %.1f ms, Edge learning: %.1f ms, ones_ratio: %.3f",
             feat_ms, learn_ms, ones_ratio)
    metrics = {
        "feature_extraction_ms": feat_ms,
        "learning_time_ms": learn_ms,
        "total_time_ms": feat_ms + learn_ms,
        "samples_processed": len(X_train),
        "num_classes": len(np.unique(y_train)),
        "ones_ratio": float(ones_ratio),
        "binarization_method": binarization_method,
    }
    return metrics, thresholds


def evaluate_model(
    feat_model,
    edge_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    thresholds: np.ndarray,
    num_features: int = NUM_FEATURES,
    wide_projection: tuple | None = None,
) -> dict:
    """Run two-stage inference and compute accuracy."""
    # Stage 1: Feature extraction + binarization (using training thresholds)
    t0 = time.perf_counter()
    features_raw = extract_features(feat_model, X_test, num_features, wide_projection)
    features_bin = binarize(features_raw, thresholds)
    feat_ms = (time.perf_counter() - t0) * 1000

    # Stage 2: Classification
    t0 = time.perf_counter()
    predictions = edge_model.predict_classes(features_bin, num_classes=num_classes)
    inf_ms = (time.perf_counter() - t0) * 1000

    correct = int((predictions == y_test).sum())
    accuracy = correct / len(y_test) if len(y_test) > 0 else 0.0

    # Per-class accuracy
    per_class = {}
    for cls in range(num_classes):
        mask = y_test == cls
        if mask.sum() > 0:
            cls_acc = float((predictions[mask] == cls).sum() / mask.sum())
            per_class[str(cls)] = cls_acc

    log.info("Eval: %.1f%% accuracy (%d/%d), feat=%.1f ms, inf=%.1f ms",
             accuracy * 100, correct, len(y_test), feat_ms, inf_ms)

    return {
        "accuracy": float(accuracy),
        "correct": correct,
        "total": len(y_test),
        "feature_extraction_ms": feat_ms,
        "inference_time_ms": inf_ms,
        "per_class_accuracy": per_class,
        "predictions": predictions.tolist(),
    }


# ---------------------------------------------------------------------------
# Weight extraction / injection
# ---------------------------------------------------------------------------

def extract_weights(edge_model) -> tuple[np.ndarray, dict]:
    """Extract edge layer weights and build class map.

    Akida weights shape: (1, 1, features, num_neurons)
    We reshape to (num_neurons, features) for federation.

    Returns:
        (weights_2d, class_map) where:
        - weights_2d: shape (num_neurons, features), dtype int8
        - class_map: {label: [neuron_indices]}
    """
    layer = edge_model.get_layer(EDGE_LAYER_NAME)
    raw_weights = layer.get_variable("weights")  # (1, 1, D, N)

    # Reshape to (num_neurons, features)
    w_squeezed = raw_weights.squeeze()  # (D, N)
    weights_2d = w_squeezed.T  # (N, D)

    num_neurons = weights_2d.shape[0]
    neurons_per_class = num_neurons // NUM_CLASSES

    class_map = {}
    for c in range(NUM_CLASSES):
        start = c * neurons_per_class
        end = min(start + neurons_per_class, num_neurons)
        class_map[c] = list(range(start, end))

    log.info("Extracted weights: shape=%s, %d classes x %d neurons",
             weights_2d.shape, NUM_CLASSES, neurons_per_class)
    return weights_2d, class_map


def inject_weights(edge_model, weights_2d: np.ndarray, num_features: int = NUM_FEATURES):
    """Inject merged weights into edge model, rebuilding if neuron count changed.

    Args:
        weights_2d: shape (num_neurons, features), will be reshaped to Akida format
        num_features: Feature dimension for rebuilding edge model if needed.

    Returns:
        edge_model (possibly rebuilt if neuron count differs)
    """
    num_neurons = weights_2d.shape[0]
    nf = weights_2d.shape[1]
    current_units = edge_model.get_layer(EDGE_LAYER_NAME).get_variable("weights").shape[-1]

    if num_neurons != current_units:
        log.info("Rebuilding edge model: %d -> %d neurons", current_units, num_neurons)
        edge_model = _build_edge_model(num_neurons, NUM_CLASSES, num_features=nf)

    # Reshape: (num_neurons, features) -> (1, 1, features, num_neurons)
    akida_weights = weights_2d.T.reshape(1, 1, nf, num_neurons)
    edge_model.get_layer(EDGE_LAYER_NAME).set_variable(
        "weights", akida_weights.astype(np.int8))

    log.info("Injected %d neurons into edge layer", num_neurons)
    return edge_model


# ---------------------------------------------------------------------------
# TCP socket protocol
# ---------------------------------------------------------------------------

def _send_msg(sock: socket.socket, data: bytes):
    sock.sendall(struct.pack("!I", len(data)) + data)


def _recv_msg(sock: socket.socket) -> bytes:
    raw_len = _recv_exact(sock, 4)
    if not raw_len:
        return b""
    msg_len = struct.unpack("!I", raw_len)[0]
    return _recv_exact(sock, msg_len)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        buf.extend(chunk)
    return bytes(buf)


def send_weights_to_peer(peer_ip: str, weights: np.ndarray, class_map: dict,
                         classes_seen: list[str], node_id: str):
    payload = pickle.dumps({
        "weights": weights,
        "class_map": class_map,
        "classes_seen": classes_seen,
        "node_id": node_id,
    })
    log.info("Sending %d bytes of weights to peer %s:%d",
             len(payload), peer_ip, EXCHANGE_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(30)
        s.connect((peer_ip, EXCHANGE_PORT))
        _send_msg(s, payload)


def receive_weights_from_peer(bind_ip: str) -> dict:
    log.info("Listening for peer weights on %s:%d ...", bind_ip, EXCHANGE_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((bind_ip, EXCHANGE_PORT))
        srv.listen(1)
        srv.settimeout(120)
        conn, addr = srv.accept()
        with conn:
            data = _recv_msg(conn)
    payload = pickle.loads(data)
    log.info("Received weights from %s: shape %s",
             payload["node_id"], payload["weights"].shape)
    return payload


# ---------------------------------------------------------------------------
# Command server
# ---------------------------------------------------------------------------

def run_command_server(node_id: str, data_dir: Path, model_dir: Path,
                       results_dir: Path, bind_ip: str):
    results_dir.mkdir(parents=True, exist_ok=True)

    log.info("Command server starting on 0.0.0.0:%d (weight exchange on %s:%d)",
             COMMAND_PORT, bind_ip, EXCHANGE_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", COMMAND_PORT))
        srv.listen(1)

        feat_model = None
        edge_model = None
        num_features = NUM_FEATURES  # Auto-detected from extractor
        wide_projection = None  # (W, b) tuple for wide feature extractors
        local_weights = None
        local_class_map = None
        local_classes = None
        bin_thresholds = None  # Per-feature binarization thresholds
        shared_thresholds = None  # Shared thresholds from orchestrator (overrides local)
        last_raw_features = None  # Cached raw features from last training
        binarization_method = "mean"  # Current binarization method

        while True:
            log.info("Waiting for command ...")
            conn, addr = srv.accept()
            with conn:
                raw = _recv_msg(conn)
                cmd = json.loads(raw.decode())
                action = cmd.get("action")
                log.info("Received command: %s from %s", action, addr)

                result = {}
                try:
                    if action == "build_model":
                        params = cmd.get("params", {})
                        # Update binarization method if specified
                        if "binarization_method" in params:
                            binarization_method = params["binarization_method"]
                            log.info("Binarization method set to: %s", binarization_method)
                        feat_model, edge_model, num_features, wide_projection = build_models(model_dir, params=params)
                        result = {"status": "ok", "message": "Models built (feat+edge)",
                                  "params": params, "num_features": num_features}

                    elif action == "train":
                        X_train = np.load(data_dir / f"{node_id}_X_train.npy")
                        y_train = np.load(data_dir / f"{node_id}_y_train.npy")
                        with open(data_dir / f"{node_id}_classes.json") as f:
                            local_classes = json.load(f)

                        # Override binarization method per-command if specified
                        bin_method = cmd.get("binarization_method", binarization_method)

                        # Use shared thresholds if available, otherwise compute local
                        if shared_thresholds is not None:
                            log.info("Using shared thresholds for training")
                            features_raw = extract_features(feat_model, X_train, num_features, wide_projection)
                            last_raw_features = features_raw
                            bin_thresholds = shared_thresholds
                            features_bin = binarize(features_raw, bin_thresholds)
                            ones_ratio = features_bin.mean()
                            t0 = time.perf_counter()
                            edge_model.fit(features_bin, y_train)
                            learn_ms = (time.perf_counter() - t0) * 1000
                            metrics = {
                                "learning_time_ms": learn_ms,
                                "samples_processed": len(X_train),
                                "num_classes": len(np.unique(y_train)),
                                "ones_ratio": float(ones_ratio),
                                "shared_thresholds": True,
                                "binarization_method": bin_method,
                            }
                        else:
                            metrics, bin_thresholds = run_edge_learning(
                                feat_model, edge_model, X_train, y_train,
                                num_features=num_features,
                                binarization_method=bin_method,
                                wide_projection=wide_projection)
                            # Cache raw features for potential save_features command
                            last_raw_features = extract_features(feat_model, X_train, num_features, wide_projection)

                        local_weights, local_class_map = extract_weights(edge_model)
                        result = {"status": "ok", "metrics": metrics}

                    elif action == "evaluate":
                        X_test = np.load(data_dir / "eval_X.npy")
                        y_test = np.load(data_dir / "eval_y.npy")
                        num_classes = cmd.get("num_classes", NUM_CLASSES)
                        metrics = evaluate_model(
                            feat_model, edge_model, X_test, y_test,
                            num_classes, bin_thresholds,
                            num_features=num_features,
                            wide_projection=wide_projection)
                        result = {"status": "ok", "metrics": metrics}

                    elif action == "get_weights":
                        payload = {
                            "weights": local_weights.tolist(),
                            "class_map": {str(k): v for k, v in local_class_map.items()},
                            "classes_seen": local_classes,
                            "node_id": node_id,
                        }
                        result = {"status": "ok", "payload": payload}

                    elif action == "set_weights":
                        w = np.array(cmd["weights"], dtype=np.int8)
                        edge_model = inject_weights(edge_model, w, num_features)
                        local_weights, local_class_map = extract_weights(edge_model)
                        result = {"status": "ok", "message": f"Injected {w.shape[0]} neurons"}

                    elif action == "set_thresholds":
                        # Receive shared binarization thresholds from orchestrator
                        shared_thresholds = np.array(cmd["thresholds"], dtype=np.float32)
                        bin_thresholds = shared_thresholds
                        log.info("Set shared thresholds: shape=%s, mean=%.3f",
                                 shared_thresholds.shape, shared_thresholds.mean())
                        result = {"status": "ok", "message": "Shared thresholds set",
                                  "threshold_mean": float(shared_thresholds.mean())}

                    elif action == "clear_thresholds":
                        # Reset to local threshold mode for dual-regime experiments
                        shared_thresholds = None
                        bin_thresholds = None
                        log.info("Cleared shared thresholds, reverting to local mode")
                        result = {"status": "ok", "message": "Shared thresholds cleared"}

                    elif action == "extract_calibration_features":
                        # Run feature extractor on calibration samples,
                        # return raw int8 features (before binarization)
                        cal_path = data_dir / "calibration_X.npy"
                        if not cal_path.exists():
                            result = {"status": "error",
                                      "message": "Calibration data not found"}
                        else:
                            X_cal = np.load(cal_path)
                            raw_feats = extract_features(feat_model, X_cal, num_features, wide_projection)
                            result = {
                                "status": "ok",
                                "features": raw_feats.tolist(),
                                "shape": list(raw_feats.shape),
                            }

                    elif action == "save_features":
                        # Save binarized AND raw int8 features + labels for SCP retrieval
                        X_train = np.load(data_dir / f"{node_id}_X_train.npy")
                        y_train = np.load(data_dir / f"{node_id}_y_train.npy")

                        # Extract raw features if not cached
                        if last_raw_features is not None and len(last_raw_features) == len(X_train):
                            features_raw = last_raw_features
                        else:
                            features_raw = extract_features(feat_model, X_train, num_features, wide_projection)

                        features_bin = binarize(features_raw, bin_thresholds)
                        # Flatten from (N, 1, 1, D) to (N, D) for baseline use
                        features_flat = features_bin.reshape(len(features_bin), -1)

                        # Save training features (binarized + raw int8)
                        save_dir = results_dir
                        np.save(save_dir / f"{node_id}_train_features_bin.npy", features_flat)
                        np.save(save_dir / f"{node_id}_train_features_int8.npy", features_raw)
                        np.save(save_dir / f"{node_id}_train_labels.npy", y_train)

                        # Also save eval features (binarized + raw int8)
                        X_eval = np.load(data_dir / "eval_X.npy")
                        y_eval = np.load(data_dir / "eval_y.npy")
                        eval_raw = extract_features(feat_model, X_eval, num_features, wide_projection)
                        eval_bin = binarize(eval_raw, bin_thresholds).reshape(len(X_eval), -1)
                        np.save(save_dir / "eval_features_bin.npy", eval_bin)
                        np.save(save_dir / "eval_features_int8.npy", eval_raw)
                        np.save(save_dir / "eval_labels.npy", y_eval)

                        result = {
                            "status": "ok",
                            "message": f"Saved features to {save_dir}",
                            "train_shape": list(features_flat.shape),
                            "train_int8_shape": list(features_raw.shape),
                            "eval_shape": list(eval_bin.shape),
                        }

                    elif action == "retrain":
                        # Re-run STDP fit using current edge model weights as starting point.
                        # Used after federation to continue local refinement.
                        X_train = np.load(data_dir / f"{node_id}_X_train.npy")
                        y_train = np.load(data_dir / f"{node_id}_y_train.npy")

                        features_raw = extract_features(feat_model, X_train, num_features, wide_projection)
                        last_raw_features = features_raw
                        features_bin = binarize(features_raw, bin_thresholds)

                        t0 = time.perf_counter()
                        edge_model.fit(features_bin, y_train)
                        learn_ms = (time.perf_counter() - t0) * 1000

                        local_weights, local_class_map = extract_weights(edge_model)
                        ones_ratio = features_bin.mean()

                        log.info("Retrained: %.1f ms, ones_ratio=%.3f", learn_ms, ones_ratio)
                        result = {
                            "status": "ok",
                            "metrics": {
                                "learning_time_ms": learn_ms,
                                "samples_processed": len(X_train),
                                "ones_ratio": float(ones_ratio),
                            },
                        }

                    elif action == "exchange_weights":
                        peer_ip = cmd["peer_ip"]
                        role = cmd["role"]

                        if role == "receiver_first":
                            remote = receive_weights_from_peer(bind_ip)
                            send_weights_to_peer(
                                peer_ip, local_weights, local_class_map,
                                local_classes, node_id)
                        else:
                            send_weights_to_peer(
                                peer_ip, local_weights, local_class_map,
                                local_classes, node_id)
                            remote = receive_weights_from_peer(bind_ip)

                        result = {
                            "status": "ok",
                            "remote_node": remote["node_id"],
                            "remote_classes": remote["classes_seen"],
                            "remote_weights_shape": list(remote["weights"].shape),
                            "local_weights_bytes": local_weights.nbytes,
                            "remote_weights_bytes": remote["weights"].nbytes,
                        }

                    elif action == "shutdown":
                        result = {"status": "ok", "message": "Shutting down"}
                        _send_msg(conn, json.dumps(result).encode())
                        log.info("Shutdown command received, exiting")
                        return

                    else:
                        result = {"status": "error", "message": f"Unknown action: {action}"}

                except Exception as e:
                    log.exception("Error handling command %s", action)
                    result = {"status": "error", "message": str(e)}

                _send_msg(conn, json.dumps(result, default=str).encode())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Federated neuromorphic node worker")
    parser.add_argument("--node-id", required=True, choices=["claudio", "paolo"])
    parser.add_argument("--data-dir", type=Path,
                        default=Path.home() / "federated_experiment" / "data")
    parser.add_argument("--model-dir", type=Path,
                        default=Path.home() / "federated_experiment" / "models")
    parser.add_argument("--results-dir", type=Path,
                        default=Path.home() / "federated_experiment" / "results")
    parser.add_argument("--bind-ip", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{args.node_id}] %(asctime)s %(levelname)s %(message)s",
    )

    if args.bind_ip is None:
        args.bind_ip = "10.0.0.1" if args.node_id == "claudio" else "10.0.0.2"

    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    run_command_server(
        node_id=args.node_id,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
        bind_ip=args.bind_ip,
    )


if __name__ == "__main__":
    main()
