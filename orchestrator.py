"""Orchestrator — runs on Mac, coordinates the federated experiment across both Pis.

Responsibilities:
  1. Prepare and upload dataset + code to both Pis via SCP
  2. Launch node_worker.py on each Pi via SSH
  3. Send commands to trigger model build, training, weight exchange, evaluation
  4. Collect results and produce comparison table
  5. Multi-trial loop with seeded data splits
  6. Shared binarization thresholds via calibration set
  7. Multi-round federation with retraining
  8. Software baselines (linear + KNN) for comparison
"""

import json
import logging
import os
import pickle
import shutil
import socket
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

from . import config
from .data_loader import (
    download_speech_commands,
    prepare_calibration_dataset,
    prepare_eval_dataset,
    prepare_node_dataset,
)
from .federation import (
    NodeWeights,
    merge_weights,
    compute_communication_overhead,
    NOVEL_CLASSES,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSH / SCP helpers
# ---------------------------------------------------------------------------

def _ssh(node: dict, cmd: str, timeout: int = 300, node_name: str | None = None) -> str:
    """Run a command on a Pi via SSH, or locally if this is the orchestrator node."""
    if node_name and config.is_local_node(node_name):
        log.info("LOCAL: %s", cmd)
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            log.error("LOCAL stderr: %s", result.stderr)
        return result.stdout.strip()

    target_ip = config.get_ssh_target(node_name) if node_name else node["lan_ip"]
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
        f"{node['ssh_user']}@{target_ip}",
        cmd,
    ]
    log.info("SSH [%s]: %s", target_ip, cmd)
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log.error("SSH stderr: %s", result.stderr)
    return result.stdout.strip()


def _scp_to(node: dict, local_path: str, remote_path: str, node_name: str | None = None):
    """SCP a file or directory to a Pi, or local copy if this is the orchestrator node."""
    if node_name and config.is_local_node(node_name):
        dest = Path(remote_path.replace("~", str(Path.home())))
        src = Path(local_path).resolve()
        # If dest is a directory (or ends with /), resolve the actual target file
        if dest.is_dir() or remote_path.endswith("/"):
            dest.mkdir(parents=True, exist_ok=True)
            dest = dest / src.name
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
        dest = dest.resolve()
        if src == dest:
            log.info("LOCAL COPY: skip (src == dest) %s", src)
            return
        if src.is_dir():
            if dest.is_dir():
                shutil.rmtree(dest)
            shutil.copytree(str(src), str(dest))
        else:
            shutil.copy2(str(src), str(dest))
        log.info("LOCAL COPY: %s -> %s", local_path, dest)
        return

    target_ip = config.get_ssh_target(node_name) if node_name else node["lan_ip"]
    target = f"{node['ssh_user']}@{target_ip}:{remote_path}"
    cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-r", str(local_path), target]
    log.info("SCP %s -> %s", local_path, target)
    subprocess.run(cmd, check=True, timeout=120)


def _scp_from(node: dict, remote_path: str, local_path: str, node_name: str | None = None):
    """SCP a file from a Pi to local, or local copy if this is the orchestrator node."""
    if node_name and config.is_local_node(node_name):
        src = Path(remote_path.replace("~", str(Path.home()))).resolve()
        dest = Path(local_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src == dest:
            log.info("LOCAL COPY: skip (src == dest) %s", src)
            return
        shutil.copy2(str(src), str(dest))
        log.info("LOCAL COPY: %s -> %s", src, dest)
        return

    target_ip = config.get_ssh_target(node_name) if node_name else node["lan_ip"]
    source = f"{node['ssh_user']}@{target_ip}:{remote_path}"
    cmd = ["scp", "-o", "StrictHostKeyChecking=no", str(source), str(local_path)]
    log.info("SCP %s -> %s", source, local_path)
    subprocess.run(cmd, check=True, timeout=120)


# ---------------------------------------------------------------------------
# Command client — sends commands to node workers
# ---------------------------------------------------------------------------

def _send_msg(sock: socket.socket, data: bytes):
    sock.sendall(struct.pack("!I", len(data)) + data)


def _recv_msg(sock: socket.socket) -> bytes:
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("Connection closed")
        raw_len += chunk
    msg_len = struct.unpack("!I", raw_len)[0]
    buf = bytearray()
    while len(buf) < msg_len:
        chunk = sock.recv(msg_len - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while receiving")
        buf.extend(chunk)
    return bytes(buf)


def send_command(node_ip: str, command: dict, timeout: int = 300) -> dict:
    """Send a JSON command to a node worker and return the response."""
    log.info("Sending command to %s: %s", node_ip, command.get("action"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect((node_ip, config.COMMAND_PORT))
        _send_msg(s, json.dumps(command).encode())
        response = _recv_msg(s)
    result = json.loads(response.decode())
    if result.get("status") != "ok":
        log.error("Command failed: %s", result)
    return result


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_data(local_data_dir: Path, seed: int | None = None,
                 use_calibration: bool = False,
                 novel_classes: list[str] | None = None):
    """Download Speech Commands and prepare per-node datasets locally.

    Args:
        local_data_dir: Directory to store prepared data.
        seed: Random seed for reproducible data splits.
        use_calibration: If True, reserve calibration samples and adjust offsets.
        novel_classes: Override default novel classes.
    """
    log.info("=== Phase 0: Data Preparation (seed=%s) ===", seed)
    raw_dir = local_data_dir / "raw"
    download_speech_commands(raw_dir)

    if novel_classes is None:
        novel_classes = config.NOVEL_CLASSES

    cal_offset = config.CALIBRATION_SAMPLES_PER_CLASS if use_calibration else 0

    # Prepare calibration set if needed
    if use_calibration:
        X_cal, y_cal = prepare_calibration_dataset(
            raw_dir, novel_classes,
            samples_per_class=config.CALIBRATION_SAMPLES_PER_CLASS,
            seed=seed if seed is not None else 0,
        )
        np.save(local_data_dir / "calibration_X.npy", X_cal)
        np.save(local_data_dir / "calibration_y.npy", y_cal)
        log.info("Calibration set: %d samples", len(X_cal))

    # Prepare per-node few-shot splits
    for node_id in ("claudio", "paolo"):
        X, y, classes = prepare_node_dataset(
            raw_dir, node_id, seed=seed,
            novel_classes=novel_classes,
            calibration_offset=cal_offset,
        )
        np.save(local_data_dir / f"{node_id}_X_train.npy", X)
        np.save(local_data_dir / f"{node_id}_y_train.npy", y)
        with open(local_data_dir / f"{node_id}_classes.json", "w") as f:
            json.dump(classes, f)
        log.info("Node %s: %d samples, classes=%s", node_id, len(X), classes)

    # Prepare shared eval set (all novel classes)
    # Eval samples come after calibration + training samples
    train_offset = cal_offset + config.SPLIT.samples_per_class
    X_eval, y_eval, eval_classes = prepare_eval_dataset(
        raw_dir, seed=seed, novel_classes=novel_classes,
        train_offset=train_offset,
    )
    np.save(local_data_dir / "eval_X.npy", X_eval)
    np.save(local_data_dir / "eval_y.npy", y_eval)
    with open(local_data_dir / "eval_classes.json", "w") as f:
        json.dump(eval_classes, f)
    log.info("Eval set: %d samples, classes=%s", len(X_eval), eval_classes)


def deploy_to_nodes(local_data_dir: Path, upload_calibration: bool = False):
    """Upload experiment code and data to both Pis."""
    log.info("=== Phase 1: Deploy to Nodes ===")
    code_dir = config.MAC_PROJECT_DIR

    for name, node in config.NODES.items():
        # Create remote directories
        _ssh(node, "mkdir -p ~/federated_experiment/data ~/federated_experiment/models ~/federated_experiment/results",
             node_name=name)

        # Upload node worker
        _scp_to(node, str(code_dir / "node_worker.py"),
                "~/federated_experiment/node_worker.py", node_name=name)

        # Upload node-specific training data
        _scp_to(node, str(local_data_dir / f"{name}_X_train.npy"),
                "~/federated_experiment/data/", node_name=name)
        _scp_to(node, str(local_data_dir / f"{name}_y_train.npy"),
                "~/federated_experiment/data/", node_name=name)
        _scp_to(node, str(local_data_dir / f"{name}_classes.json"),
                "~/federated_experiment/data/", node_name=name)

        # Upload shared eval data
        _scp_to(node, str(local_data_dir / "eval_X.npy"),
                "~/federated_experiment/data/", node_name=name)
        _scp_to(node, str(local_data_dir / "eval_y.npy"),
                "~/federated_experiment/data/", node_name=name)

        # Upload calibration data if needed
        if upload_calibration:
            cal_path = local_data_dir / "calibration_X.npy"
            if cal_path.exists():
                _scp_to(node, str(cal_path), "~/federated_experiment/data/",
                         node_name=name)

        log.info("Deployed to %s", name)


# ---------------------------------------------------------------------------
# Worker lifecycle
# ---------------------------------------------------------------------------

def launch_workers():
    """Start node_worker.py on each Pi in the background via SSH."""
    log.info("=== Phase 2: Launch Workers ===")
    procs = {}
    for name, node in config.NODES.items():
        # Kill any leftover worker first
        _ssh(node, f"kill $(pgrep -f 'node_worker.py --node-id {name}') 2>/dev/null || true",
             node_name=name)
        time.sleep(1)

        # Launch worker fully detached using setsid + fd redirection
        remote_cmd = (
            f"setsid bash -c '"
            f"source ~/akida-env/bin/activate && "
            f"cd ~/federated_experiment && "
            f"exec python3 node_worker.py --node-id {name}"
            f"' > ~/federated_experiment/worker.log 2>&1 < /dev/null &"
        )
        _ssh(node, remote_cmd, node_name=name)
        log.info("Worker launched on %s", name)
        procs[name] = node

    # Wait for workers to be ready
    log.info("Waiting for workers to start ...")
    time.sleep(5)

    for name, node in procs.items():
        cmd_ip = config.get_command_ip(name)
        for attempt in range(10):
            try:
                result = send_command(cmd_ip, {"action": "build_model"})
                if result["status"] == "ok":
                    log.info("Worker %s is ready", name)
                    break
            except (ConnectionRefusedError, OSError):
                log.info("Worker %s not ready yet, retrying ...", name)
                time.sleep(3)
        else:
            raise RuntimeError(f"Worker {name} failed to start")


def shutdown_workers():
    """Send shutdown command to all node workers."""
    log.info("=== Shutting down workers ===")
    for name, node in config.NODES.items():
        try:
            send_command(config.get_command_ip(name), {"action": "shutdown"}, timeout=10)
        except Exception:
            log.warning("Could not cleanly shut down %s", name)


# ---------------------------------------------------------------------------
# Experiment phases (unchanged from original, used within run_trial)
# ---------------------------------------------------------------------------

def run_local_training() -> dict:
    """Trigger few-shot edge learning on each Pi."""
    log.info("=== Local Edge Learning ===")
    results = {}
    for name, node in config.NODES.items():
        result = send_command(
            config.get_command_ip(name),
            {"action": "train"},
            timeout=120,
        )
        results[name] = result.get("metrics", {})
        log.info("Node %s training: %s", name, results[name])
    return results


def run_individual_evaluation() -> dict:
    """Evaluate each node on the full test set WITHOUT federation."""
    log.info("=== Individual Evaluation (no federation) ===")
    results = {}
    for name, node in config.NODES.items():
        result = send_command(
            config.get_command_ip(name),
            {"action": "evaluate", "num_classes": config.NUM_NEW_CLASSES},
        )
        results[name] = result.get("metrics", {})
        log.info("Node %s individual accuracy: %s", name, results[name])
    return results


def _get_node_weights() -> dict:
    """Fetch weights from both nodes, returning NodeWeights objects."""
    claudio_resp = send_command(config.get_command_ip("claudio"), {"action": "get_weights"})
    paolo_resp = send_command(config.get_command_ip("paolo"), {"action": "get_weights"})

    claudio_payload = claudio_resp["payload"]
    paolo_payload = paolo_resp["payload"]

    claudio_weights = NodeWeights(
        weights=np.array(claudio_payload["weights"], dtype=np.int8),
        class_map={int(k): v for k, v in claudio_payload["class_map"].items()},
        node_id="claudio",
        classes_seen=claudio_payload["classes_seen"],
    )
    paolo_weights = NodeWeights(
        weights=np.array(paolo_payload["weights"], dtype=np.int8),
        class_map={int(k): v for k, v in paolo_payload["class_map"].items()},
        node_id="paolo",
        classes_seen=paolo_payload["classes_seen"],
    )

    return {"claudio": claudio_weights, "paolo": paolo_weights}


def run_weight_exchange() -> dict:
    """Trigger weight exchange between the two Pis over direct Ethernet."""
    log.info("=== Weight Exchange ===")
    weights = _get_node_weights()
    comm_overhead = compute_communication_overhead(weights["claudio"], weights["paolo"])
    log.info("Communication overhead: %s", comm_overhead)
    return {
        "claudio_weights": weights["claudio"],
        "paolo_weights": weights["paolo"],
        "comm_overhead": comm_overhead,
    }


def _apply_strategy_and_evaluate(
    strategy: str,
    claudio_weights: NodeWeights,
    paolo_weights: NodeWeights,
    novel_classes: list[str] | None = None,
) -> dict:
    """Apply a single federation strategy on both nodes and evaluate."""
    results = {}
    for node_name, local_w, remote_w in [
        ("claudio", claudio_weights, paolo_weights),
        ("paolo", paolo_weights, claudio_weights),
    ]:
        merged, class_map = merge_weights(local_w, remote_w, strategy,
                                          novel_classes=novel_classes)
        cmd_ip = config.get_command_ip(node_name)
        send_command(cmd_ip, {
            "action": "set_weights",
            "weights": merged.tolist(),
        })
        eval_result = send_command(cmd_ip, {
            "action": "evaluate",
            "num_classes": config.NUM_NEW_CLASSES,
        })
        results[node_name] = eval_result.get("metrics", {})
        results[node_name]["all_classes"] = novel_classes or NOVEL_CLASSES
        log.info("  %s on %s: acc=%.3f", strategy, node_name,
                 results[node_name].get("accuracy", 0))
    return results


def run_federated_evaluation(exchange_data: dict,
                             novel_classes: list[str] | None = None) -> dict:
    """Apply each federation strategy and evaluate on both nodes."""
    log.info("=== Federated Evaluation ===")
    claudio_weights = exchange_data["claudio_weights"]
    paolo_weights = exchange_data["paolo_weights"]

    strategies = [s for s in config.FEDERATION_STRATEGIES if s != "individual"]
    results = {}
    for strategy in strategies:
        log.info("--- Strategy: %s ---", strategy)
        strat_results = _apply_strategy_and_evaluate(
            strategy, claudio_weights, paolo_weights, novel_classes)
        for node_name, metrics in strat_results.items():
            results[f"{strategy}_{node_name}"] = metrics
    return results


# ---------------------------------------------------------------------------
# New: Shared thresholds via calibration
# ---------------------------------------------------------------------------

def compute_shared_thresholds() -> np.ndarray:
    """Collect calibration features from both nodes and compute shared thresholds.

    Requires calibration_X.npy to have been uploaded to both nodes.
    Returns per-feature mean threshold (shape (64,)).
    """
    log.info("=== Computing Shared Thresholds ===")
    all_features = []
    for name, node in config.NODES.items():
        resp = send_command(
            config.get_command_ip(name),
            {"action": "extract_calibration_features"},
            timeout=120,
        )
        if resp.get("status") != "ok":
            raise RuntimeError(f"Calibration feature extraction failed on {name}: {resp}")
        feats = np.array(resp["features"], dtype=np.float32)
        log.info("Node %s calibration features: shape=%s", name, feats.shape)
        all_features.append(feats)

    # Pool features from both nodes and compute per-feature mean
    combined = np.concatenate(all_features, axis=0)
    thresholds = combined.mean(axis=0)
    log.info("Shared thresholds computed: shape=%s, mean=%.3f",
             thresholds.shape, thresholds.mean())
    return thresholds


def distribute_shared_thresholds(thresholds: np.ndarray):
    """Send shared binarization thresholds to both nodes."""
    log.info("=== Distributing Shared Thresholds ===")
    for name, node in config.NODES.items():
        resp = send_command(config.get_command_ip(name), {
            "action": "set_thresholds",
            "thresholds": thresholds.tolist(),
        })
        log.info("Node %s threshold response: %s", name, resp.get("message"))


def clear_node_thresholds():
    """Clear shared thresholds on all nodes, reverting to local mode."""
    log.info("=== Clearing Shared Thresholds ===")
    for name, node in config.NODES.items():
        resp = send_command(config.get_command_ip(name), {"action": "clear_thresholds"})
        log.info("Node %s: %s", name, resp.get("message"))


# ---------------------------------------------------------------------------
# New: Feature collection for baselines
# ---------------------------------------------------------------------------

def collect_binarized_features(local_results_dir: Path) -> dict:
    """Tell nodes to save binarized features, then SCP them back.

    Returns dict with per-node train features and shared eval features.
    """
    log.info("=== Collecting Binarized Features ===")
    local_results_dir.mkdir(parents=True, exist_ok=True)

    for name, node in config.NODES.items():
        # Tell node to save features
        resp = send_command(config.get_command_ip(name), {"action": "save_features"}, timeout=120)
        if resp.get("status") != "ok":
            log.error("save_features failed on %s: %s", name, resp)
            continue
        log.info("Node %s saved features: train=%s, eval=%s",
                 name, resp.get("train_shape"), resp.get("eval_shape"))

        # SCP back
        for fname in [f"{name}_train_features_bin.npy", f"{name}_train_labels.npy",
                      "eval_features_bin.npy", "eval_labels.npy"]:
            try:
                _scp_from(node, f"~/federated_experiment/results/{fname}",
                          str(local_results_dir / fname), node_name=name)
            except subprocess.CalledProcessError:
                log.warning("Could not SCP %s from %s", fname, name)

    # Load and return
    result = {}
    for name in ("claudio", "paolo"):
        train_X_path = local_results_dir / f"{name}_train_features_bin.npy"
        train_y_path = local_results_dir / f"{name}_train_labels.npy"
        if train_X_path.exists() and train_y_path.exists():
            result[name] = {
                "train_X": np.load(train_X_path),
                "train_y": np.load(train_y_path),
            }

    eval_X_path = local_results_dir / "eval_features_bin.npy"
    eval_y_path = local_results_dir / "eval_labels.npy"
    if eval_X_path.exists() and eval_y_path.exists():
        result["eval_X"] = np.load(eval_X_path)
        result["eval_y"] = np.load(eval_y_path)

    return result


def run_baselines(features_data: dict) -> dict:
    """Run software baselines on collected binarized features.

    Returns dict with results for linear and KNN baselines.
    Skipped when running on Pi (no PyTorch available).
    """
    if config.PI_ORCHESTRATOR_NODE is not None:
        log.info("Skipping baselines (running on Pi, no PyTorch)")
        return {}

    from .baselines import run_all_baselines

    if "eval_X" not in features_data or len(features_data) < 3:
        log.warning("Insufficient feature data for baselines")
        return {}

    node_features = {
        name: features_data[name]
        for name in ("claudio", "paolo")
        if name in features_data
    }

    return run_all_baselines(
        node_features=node_features,
        eval_features=features_data["eval_X"],
        eval_labels=features_data["eval_y"],
        num_classes=config.NUM_NEW_CLASSES,
        epochs=config.BASELINE_EPOCHS,
        lr=config.BASELINE_LR,
        hidden_size=config.MLP_HIDDEN_SIZE,
        weight_decay=config.BASELINE_WEIGHT_DECAY,
    )


# ---------------------------------------------------------------------------
# New: Multi-round federation with retraining
# ---------------------------------------------------------------------------

def run_multi_round_federation(
    num_rounds: int,
    exchange_data: dict,
    novel_classes: list[str] | None = None,
    retraining_strategy: str = "fedavg",
) -> dict:
    """Run multiple federation rounds with retraining between rounds.

    Each round:
      1. Apply each strategy, evaluate
      2. Retrain on local data using ``retraining_strategy``-merged weights
      3. Re-extract weights for next round

    Args:
        retraining_strategy: Which federation strategy to use when injecting
            weights before retraining.  Default ``"fedavg"`` for backwards
            compatibility; ``"fedunion"`` keeps neuron prototypes intact.

    Returns dict keyed by round number, each containing per-strategy results.
    """
    log.info("=== Multi-Round Federation (%d rounds, retrain=%s) ===",
             num_rounds, retraining_strategy)
    claudio_weights = exchange_data["claudio_weights"]
    paolo_weights = exchange_data["paolo_weights"]

    strategies = [s for s in config.FEDERATION_STRATEGIES if s != "individual"]
    all_rounds = {}

    for round_num in range(1, num_rounds + 1):
        log.info("--- Round %d/%d ---", round_num, num_rounds)
        round_results = {}

        for strategy in strategies:
            log.info("  Strategy: %s", strategy)
            strat_results = _apply_strategy_and_evaluate(
                strategy, claudio_weights, paolo_weights, novel_classes)
            round_results[strategy] = strat_results

        all_rounds[f"round_{round_num}"] = round_results

        # After evaluating all strategies for this round, retrain using the
        # chosen retraining strategy's merged weights for the next round
        if round_num < num_rounds:
            log.info("  Retraining with %s weights for next round ...",
                     retraining_strategy)
            for node_name, local_w, remote_w in [
                ("claudio", claudio_weights, paolo_weights),
                ("paolo", paolo_weights, claudio_weights),
            ]:
                merged, _ = merge_weights(local_w, remote_w,
                                          retraining_strategy,
                                          novel_classes=novel_classes)
                cmd_ip = config.get_command_ip(node_name)
                send_command(cmd_ip, {
                    "action": "set_weights",
                    "weights": merged.tolist(),
                })
                retrain_resp = send_command(
                    cmd_ip, {"action": "retrain"}, timeout=120)
                log.info("  %s retrained: %s", node_name,
                         retrain_resp.get("metrics", {}))

            weights = _get_node_weights()
            claudio_weights = weights["claudio"]
            paolo_weights = weights["paolo"]

    return all_rounds


# ---------------------------------------------------------------------------
# Single trial execution
# ---------------------------------------------------------------------------

def run_trial(
    seed: int,
    data_dir: Path,
    results_dir: Path,
    use_shared_thresholds: bool = False,
    num_rounds: int = 1,
    run_software_baselines: bool = False,
    novel_classes: list[str] | None = None,
    skip_data: bool = False,
    skip_deploy: bool = False,
    compare_retraining: bool = False,
) -> dict:
    """Execute a single trial of the experiment with a given seed.

    Returns a dict with all metrics for this trial.
    """
    log.info("========== TRIAL (seed=%d) ==========", seed)

    # 1. Regenerate data splits with this seed
    if not skip_data:
        prepare_data(data_dir, seed=seed,
                     use_calibration=use_shared_thresholds,
                     novel_classes=novel_classes)

    # 2. Deploy to nodes
    if not skip_deploy:
        deploy_to_nodes(data_dir, upload_calibration=use_shared_thresholds)

    # 2.5. Rebuild edge model for fresh start (ensures trials are independent)
    for name, node in config.NODES.items():
        send_command(config.get_command_ip(name), {"action": "build_model"})

    # 2.6. Clear or set shared thresholds
    if use_shared_thresholds:
        thresholds = compute_shared_thresholds()
        distribute_shared_thresholds(thresholds)
    else:
        # Ensure local threshold mode (clear any leftover shared thresholds)
        clear_node_thresholds()

    # 4. Local training
    training_metrics = run_local_training()

    # 5. Individual evaluation (before federation)
    individual_results = run_individual_evaluation()

    # 6. Collect binarized features for baselines (if enabled)
    baseline_results = {}
    if run_software_baselines:
        features_data = collect_binarized_features(results_dir)
        baseline_results = run_baselines(features_data)

    # 7. Weight exchange
    exchange_data = run_weight_exchange()

    # 8. Federation evaluation (single or multi-round)
    strategies = [s for s in config.FEDERATION_STRATEGIES if s != "individual"]
    if num_rounds > 1:
        federated_results = run_multi_round_federation(
            num_rounds, exchange_data, novel_classes=novel_classes)
    else:
        # Single round — wrap in round_1 for consistent format
        raw_fed = run_federated_evaluation(exchange_data, novel_classes=novel_classes)
        federated_results = {"round_1": {}}
        for strategy in strategies:
            federated_results["round_1"][strategy] = {
                "claudio": raw_fed.get(f"{strategy}_claudio", {}),
                "paolo": raw_fed.get(f"{strategy}_paolo", {}),
            }

    # 9. Optional: second multi-round run with FedUnion-based retraining
    fedunion_retrain_results = {}
    if compare_retraining and num_rounds > 1:
        log.info("=== FedUnion Retraining Comparison ===")
        # Reset: rebuild edge models and retrain from scratch for a fair comparison
        for name, node in config.NODES.items():
            send_command(config.get_command_ip(name), {"action": "build_model"})
        if use_shared_thresholds:
            distribute_shared_thresholds(thresholds)
        else:
            clear_node_thresholds()
        run_local_training()
        exchange_data_2 = run_weight_exchange()
        fedunion_retrain_results = run_multi_round_federation(
            num_rounds, exchange_data_2,
            novel_classes=novel_classes,
            retraining_strategy="fedunion",
        )

    trial_result = {
        "seed": seed,
        "training": training_metrics,
        "individual": individual_results,
        "federated": federated_results,
        "communication": exchange_data["comm_overhead"],
        "shared_thresholds": use_shared_thresholds,
    }
    if baseline_results:
        trial_result["baselines"] = baseline_results
    if fedunion_retrain_results:
        trial_result["federated_union_retrain"] = fedunion_retrain_results

    return trial_result


# ---------------------------------------------------------------------------
# Results formatting
# ---------------------------------------------------------------------------

def format_results_table(
    training_metrics: dict,
    individual_results: dict,
    federated_results: dict,
    comm_overhead: dict,
) -> str:
    """Format a comparison table of all results."""
    lines = []
    lines.append("=" * 80)
    lines.append("FEDERATED NEUROMORPHIC FEW-SHOT LEARNING — RESULTS")
    lines.append("=" * 80)

    # Training metrics
    lines.append("\n--- Training Metrics ---")
    lines.append(f"{'Node':<12} {'Samples':>8} {'Classes':>8} {'Time (ms)':>12}")
    lines.append("-" * 44)
    for name in ("claudio", "paolo"):
        m = training_metrics.get(name, {})
        lines.append(
            f"{name:<12} {m.get('samples_processed', 'N/A'):>8} "
            f"{m.get('num_classes', 'N/A'):>8} "
            f"{m.get('learning_time_ms', 'N/A'):>12.1f}"
        )

    # Individual accuracy
    lines.append("\n--- Individual Accuracy (no federation) ---")
    lines.append(f"{'Node':<12} {'Accuracy':>10} {'Correct':>8} {'Total':>8} {'Inf. Time':>12}")
    lines.append("-" * 54)
    for name in ("claudio", "paolo"):
        m = individual_results.get(name, {})
        acc = m.get("accuracy", 0) * 100
        lines.append(
            f"{name:<12} {acc:>9.1f}% {m.get('correct', 'N/A'):>8} "
            f"{m.get('total', 'N/A'):>8} {m.get('inference_time_ms', 0):>11.1f}ms"
        )

    # Federated accuracy
    lines.append("\n--- Federated Accuracy ---")
    lines.append(f"{'Strategy':<12} {'Node':<12} {'Accuracy':>10} {'Correct':>8} {'Total':>8}")
    lines.append("-" * 54)
    for strategy in ("fedavg", "fedunion", "fedbest"):
        for name in ("claudio", "paolo"):
            key = f"{strategy}_{name}"
            m = federated_results.get(key, {})
            acc = m.get("accuracy", 0) * 100
            lines.append(
                f"{strategy:<12} {name:<12} {acc:>9.1f}% "
                f"{m.get('correct', 'N/A'):>8} {m.get('total', 'N/A'):>8}"
            )

    # Communication overhead
    lines.append("\n--- Communication Overhead ---")
    lines.append(f"  Claudio -> Paolo:  {comm_overhead.get('local_to_remote_bytes', 0):>10,} bytes")
    lines.append(f"  Paolo -> Claudio:  {comm_overhead.get('remote_to_local_bytes', 0):>10,} bytes")
    lines.append(f"  Total exchanged:  {comm_overhead.get('total_bytes', 0):>10,} bytes")

    # Power estimates
    lines.append("\n--- Estimated Power Consumption ---")
    for name in ("claudio", "paolo"):
        m = training_metrics.get(name, {})
        learn_ms = m.get("learning_time_ms", 0)
        learn_energy_mj = learn_ms * config.AKIDA_LEARNING_POWER_MW / 1000
        inf_ms = individual_results.get(name, {}).get("inference_time_ms", 0)
        inf_energy_mj = inf_ms * config.AKIDA_INFERENCE_POWER_MW / 1000
        lines.append(
            f"  {name}: learning={learn_energy_mj:.2f} mJ, "
            f"inference={inf_energy_mj:.2f} mJ"
        )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def format_multi_trial_summary(all_results: dict) -> str:
    """Format a summary table across multiple trials with mean +/- std."""
    trials = all_results.get("trials", [])
    if not trials:
        return "No trial data available."

    lines = []
    lines.append("=" * 80)
    lines.append(f"MULTI-TRIAL SUMMARY ({len(trials)} trials)")
    lines.append("=" * 80)

    # Collect individual accuracies
    for name in ("claudio", "paolo"):
        accs = [t["individual"].get(name, {}).get("accuracy", 0) for t in trials]
        lines.append(f"  {name} individual: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%")

    # Collect federated accuracies (round 1)
    strategies = [s for s in config.FEDERATION_STRATEGIES if s != "individual"]
    for strategy in strategies:
        for name in ("claudio", "paolo"):
            accs = []
            for t in trials:
                fed = t.get("federated", {})
                r1 = fed.get("round_1", {})
                accs.append(r1.get(strategy, {}).get(name, {}).get("accuracy", 0))
            lines.append(
                f"  {strategy}/{name}: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%"
            )

    # Baseline summary if available
    has_baselines = any("baselines" in t for t in trials)
    if has_baselines:
        lines.append("\n--- Baselines ---")
        for bl_type in ("linear_individual", "linear_fedavg", "mlp_individual",
                        "mlp_fedavg", "knn_individual", "knn_fedavg"):
            accs = []
            for t in trials:
                bl = t.get("baselines", {}).get(bl_type, {})
                if isinstance(bl, dict) and "accuracy" in bl:
                    accs.append(bl["accuracy"])
                elif isinstance(bl, dict):
                    # Per-node dict
                    for node_bl in bl.values():
                        if isinstance(node_bl, dict) and "accuracy" in node_bl:
                            accs.append(node_bl["accuracy"])
            if accs:
                lines.append(f"  {bl_type}: {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%")

    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main experiment entry points
# ---------------------------------------------------------------------------

def run_full_experiment(
    data_dir: Path | None = None,
    skip_data: bool = False,
    skip_deploy: bool = False,
    num_trials: int = 1,
    num_rounds: int = 1,
    use_shared_thresholds: bool = False,
    run_software_baselines: bool = False,
    class_set: str | None = None,
    seeds: list[int] | None = None,
    compare_retraining: bool = False,
):
    """Execute the complete federated experiment pipeline.

    Supports single-trial (legacy) and multi-trial modes.
    """
    if data_dir is None:
        data_dir = config.MAC_PROJECT_DIR / "data"

    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir = config.MAC_PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve class set
    novel_classes = None
    if class_set and class_set in config.PRETRAINED_CLASS_SETS:
        cs = config.PRETRAINED_CLASS_SETS[class_set]
        novel_classes = [cs["shared"], cs["claudio_exclusive"], cs["paolo_exclusive"]]
        log.info("Using class set '%s': %s", class_set, novel_classes)

    # Resolve seeds
    if seeds is None:
        seeds = config.SEEDS[:num_trials]

    # Single-trial legacy mode (backwards compatible)
    if num_trials == 1 and not use_shared_thresholds and num_rounds == 1 and not run_software_baselines:
        if not skip_data:
            prepare_data(data_dir)
        if not skip_deploy:
            deploy_to_nodes(data_dir)

        launch_workers()
        try:
            training_metrics = run_local_training()
            individual_results = run_individual_evaluation()
            exchange_data = run_weight_exchange()
            federated_results = run_federated_evaluation(exchange_data)

            table = format_results_table(
                training_metrics, individual_results,
                federated_results, exchange_data["comm_overhead"],
            )
            print(table)

            all_results = {
                "training": training_metrics,
                "individual": individual_results,
                "federated": {k: {kk: vv for kk, vv in v.items() if kk != "all_classes"}
                              for k, v in federated_results.items()},
                "communication": exchange_data["comm_overhead"],
            }
            results_path = results_dir / "experiment_results.json"
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            log.info("Results saved to %s", results_path)
            return all_results
        finally:
            shutdown_workers()

    # Multi-trial mode
    log.info("Starting multi-trial experiment: %d trials, %d rounds, "
             "shared_thresholds=%s, baselines=%s",
             num_trials, num_rounds, use_shared_thresholds, run_software_baselines)

    # Launch workers once (they persist across trials since data gets re-uploaded)
    # For first trial, prepare + deploy + launch
    first_seed = seeds[0]
    prepare_data(data_dir, seed=first_seed,
                 use_calibration=use_shared_thresholds,
                 novel_classes=novel_classes)
    deploy_to_nodes(data_dir, upload_calibration=use_shared_thresholds)
    launch_workers()

    all_trials = []
    try:
        for trial_idx, seed in enumerate(seeds):
            log.info("=" * 60)
            log.info("TRIAL %d/%d (seed=%d)", trial_idx + 1, num_trials, seed)
            log.info("=" * 60)

            # Skip data prep + deploy for first trial (already done above)
            sd = trial_idx == 0
            trial_result = run_trial(
                seed=seed,
                data_dir=data_dir,
                results_dir=results_dir,
                use_shared_thresholds=use_shared_thresholds,
                num_rounds=num_rounds,
                run_software_baselines=run_software_baselines,
                novel_classes=novel_classes,
                skip_data=sd,
                skip_deploy=sd,
                compare_retraining=compare_retraining,
            )
            trial_result["trial"] = trial_idx
            all_trials.append(trial_result)

            # Save incremental results
            experiment_results = {
                "experiment_config": {
                    "num_trials": num_trials,
                    "num_rounds": num_rounds,
                    "shared_thresholds": use_shared_thresholds,
                    "run_baselines": run_software_baselines,
                    "class_set": class_set,
                    "seeds": seeds,
                },
                "trials": all_trials,
            }
            results_path = results_dir / "multi_trial_results.json"
            with open(results_path, "w") as f:
                json.dump(experiment_results, f, indent=2, default=str)
            log.info("Incremental results saved to %s", results_path)

        # Print summary
        summary = format_multi_trial_summary(experiment_results)
        print(summary)

        return experiment_results

    finally:
        shutdown_workers()


def run_comparison_experiment(
    data_dir: Path | None = None,
    num_trials: int = 10,
    num_rounds: int = 5,
    run_software_baselines: bool = True,
    class_set: str | None = None,
    seeds: list[int] | None = None,
    compare_retraining: bool = False,
):
    """Run experiment with both local and shared threshold regimes for comparison.

    Executes two full experiment runs (local thresholds, then shared thresholds)
    with the same seeds, and combines the results.
    """
    if data_dir is None:
        data_dir = config.MAC_PROJECT_DIR / "data"

    results_dir = config.MAC_PROJECT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = config.SEEDS[:num_trials]

    log.info("=" * 60)
    log.info("COMPARISON EXPERIMENT: local vs shared thresholds")
    log.info("  %d trials, %d rounds, class_set=%s", num_trials, num_rounds, class_set)
    log.info("=" * 60)

    # Run with local thresholds
    log.info("========== REGIME: LOCAL THRESHOLDS ==========")
    local_results = run_full_experiment(
        data_dir=data_dir,
        num_trials=num_trials,
        num_rounds=num_rounds,
        use_shared_thresholds=False,
        run_software_baselines=run_software_baselines,
        class_set=class_set,
        seeds=seeds,
        compare_retraining=compare_retraining,
    )

    # Run with shared thresholds
    log.info("========== REGIME: SHARED THRESHOLDS ==========")
    shared_results = run_full_experiment(
        data_dir=data_dir,
        num_trials=num_trials,
        num_rounds=num_rounds,
        use_shared_thresholds=True,
        run_software_baselines=run_software_baselines,
        class_set=class_set,
        seeds=seeds,
        compare_retraining=compare_retraining,
    )

    # Combine results
    combined = {
        "experiment_config": {
            "comparison_mode": True,
            "num_trials": num_trials,
            "num_rounds": num_rounds,
            "class_set": class_set,
            "seeds": seeds,
        },
        "local_thresholds": local_results,
        "shared_thresholds": shared_results,
    }

    results_path = results_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    log.info("Comparison results saved to %s", results_path)

    return combined
