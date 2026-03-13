#!/usr/bin/env python3
"""Numpy-only software baselines for running on Raspberry Pi (no PyTorch).

Provides k-NN, linear classifier, and MLP baselines that operate on the same
binarized or int8 features as the Akida STDP edge models. Designed to run
on Pi where PyTorch is not installed — only requires numpy and scikit-learn.

Usage (standalone test):
    python baselines_pi.py --features-dir ./results --num-classes 3

Integration:
    Called from comprehensive_sweep.py after each trial's STDP evaluation.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numpy utilities
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along last axis."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot encoding."""
    oh = np.zeros((len(labels), num_classes), dtype=np.float32)
    oh[np.arange(len(labels)), labels] = 1.0
    return oh


def _accuracy_from_preds(preds: np.ndarray, labels: np.ndarray,
                         num_classes: int) -> dict:
    """Compute accuracy dict matching existing baselines.py format."""
    correct = int((preds == labels).sum())
    accuracy = correct / len(labels) if len(labels) > 0 else 0.0

    per_class = {}
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() > 0:
            per_class[str(cls)] = float((preds[mask] == cls).sum() / mask.sum())

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(labels),
        "per_class_accuracy": per_class,
    }


# ---------------------------------------------------------------------------
# k-NN baseline (scikit-learn, already available on Pi)
# ---------------------------------------------------------------------------

def knn_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    k: int = 5,
    num_classes: int = 3,
    metric: str = "hamming",
) -> dict:
    """k-NN classification on binary or int8 features.

    Args:
        metric: "hamming" for binary features, "euclidean" for int8/float features.
    """
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(
        n_neighbors=min(k, len(train_labels)),
        metric=metric,
    )
    clf.fit(train_features, train_labels)
    preds = clf.predict(eval_features)

    result = _accuracy_from_preds(preds, eval_labels, num_classes)
    result["k"] = k
    result["metric"] = metric
    log.info("KNN (k=%d, %s): %.1f%% accuracy (%d/%d)",
             k, metric, result["accuracy"] * 100,
             result["correct"], result["total"])
    return result


def knn_fedavg(
    train_features_a: np.ndarray,
    train_labels_a: np.ndarray,
    train_features_b: np.ndarray,
    train_labels_b: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    k: int = 5,
    num_classes: int = 3,
    metric: str = "hamming",
) -> dict:
    """Federated KNN: pool training features from both nodes."""
    merged_features = np.concatenate([train_features_a, train_features_b], axis=0)
    merged_labels = np.concatenate([train_labels_a, train_labels_b], axis=0)
    return knn_baseline(merged_features, merged_labels,
                        eval_features, eval_labels,
                        k=k, num_classes=num_classes, metric=metric)


# ---------------------------------------------------------------------------
# Numpy linear classifier
# ---------------------------------------------------------------------------

class NumpyLinear:
    """Single linear layer: features @ W + b -> softmax -> cross-entropy."""

    def __init__(self, n_features: int, n_classes: int):
        self.W = np.random.randn(n_features, n_classes).astype(np.float32) * 0.01
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.n_features = n_features
        self.n_classes = n_classes

    def forward(self, X: np.ndarray) -> np.ndarray:
        return X @ self.W + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=-1)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 50, lr: float = 0.01) -> float:
        """Mini-batch SGD with linear LR decay. Returns final loss."""
        X = X.astype(np.float32)
        n = len(y)
        final_loss = 0.0

        for epoch in range(epochs):
            # Forward
            logits = self.forward(X)
            probs = _softmax(logits)

            # Cross-entropy loss
            log_probs = np.log(probs[np.arange(n), y] + 1e-8)
            final_loss = -log_probs.mean()

            # Gradient of softmax cross-entropy
            grad_logits = probs.copy()
            grad_logits[np.arange(n), y] -= 1.0
            grad_logits /= n

            # LR schedule (linear decay)
            current_lr = lr * (1.0 - epoch / epochs)

            # Update
            self.W -= current_lr * (X.T @ grad_logits)
            self.b -= current_lr * grad_logits.sum(axis=0)

        return float(final_loss)

    def get_params(self) -> dict:
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_params(self, params: dict):
        self.W = params["W"].copy()
        self.b = params["b"].copy()


def train_linear_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int = 3,
    epochs: int = 50,
    lr: float = 0.01,
) -> NumpyLinear:
    """Train a numpy linear classifier."""
    model = NumpyLinear(train_features.shape[1], num_classes)
    loss = model.train(train_features, train_labels, epochs=epochs, lr=lr)
    log.info("Linear baseline trained: %d samples, %d epochs, loss=%.4f",
             len(train_labels), epochs, loss)
    return model


def evaluate_linear_baseline(
    model: NumpyLinear,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
) -> dict:
    """Evaluate a numpy linear classifier."""
    preds = model.predict(eval_features.astype(np.float32))
    return _accuracy_from_preds(preds, eval_labels, model.n_classes)


def fedavg_linear(model_a: NumpyLinear, model_b: NumpyLinear) -> NumpyLinear:
    """FedAvg two linear classifiers by averaging parameters."""
    merged = NumpyLinear(model_a.n_features, model_a.n_classes)
    pa, pb = model_a.get_params(), model_b.get_params()
    merged.set_params({
        "W": (pa["W"] + pb["W"]) / 2.0,
        "b": (pa["b"] + pb["b"]) / 2.0,
    })
    return merged


# ---------------------------------------------------------------------------
# Numpy MLP (2-layer)
# ---------------------------------------------------------------------------

class NumpyMLP:
    """Two-layer MLP: features -> Dense(hidden, ReLU) -> Dense(n_classes, softmax)."""

    def __init__(self, n_features: int, hidden_size: int, n_classes: int):
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        # Xavier initialization
        self.W1 = np.random.randn(n_features, hidden_size).astype(np.float32) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, n_classes).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(n_classes, dtype=np.float32)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (logits, hidden_activations)."""
        hidden = X @ self.W1 + self.b1
        hidden_relu = np.maximum(hidden, 0)  # ReLU
        logits = hidden_relu @ self.W2 + self.b2
        return logits, hidden_relu

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(X)
        return logits.argmax(axis=-1)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 50, lr: float = 0.01) -> float:
        """Full-batch gradient descent with linear LR decay."""
        X = X.astype(np.float32)
        n = len(y)
        final_loss = 0.0

        for epoch in range(epochs):
            # Forward
            logits, hidden_relu = self.forward(X)
            probs = _softmax(logits)

            # Loss
            log_probs = np.log(probs[np.arange(n), y] + 1e-8)
            final_loss = -log_probs.mean()

            # Backprop through output layer
            d_logits = probs.copy()
            d_logits[np.arange(n), y] -= 1.0
            d_logits /= n

            grad_W2 = hidden_relu.T @ d_logits
            grad_b2 = d_logits.sum(axis=0)

            # Backprop through ReLU + hidden layer
            d_hidden = d_logits @ self.W2.T
            d_hidden[hidden_relu <= 0] = 0  # ReLU derivative

            grad_W1 = X.T @ d_hidden
            grad_b1 = d_hidden.sum(axis=0)

            # LR schedule
            current_lr = lr * (1.0 - epoch / epochs)

            # Update
            self.W2 -= current_lr * grad_W2
            self.b2 -= current_lr * grad_b2
            self.W1 -= current_lr * grad_W1
            self.b1 -= current_lr * grad_b1

        return float(final_loss)

    def get_params(self) -> dict:
        return {
            "W1": self.W1.copy(), "b1": self.b1.copy(),
            "W2": self.W2.copy(), "b2": self.b2.copy(),
        }

    def set_params(self, params: dict):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()
        self.W2 = params["W2"].copy()
        self.b2 = params["b2"].copy()


def train_mlp_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int = 3,
    hidden_size: int = 32,
    epochs: int = 50,
    lr: float = 0.01,
) -> NumpyMLP:
    """Train a numpy MLP baseline."""
    model = NumpyMLP(train_features.shape[1], hidden_size, num_classes)
    loss = model.train(train_features, train_labels, epochs=epochs, lr=lr)
    log.info("MLP baseline trained: %d samples, hidden=%d, %d epochs, loss=%.4f",
             len(train_labels), hidden_size, epochs, loss)
    return model


def evaluate_mlp_baseline(
    model: NumpyMLP,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
) -> dict:
    """Evaluate a numpy MLP baseline."""
    preds = model.predict(eval_features.astype(np.float32))
    return _accuracy_from_preds(preds, eval_labels, model.n_classes)


def fedavg_mlp(model_a: NumpyMLP, model_b: NumpyMLP) -> NumpyMLP:
    """FedAvg two MLP models by averaging all parameters."""
    merged = NumpyMLP(model_a.n_features, model_a.hidden_size, model_a.n_classes)
    pa, pb = model_a.get_params(), model_b.get_params()
    merged.set_params({
        k: (pa[k] + pb[k]) / 2.0 for k in pa
    })
    return merged


# ---------------------------------------------------------------------------
# Unified baseline runner (Pi-compatible)
# ---------------------------------------------------------------------------

def run_all_baselines_pi(
    node_features: dict[str, dict],
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    num_classes: int = 3,
    epochs: int = 50,
    lr: float = 0.01,
    k: int = 5,
    hidden_size: int = 32,
    feature_type: str = "binary",
) -> dict:
    """Run all numpy-based baselines on collected features.

    Args:
        node_features: {"claudio": {"train_X": ..., "train_y": ...},
                        "paolo":  {"train_X": ..., "train_y": ...}}
        eval_features: (N, D) eval features
        eval_labels: (N,) eval labels
        num_classes: Number of classes.
        feature_type: "binary" or "int8" — controls KNN metric.

    Returns:
        Dict with results for each baseline variant.
    """
    results = {}
    knn_metric = "hamming" if feature_type == "binary" else "euclidean"

    # --- Individual linear baselines ---
    individual_linear = {}
    linear_models = {}
    for node_id, data in node_features.items():
        model = train_linear_baseline(
            data["train_X"], data["train_y"],
            num_classes=num_classes, epochs=epochs, lr=lr,
        )
        linear_models[node_id] = model
        individual_linear[node_id] = evaluate_linear_baseline(
            model, eval_features, eval_labels)
    results["linear_individual"] = individual_linear

    # --- FedAvg linear ---
    node_ids = list(linear_models.keys())
    if len(node_ids) >= 2:
        merged_linear = fedavg_linear(
            linear_models[node_ids[0]], linear_models[node_ids[1]])
        results["linear_fedavg"] = evaluate_linear_baseline(
            merged_linear, eval_features, eval_labels)

    # --- Individual MLP baselines ---
    individual_mlp = {}
    mlp_models = {}
    for node_id, data in node_features.items():
        model = train_mlp_baseline(
            data["train_X"], data["train_y"],
            num_classes=num_classes, hidden_size=hidden_size,
            epochs=epochs, lr=lr,
        )
        mlp_models[node_id] = model
        individual_mlp[node_id] = evaluate_mlp_baseline(
            model, eval_features, eval_labels)
    results["mlp_individual"] = individual_mlp

    # --- FedAvg MLP ---
    if len(node_ids) >= 2:
        merged_mlp = fedavg_mlp(
            mlp_models[node_ids[0]], mlp_models[node_ids[1]])
        results["mlp_fedavg"] = evaluate_mlp_baseline(
            merged_mlp, eval_features, eval_labels)

    # --- Individual KNN baselines ---
    individual_knn = {}
    for node_id, data in node_features.items():
        individual_knn[node_id] = knn_baseline(
            data["train_X"], data["train_y"],
            eval_features, eval_labels,
            k=k, num_classes=num_classes, metric=knn_metric,
        )
    results["knn_individual"] = individual_knn

    # --- FedAvg KNN (pooled features) ---
    if len(node_ids) >= 2:
        results["knn_fedavg"] = knn_fedavg(
            node_features[node_ids[0]]["train_X"],
            node_features[node_ids[0]]["train_y"],
            node_features[node_ids[1]]["train_X"],
            node_features[node_ids[1]]["train_y"],
            eval_features, eval_labels,
            k=k, num_classes=num_classes, metric=knn_metric,
        )

    return results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test numpy baselines on Pi")
    parser.add_argument("--features-dir", type=Path,
                        default=Path.home() / "federated_experiment" / "results")
    parser.add_argument("--num-classes", type=int, default=3)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Load features saved by node_worker save_features command
    node_features = {}
    for name in ("claudio", "paolo"):
        X_path = args.features_dir / f"{name}_train_features_bin.npy"
        y_path = args.features_dir / f"{name}_train_labels.npy"
        if X_path.exists() and y_path.exists():
            node_features[name] = {
                "train_X": np.load(X_path),
                "train_y": np.load(y_path),
            }
            log.info("Loaded %s: X=%s, y=%s", name,
                     node_features[name]["train_X"].shape,
                     node_features[name]["train_y"].shape)

    eval_X = np.load(args.features_dir / "eval_features_bin.npy")
    eval_y = np.load(args.features_dir / "eval_labels.npy")
    log.info("Eval: X=%s, y=%s", eval_X.shape, eval_y.shape)

    results = run_all_baselines_pi(
        node_features, eval_X, eval_y,
        num_classes=args.num_classes,
    )

    import json
    print(json.dumps(results, indent=2, default=str))
