"""Software baselines for comparison with neuromorphic edge learning.

Runs on the Mac using PyTorch (linear) and scikit-learn (KNN).
Operates on the same binarized features (64-dim, {0,1}) that the Akida edge
models receive, providing a fair comparison of federation strategies.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Linear baseline (PyTorch)
# ---------------------------------------------------------------------------

def _get_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class LinearBaseline:
    """Single linear layer 64 → num_classes, trained with cross-entropy."""

    def __init__(self, num_features: int = 64, num_classes: int = 3):
        torch, nn = _get_torch()
        self.model = nn.Linear(num_features, num_classes)
        self.num_classes = num_classes
        self.num_features = num_features

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


def train_linear_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int = 3,
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
) -> LinearBaseline:
    """Train a linear baseline on binarized features.

    Args:
        train_features: (N, 64), uint8 or float — binarized features
        train_labels: (N,), int — class labels
        num_classes: Number of output classes.
        epochs: Training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization weight decay.

    Returns:
        Trained LinearBaseline.
    """
    torch, nn = _get_torch()

    baseline = LinearBaseline(train_features.shape[1], num_classes)
    optimizer = torch.optim.SGD(baseline.model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X = torch.tensor(train_features, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.long)

    baseline.model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = baseline.model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

    log.info("Linear baseline trained: %d samples, %d epochs, wd=%.0e, final loss=%.4f",
             len(train_labels), epochs, weight_decay, loss.item())
    return baseline


def evaluate_linear_baseline(
    baseline: LinearBaseline,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
) -> dict:
    """Evaluate a linear baseline, returning accuracy dict."""
    torch, _ = _get_torch()

    X = torch.tensor(eval_features, dtype=torch.float32)
    y_true = eval_labels

    baseline.model.eval()
    with torch.no_grad():
        logits = baseline.model(X)
        preds = logits.argmax(dim=1).numpy()

    correct = int((preds == y_true).sum())
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0

    per_class = {}
    for cls in range(baseline.num_classes):
        mask = y_true == cls
        if mask.sum() > 0:
            per_class[str(cls)] = float((preds[mask] == cls).sum() / mask.sum())

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(y_true),
        "per_class_accuracy": per_class,
    }


def fedavg_linear_baseline(
    baseline_a: LinearBaseline,
    baseline_b: LinearBaseline,
) -> LinearBaseline:
    """FedAvg two linear baselines by averaging their parameters."""
    torch, _ = _get_torch()

    merged = LinearBaseline(baseline_a.num_features, baseline_a.num_classes)
    sd_a = baseline_a.state_dict()
    sd_b = baseline_b.state_dict()

    merged_sd = {}
    for key in sd_a:
        merged_sd[key] = (sd_a[key] + sd_b[key]) / 2.0
    merged.load_state_dict(merged_sd)

    log.info("FedAvg linear baseline: averaged %d parameters", sum(p.numel() for p in merged.model.parameters()))
    return merged


# ---------------------------------------------------------------------------
# MLP baseline (PyTorch) — 2-layer, float32 FedAvg
# ---------------------------------------------------------------------------

class MLPBaseline:
    """Two-layer MLP: 64 → hidden → num_classes, trained with cross-entropy.

    This is the critical missing comparison: what standard federated learning
    achieves with a conventional learner on the same features.
    """

    def __init__(self, num_features: int = 64, hidden_size: int = 32,
                 num_classes: int = 3):
        torch, nn = _get_torch()
        self.model = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.num_classes = num_classes
        self.num_features = num_features
        self.hidden_size = hidden_size

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd)


def train_mlp_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    num_classes: int = 3,
    hidden_size: int = 32,
    epochs: int = 50,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
) -> MLPBaseline:
    """Train a 2-layer MLP baseline on binarized features."""
    torch, nn = _get_torch()

    baseline = MLPBaseline(train_features.shape[1], hidden_size, num_classes)
    optimizer = torch.optim.SGD(baseline.model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X = torch.tensor(train_features, dtype=torch.float32)
    y = torch.tensor(train_labels, dtype=torch.long)

    baseline.model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = baseline.model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

    log.info("MLP baseline trained: %d samples, %d epochs, hidden=%d, wd=%.0e, final loss=%.4f",
             len(train_labels), epochs, hidden_size, weight_decay, loss.item())
    return baseline


def evaluate_mlp_baseline(
    baseline: MLPBaseline,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
) -> dict:
    """Evaluate an MLP baseline, returning accuracy dict."""
    torch, _ = _get_torch()

    X = torch.tensor(eval_features, dtype=torch.float32)
    y_true = eval_labels

    baseline.model.eval()
    with torch.no_grad():
        logits = baseline.model(X)
        preds = logits.argmax(dim=1).numpy()

    correct = int((preds == y_true).sum())
    accuracy = correct / len(y_true) if len(y_true) > 0 else 0.0

    per_class = {}
    for cls in range(baseline.num_classes):
        mask = y_true == cls
        if mask.sum() > 0:
            per_class[str(cls)] = float((preds[mask] == cls).sum() / mask.sum())

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(y_true),
        "per_class_accuracy": per_class,
    }


def fedavg_mlp_baseline(
    baseline_a: MLPBaseline,
    baseline_b: MLPBaseline,
) -> MLPBaseline:
    """FedAvg two MLP baselines by averaging all float32 parameters."""
    torch, _ = _get_torch()

    merged = MLPBaseline(baseline_a.num_features, baseline_a.hidden_size,
                         baseline_a.num_classes)
    sd_a = baseline_a.state_dict()
    sd_b = baseline_b.state_dict()

    merged_sd = {}
    for key in sd_a:
        merged_sd[key] = (sd_a[key] + sd_b[key]) / 2.0
    merged.load_state_dict(merged_sd)

    log.info("FedAvg MLP baseline: averaged %d parameters",
             sum(p.numel() for p in merged.model.parameters()))
    return merged


# ---------------------------------------------------------------------------
# KNN baseline (scikit-learn)
# ---------------------------------------------------------------------------

def knn_baseline(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    k: int = 5,
    num_classes: int = 3,
) -> dict:
    """KNN classification on binarized features.

    Returns accuracy dict matching the same format as evaluate_linear_baseline.
    """
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=min(k, len(train_labels)), metric="hamming")
    clf.fit(train_features, train_labels)
    preds = clf.predict(eval_features)

    correct = int((preds == eval_labels).sum())
    accuracy = correct / len(eval_labels) if len(eval_labels) > 0 else 0.0

    per_class = {}
    for cls in range(num_classes):
        mask = eval_labels == cls
        if mask.sum() > 0:
            per_class[str(cls)] = float((preds[mask] == cls).sum() / mask.sum())

    log.info("KNN (k=%d): %.1f%% accuracy (%d/%d)",
             k, accuracy * 100, correct, len(eval_labels))
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(eval_labels),
        "per_class_accuracy": per_class,
        "k": k,
    }


def fedavg_knn_baseline(
    train_features_a: np.ndarray,
    train_labels_a: np.ndarray,
    train_features_b: np.ndarray,
    train_labels_b: np.ndarray,
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    k: int = 5,
    num_classes: int = 3,
) -> dict:
    """Federated KNN: pool training features from both nodes, then classify.

    This is the KNN equivalent of FedUnion — combine all training data.
    """
    merged_features = np.concatenate([train_features_a, train_features_b], axis=0)
    merged_labels = np.concatenate([train_labels_a, train_labels_b], axis=0)

    log.info("FedAvg KNN: %d train samples from node A + %d from node B",
             len(train_labels_a), len(train_labels_b))
    return knn_baseline(merged_features, merged_labels, eval_features, eval_labels,
                        k=k, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Unified baseline runner
# ---------------------------------------------------------------------------

def run_all_baselines(
    node_features: dict[str, dict],
    eval_features: np.ndarray,
    eval_labels: np.ndarray,
    num_classes: int = 3,
    epochs: int = 50,
    lr: float = 0.01,
    k: int = 5,
    hidden_size: int = 32,
    weight_decay: float = 1e-4,
) -> dict:
    """Run all software baselines given collected features from both nodes.

    Args:
        node_features: {"claudio": {"train_X": ..., "train_y": ...},
                        "paolo": {"train_X": ..., "train_y": ...}}
        eval_features: (N, 64) binarized eval features
        eval_labels: (N,) eval labels
        num_classes: Number of classes.
        epochs: Training epochs for linear and MLP baselines.
        lr: Learning rate for linear and MLP baselines.
        k: KNN neighbor count.
        hidden_size: MLP hidden layer size.
        weight_decay: L2 regularization for gradient baselines.

    Returns:
        Dict with results for each baseline variant.
    """
    results = {}

    # --- Individual linear baselines ---
    individual_linear = {}
    linear_baselines = {}
    for node_id, data in node_features.items():
        bl = train_linear_baseline(
            data["train_X"], data["train_y"],
            num_classes=num_classes, epochs=epochs, lr=lr,
            weight_decay=weight_decay,
        )
        linear_baselines[node_id] = bl
        individual_linear[node_id] = evaluate_linear_baseline(bl, eval_features, eval_labels)
    results["linear_individual"] = individual_linear

    # --- FedAvg linear baseline ---
    node_ids = list(linear_baselines.keys())
    if len(node_ids) >= 2:
        merged_bl = fedavg_linear_baseline(linear_baselines[node_ids[0]], linear_baselines[node_ids[1]])
        results["linear_fedavg"] = evaluate_linear_baseline(merged_bl, eval_features, eval_labels)

    # --- Individual MLP baselines ---
    individual_mlp = {}
    mlp_baselines = {}
    for node_id, data in node_features.items():
        bl = train_mlp_baseline(
            data["train_X"], data["train_y"],
            num_classes=num_classes, hidden_size=hidden_size,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
        )
        mlp_baselines[node_id] = bl
        individual_mlp[node_id] = evaluate_mlp_baseline(bl, eval_features, eval_labels)
    results["mlp_individual"] = individual_mlp

    # --- FedAvg MLP baseline ---
    if len(node_ids) >= 2:
        merged_mlp = fedavg_mlp_baseline(mlp_baselines[node_ids[0]], mlp_baselines[node_ids[1]])
        results["mlp_fedavg"] = evaluate_mlp_baseline(merged_mlp, eval_features, eval_labels)

    # --- Individual KNN baselines ---
    individual_knn = {}
    for node_id, data in node_features.items():
        individual_knn[node_id] = knn_baseline(
            data["train_X"], data["train_y"],
            eval_features, eval_labels,
            k=k, num_classes=num_classes,
        )
    results["knn_individual"] = individual_knn

    # --- FedAvg KNN baseline (pooled features) ---
    if len(node_ids) >= 2:
        results["knn_fedavg"] = fedavg_knn_baseline(
            node_features[node_ids[0]]["train_X"],
            node_features[node_ids[0]]["train_y"],
            node_features[node_ids[1]]["train_X"],
            node_features[node_ids[1]]["train_y"],
            eval_features, eval_labels,
            k=k, num_classes=num_classes,
        )

    return results
