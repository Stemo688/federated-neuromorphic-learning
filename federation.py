"""Federation strategies for merging edge learning weights from multiple nodes.

Weight convention:
  - Shape: (num_neurons, num_features) where num_features=64
  - Neuron-to-class mapping uses global label ordering (backward=0, follow=1, forward=2)
  - With N neurons per class, neurons [0:N] = class 0, [N:2N] = class 1, [2N:3N] = class 2
  - Each node only trains neurons for its local classes; other neurons remain zeroed
"""

import logging
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

NUM_CLASSES = 3
NEURONS_PER_CLASS = 50
NOVEL_CLASSES = ["backward", "follow", "forward"]


@dataclass
class NodeWeights:
    """Weights extracted from one node's edge learning layer."""
    weights: np.ndarray              # (num_neurons, features)
    class_map: dict[int, list[int]]  # global_label -> [neuron_indices]
    node_id: str
    classes_seen: list[str]          # Class names this node trained on


def _get_class_neurons(nw: NodeWeights, cls_name: str,
                       novel_classes: list[str] | None = None) -> np.ndarray | None:
    """Get the neuron weight vectors for a specific class by name."""
    if novel_classes is None:
        novel_classes = NOVEL_CLASSES
    if cls_name not in novel_classes:
        return None
    label = novel_classes.index(cls_name)
    if label not in nw.class_map:
        return None
    indices = nw.class_map[label]
    return nw.weights[indices]


def _is_trained(neurons: np.ndarray) -> bool:
    """Check if neurons have been trained (non-zero weights)."""
    return neurons is not None and np.any(neurons != 0)


def fedavg(local: NodeWeights, remote: NodeWeights,
           novel_classes: list[str] | None = None) -> np.ndarray:
    """FedAvg: Average weights for shared classes, keep exclusive as-is.

    For classes trained by both nodes, average their neuron prototypes.
    For classes trained by only one node, use that node's neurons.
    For classes neither trained on, leave as zeros.
    """
    if novel_classes is None:
        novel_classes = NOVEL_CLASSES
    merged_blocks = []

    for cls_name in novel_classes:
        local_neurons = _get_class_neurons(local, cls_name, novel_classes)
        remote_neurons = _get_class_neurons(remote, cls_name, novel_classes)

        local_trained = _is_trained(local_neurons)
        remote_trained = _is_trained(remote_neurons)

        if local_trained and remote_trained:
            # Both trained on this class: average the prototypes
            n = min(len(local_neurons), len(remote_neurons))
            avg_neurons = (local_neurons[:n].astype(np.float32) +
                          remote_neurons[:n].astype(np.float32)) / 2.0
            # Pad if different neuron counts
            if len(local_neurons) > n:
                avg_neurons = np.vstack([avg_neurons, local_neurons[n:]])
            elif len(remote_neurons) > n:
                avg_neurons = np.vstack([avg_neurons, remote_neurons[n:]])
            merged_blocks.append(avg_neurons)
        elif local_trained:
            merged_blocks.append(local_neurons.astype(np.float32))
        elif remote_trained:
            merged_blocks.append(remote_neurons.astype(np.float32))
        else:
            # Neither trained: zeros
            merged_blocks.append(np.zeros((NEURONS_PER_CLASS, local.weights.shape[1]),
                                         dtype=np.float32))

    merged = np.vstack(merged_blocks).astype(np.int8)
    log.info("FedAvg: merged %d neurons (%d per class x %d classes)",
             len(merged), NEURONS_PER_CLASS, NUM_CLASSES)
    return merged


def fedunion(local: NodeWeights, remote: NodeWeights,
             novel_classes: list[str] | None = None) -> np.ndarray:
    """FedUnion: Concatenate neurons from both nodes for each class.

    For shared classes, both nodes contribute neurons (doubled count).
    For exclusive classes, only the training node contributes.
    """
    if novel_classes is None:
        novel_classes = NOVEL_CLASSES
    merged_blocks = []

    for cls_name in novel_classes:
        local_neurons = _get_class_neurons(local, cls_name, novel_classes)
        remote_neurons = _get_class_neurons(remote, cls_name, novel_classes)

        local_trained = _is_trained(local_neurons)
        remote_trained = _is_trained(remote_neurons)

        block = []
        if local_trained:
            block.append(local_neurons)
        if remote_trained:
            block.append(remote_neurons)

        if block:
            merged_blocks.append(np.vstack(block))
        else:
            merged_blocks.append(np.zeros((NEURONS_PER_CLASS, local.weights.shape[1]),
                                         dtype=np.int8))

    merged = np.vstack(merged_blocks).astype(np.int8)
    log.info("FedUnion: merged %d total neurons", len(merged))
    return merged


def fedbest(local: NodeWeights, remote: NodeWeights,
            novel_classes: list[str] | None = None) -> np.ndarray:
    """FedBest: Each node keeps its own neurons for classes it trained on,
    and takes the other node's neurons for classes it never saw.

    For shared classes, the local node's neurons are preferred.
    """
    if novel_classes is None:
        novel_classes = NOVEL_CLASSES
    merged_blocks = []

    for cls_name in novel_classes:
        local_neurons = _get_class_neurons(local, cls_name, novel_classes)
        remote_neurons = _get_class_neurons(remote, cls_name, novel_classes)

        local_trained = _is_trained(local_neurons)
        remote_trained = _is_trained(remote_neurons)

        if local_trained:
            # Prefer local for any class we trained on (including shared)
            merged_blocks.append(local_neurons)
        elif remote_trained:
            # Take remote for classes we never trained on
            merged_blocks.append(remote_neurons)
        else:
            merged_blocks.append(np.zeros((NEURONS_PER_CLASS, local.weights.shape[1]),
                                         dtype=np.int8))

    merged = np.vstack(merged_blocks).astype(np.int8)
    log.info("FedBest: merged %d neurons (prefer local, fill from remote)", len(merged))
    return merged


def fedmajority(local: NodeWeights, remote: NodeWeights,
                novel_classes: list[str] | None = None) -> np.ndarray:
    """FedMajority: Element-wise max of weights for shared classes.

    For shared classes, take the element-wise maximum at each weight position
    from both nodes. This preserves the sparse integer structure better than
    averaging, since int8 STDP weights are sparse (many zeros) and averaging
    halves all values, destroying the learned selectivity patterns.

    For exclusive classes, keep the training node's neurons as-is.
    """
    if novel_classes is None:
        novel_classes = NOVEL_CLASSES
    merged_blocks = []

    for cls_name in novel_classes:
        local_neurons = _get_class_neurons(local, cls_name, novel_classes)
        remote_neurons = _get_class_neurons(remote, cls_name, novel_classes)

        local_trained = _is_trained(local_neurons)
        remote_trained = _is_trained(remote_neurons)

        if local_trained and remote_trained:
            n = min(len(local_neurons), len(remote_neurons))
            max_neurons = np.maximum(
                local_neurons[:n].astype(np.int8),
                remote_neurons[:n].astype(np.int8),
            )
            if len(local_neurons) > n:
                max_neurons = np.vstack([max_neurons, local_neurons[n:]])
            elif len(remote_neurons) > n:
                max_neurons = np.vstack([max_neurons, remote_neurons[n:]])
            merged_blocks.append(max_neurons)
        elif local_trained:
            merged_blocks.append(local_neurons.astype(np.int8))
        elif remote_trained:
            merged_blocks.append(remote_neurons.astype(np.int8))
        else:
            merged_blocks.append(np.zeros((NEURONS_PER_CLASS, local.weights.shape[1]),
                                         dtype=np.int8))

    merged = np.vstack(merged_blocks).astype(np.int8)
    log.info("FedMajority: merged %d neurons (element-wise max for shared)", len(merged))
    return merged


def fedselective(local: NodeWeights, remote: NodeWeights,
                 novel_classes: list[str] | None = None) -> np.ndarray:
    """FedSelective: Per-neuron selection based on activation confidence.

    For shared classes, compare each paired neuron (local vs remote) by L1 norm
    and select the one with higher norm, indicating stronger/more confident
    learning. This avoids the destructive averaging of FedAvg while potentially
    keeping the best-learned neurons from each node.

    For exclusive classes, keep the training node's neurons as-is.
    """
    if novel_classes is None:
        novel_classes = NOVEL_CLASSES
    merged_blocks = []

    for cls_name in novel_classes:
        local_neurons = _get_class_neurons(local, cls_name, novel_classes)
        remote_neurons = _get_class_neurons(remote, cls_name, novel_classes)

        local_trained = _is_trained(local_neurons)
        remote_trained = _is_trained(remote_neurons)

        if local_trained and remote_trained:
            n = min(len(local_neurons), len(remote_neurons))
            selected = np.empty_like(local_neurons[:n])
            for i in range(n):
                local_norm = np.abs(local_neurons[i].astype(np.float32)).sum()
                remote_norm = np.abs(remote_neurons[i].astype(np.float32)).sum()
                selected[i] = local_neurons[i] if local_norm >= remote_norm else remote_neurons[i]
            if len(local_neurons) > n:
                selected = np.vstack([selected, local_neurons[n:]])
            elif len(remote_neurons) > n:
                selected = np.vstack([selected, remote_neurons[n:]])
            merged_blocks.append(selected)
        elif local_trained:
            merged_blocks.append(local_neurons)
        elif remote_trained:
            merged_blocks.append(remote_neurons)
        else:
            merged_blocks.append(np.zeros((NEURONS_PER_CLASS, local.weights.shape[1]),
                                         dtype=np.int8))

    merged = np.vstack(merged_blocks).astype(np.int8)
    log.info("FedSelective: merged %d neurons (per-neuron confidence selection)", len(merged))
    return merged


# ---------------------------------------------------------------------------
# Unified merge interface
# ---------------------------------------------------------------------------

def merge_weights(
    local: NodeWeights,
    remote: NodeWeights,
    strategy: str,
    novel_classes: list[str] | None = None,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Apply a federation strategy and build the class map for merged weights.

    Returns:
        (merged_weights, merged_class_map)
    """
    if novel_classes is None:
        novel_classes = NOVEL_CLASSES

    strategy_fn = {
        "fedavg": fedavg,
        "fedunion": fedunion,
        "fedbest": fedbest,
        "fedmajority": fedmajority,
        "fedselective": fedselective,
    }
    if strategy not in strategy_fn:
        raise ValueError(f"Unknown strategy: {strategy}")
    merged = strategy_fn[strategy](local, remote, novel_classes)

    # Strategies that concatenate neurons (union-like neuron count)
    union_strategies = {"fedunion"}
    # Strategies that average neurons (avg-like neuron count)
    avg_strategies = {"fedavg", "fedmajority"}
    # Strategies that select neurons (best-like neuron count)
    select_strategies = {"fedbest", "fedselective"}

    # Build class map for merged weights
    class_map = {}
    idx = 0
    num_classes = len(novel_classes)
    for label in range(num_classes):
        cls_name = novel_classes[label]
        local_neurons = _get_class_neurons(local, cls_name, novel_classes)
        remote_neurons = _get_class_neurons(remote, cls_name, novel_classes)
        local_trained = _is_trained(local_neurons)
        remote_trained = _is_trained(remote_neurons)

        if strategy in union_strategies:
            n = 0
            if local_trained:
                n += len(local_neurons)
            if remote_trained:
                n += len(remote_neurons)
            if n == 0:
                n = NEURONS_PER_CLASS
        elif strategy in avg_strategies:
            if local_trained and remote_trained:
                n = max(len(local_neurons), len(remote_neurons))
            elif local_trained:
                n = len(local_neurons)
            elif remote_trained:
                n = len(remote_neurons)
            else:
                n = NEURONS_PER_CLASS
        else:  # select_strategies (fedbest, fedselective)
            if local_trained:
                n = len(local_neurons)
            elif remote_trained:
                n = len(remote_neurons)
            else:
                n = NEURONS_PER_CLASS

        class_map[label] = list(range(idx, idx + n))
        idx += n

    return merged, class_map


# ---------------------------------------------------------------------------
# N-node federation (N > 2)
# ---------------------------------------------------------------------------

def fedunion_n(nodes: list[NodeWeights],
               novel_classes: list[str] | None = None) -> np.ndarray:
    """FedUnion for N nodes: concatenate neurons from all nodes per class.

    For each class, all nodes that trained on it contribute their neurons.
    Neuron count for shared classes scales as N * neurons_per_class.
    """
    if novel_classes is None:
        novel_classes = NOVEL_CLASSES

    num_features = nodes[0].weights.shape[1]
    merged_blocks = []

    for cls_name in novel_classes:
        block = []
        for nw in nodes:
            neurons = _get_class_neurons(nw, cls_name, novel_classes)
            if _is_trained(neurons):
                block.append(neurons)

        if block:
            merged_blocks.append(np.vstack(block))
        else:
            merged_blocks.append(np.zeros((NEURONS_PER_CLASS, num_features),
                                         dtype=np.int8))

    merged = np.vstack(merged_blocks).astype(np.int8)
    log.info("FedUnion-N (N=%d): merged %d total neurons", len(nodes), len(merged))
    return merged


def fedavg_n(nodes: list[NodeWeights],
             novel_classes: list[str] | None = None) -> np.ndarray:
    """FedAvg for N nodes: average weights across all nodes per class.

    For each class, average neuron prototypes from all nodes that trained on it.
    Uses the minimum neuron count across contributing nodes.
    """
    if novel_classes is None:
        novel_classes = NOVEL_CLASSES

    num_features = nodes[0].weights.shape[1]
    merged_blocks = []

    for cls_name in novel_classes:
        trained_neurons = []
        for nw in nodes:
            neurons = _get_class_neurons(nw, cls_name, novel_classes)
            if _is_trained(neurons):
                trained_neurons.append(neurons)

        if len(trained_neurons) >= 2:
            n = min(len(tn) for tn in trained_neurons)
            avg = np.zeros((n, num_features), dtype=np.float32)
            for tn in trained_neurons:
                avg += tn[:n].astype(np.float32)
            avg /= len(trained_neurons)
            # If nodes have more neurons than the min, keep extras from first node
            if len(trained_neurons[0]) > n:
                avg = np.vstack([avg, trained_neurons[0][n:].astype(np.float32)])
            merged_blocks.append(avg)
        elif len(trained_neurons) == 1:
            merged_blocks.append(trained_neurons[0].astype(np.float32))
        else:
            merged_blocks.append(np.zeros((NEURONS_PER_CLASS, num_features),
                                         dtype=np.float32))

    merged = np.vstack(merged_blocks).astype(np.int8)
    log.info("FedAvg-N (N=%d): merged %d neurons", len(nodes), len(merged))
    return merged


def merge_weights_n(
    nodes: list[NodeWeights],
    strategy: str,
    novel_classes: list[str] | None = None,
) -> tuple[np.ndarray, dict[int, list[int]]]:
    """Apply an N-node federation strategy and build class map.

    Returns:
        (merged_weights, merged_class_map)
    """
    if novel_classes is None:
        novel_classes = NOVEL_CLASSES

    n_fn = {
        "fedunion": fedunion_n,
        "fedavg": fedavg_n,
    }
    if strategy not in n_fn:
        raise ValueError(f"Unknown N-node strategy: {strategy}. "
                         f"Supported: {list(n_fn.keys())}")

    merged = n_fn[strategy](nodes, novel_classes)

    # Build class map from merged weight array
    class_map = {}
    idx = 0
    for label, cls_name in enumerate(novel_classes):
        # Count neurons for this class
        if strategy == "fedunion":
            n = 0
            for nw in nodes:
                neurons = _get_class_neurons(nw, cls_name, novel_classes)
                if _is_trained(neurons):
                    n += len(neurons)
            if n == 0:
                n = NEURONS_PER_CLASS
        else:  # fedavg
            trained = [nw for nw in nodes
                       if _is_trained(_get_class_neurons(nw, cls_name, novel_classes))]
            if trained:
                n = max(len(_get_class_neurons(nw, cls_name, novel_classes))
                        for nw in trained)
            else:
                n = NEURONS_PER_CLASS
        class_map[label] = list(range(idx, idx + n))
        idx += n

    return merged, class_map


def compute_communication_overhead(local: NodeWeights, remote: NodeWeights) -> dict:
    """Calculate the bytes exchanged during weight sharing."""
    import json as _json

    local_bytes = local.weights.nbytes
    remote_bytes = remote.weights.nbytes
    local_meta = len(_json.dumps({
        "class_map": {str(k): v for k, v in local.class_map.items()},
        "classes_seen": local.classes_seen,
    }).encode())
    remote_meta = len(_json.dumps({
        "class_map": {str(k): v for k, v in remote.class_map.items()},
        "classes_seen": remote.classes_seen,
    }).encode())

    return {
        "local_to_remote_bytes": local_bytes + local_meta,
        "remote_to_local_bytes": remote_bytes + remote_meta,
        "total_bytes": local_bytes + local_meta + remote_bytes + remote_meta,
    }
