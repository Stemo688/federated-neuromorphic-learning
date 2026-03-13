"""Configuration for the federated neuromorphic few-shot learning experiment."""

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Network topology
# ---------------------------------------------------------------------------
NODES = {
    "claudio": {
        "hostname": "raspberry-claudio",
        "lan_ip": "192.168.1.52",
        "direct_ip": "10.0.0.1",
        "ssh_user": "admin",
    },
    "paolo": {
        "hostname": "raspberry-paolo",
        "lan_ip": "192.168.1.53",
        "direct_ip": "10.0.0.2",
        "ssh_user": "admin",
    },
}

EXCHANGE_PORT = 9999          # TCP port for weight exchange over direct link
COMMAND_PORT = 9998           # TCP port for orchestrator commands

# ---------------------------------------------------------------------------
# Paths (on the Pis)
# ---------------------------------------------------------------------------
PI_VENV = Path.home() / "akida-env"
PI_WORK_DIR = Path.home() / "federated_experiment"
PI_DATA_DIR = PI_WORK_DIR / "data"
PI_MODEL_DIR = PI_WORK_DIR / "models"
PI_RESULTS_DIR = PI_WORK_DIR / "results"

# Path on Mac
MAC_PROJECT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "ds_cnn_kws"       # BrainChip DS-CNN for Keyword Spotting
INPUT_SHAPE = (49, 10, 1)       # MFCC frames: 49 time steps, 10 coefficients
NUM_BASE_CLASSES = 30           # Classes learned during base training
NUM_NEW_CLASSES = 3             # Novel classes for few-shot learning
EDGE_LAYER_NAME = "edge_layer"  # Name of the FullyConnected edge learning layer


@dataclass
class FewShotSplit:
    """Defines which novel classes each node receives."""
    shared_class: str = "backward"        # Both nodes see this class
    claudio_exclusive: str = "follow"     # Only claudio sees this
    paolo_exclusive: str = "forward"      # Only paolo sees this
    samples_per_class: int = 50           # Few-shot: 10-50 samples each


SPLIT = FewShotSplit()

# ---------------------------------------------------------------------------
# Edge learning parameters
# ---------------------------------------------------------------------------
NUM_FEATURES = 64               # Output features from DS-CNN pw_separable_4
NEURONS_PER_CLASS = 50          # Neurons per class in edge layer


@dataclass
class EdgeLearningParams:
    num_weights: int = 20             # Must be << NUM_FEATURES for STDP selectivity
    num_classes: int = NUM_NEW_CLASSES
    neurons_per_class: int = NEURONS_PER_CLASS
    learning_competition: float = 0.1


EDGE_PARAMS = EdgeLearningParams()

# ---------------------------------------------------------------------------
# Federation strategies to compare
# ---------------------------------------------------------------------------
FEDERATION_STRATEGIES = ["individual", "fedavg", "fedunion", "fedbest", "fedmajority", "fedselective"]

# ---------------------------------------------------------------------------
# Google Speech Commands dataset
# ---------------------------------------------------------------------------
SPEECH_COMMANDS_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/data/"
    "speech_commands_v0.02.tar.gz"
)

# The 35 standard classes in Speech Commands v0.02
ALL_CLASSES = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven",
    "sheila", "six", "stop", "three", "tree", "two", "up", "visual",
    "wow", "yes", "zero",
]

# Base classes = first 30 alphabetically (excluding the 3 novel ones)
NOVEL_CLASSES = [SPLIT.shared_class, SPLIT.claudio_exclusive, SPLIT.paolo_exclusive]
BASE_CLASSES = sorted([c for c in ALL_CLASSES if c not in NOVEL_CLASSES])[:NUM_BASE_CLASSES]

# ---------------------------------------------------------------------------
# Akida power estimate (from BrainChip AKD1000 datasheet)
# ---------------------------------------------------------------------------
AKIDA_IDLE_POWER_MW = 30.0
AKIDA_INFERENCE_POWER_MW = 100.0   # Approximate per-inference energy
AKIDA_LEARNING_POWER_MW = 150.0    # Approximate during STDP learning

# ---------------------------------------------------------------------------
# Multi-trial experiment settings
# ---------------------------------------------------------------------------
NUM_TRIALS = 10
SEEDS = [42 + i for i in range(NUM_TRIALS)]

# ---------------------------------------------------------------------------
# Multi-round federation
# ---------------------------------------------------------------------------
NUM_FEDERATION_ROUNDS = 5

# ---------------------------------------------------------------------------
# Shared binarization thresholds (calibration)
# ---------------------------------------------------------------------------
CALIBRATION_SAMPLES_PER_CLASS = 20  # Held-out samples for computing shared thresholds

# ---------------------------------------------------------------------------
# Alternative class sets for pretrained class comparison
# ---------------------------------------------------------------------------
PRETRAINED_CLASS_SETS = {
    "default": {
        "shared": "backward",
        "claudio_exclusive": "follow",
        "paolo_exclusive": "forward",
    },
    "movement": {
        "shared": "go",
        "claudio_exclusive": "left",
        "paolo_exclusive": "right",
    },
    "digits": {
        "shared": "one",
        "claudio_exclusive": "two",
        "paolo_exclusive": "three",
    },
    "pretrained": {
        "shared": "yes",
        "claudio_exclusive": "no",
        "paolo_exclusive": "stop",
    },
}

# ---------------------------------------------------------------------------
# Software baselines (runs on Mac)
# ---------------------------------------------------------------------------
BASELINE_EPOCHS = 50
BASELINE_LR = 0.01
BASELINE_WEIGHT_DECAY = 1e-4
MLP_HIDDEN_SIZE = 32

# ---------------------------------------------------------------------------
# Extended hyperparameter grid (Phase 2 comprehensive sweep)
# ---------------------------------------------------------------------------
EXTENDED_GRID = {
    "num_weights": [10, 15, 20, 25, 30, 35, 40],
    "neurons_per_class": [25, 50, 75],
    "learning_competition": [0.1, 1.0],
}

# Wide feature grids (for wider feature extractor experiments)
WIDE_128_NUM_WEIGHTS = [15, 20, 25, 30, 40, 50, 60]
WIDE_256_NUM_WEIGHTS = [20, 30, 40, 60, 80, 100]

# Binarization methods
BINARIZATION_METHODS = ["mean", "median", "entropy"]

# Disjoint fine-tuning classes (not overlapping with target STDP classes)
DISJOINT_FINETUNE_CLASSES = ["yes", "no", "stop"]

# Feature dimensions to test
FEATURE_DIMS = [64, 128, 256]

# Top-N configs for focused experiments
TOP_N_FOR_BINARIZATION = 5
TOP_N_FOR_DISJOINT = 5
TOP_N_FOR_MULTI_ROUND = 3

# ---------------------------------------------------------------------------
# Pi orchestrator mode (set via --pi-orchestrator flag)
# ---------------------------------------------------------------------------
PI_ORCHESTRATOR_NODE: str | None = None  # Set to "claudio" or "paolo" when running on Pi


def get_command_ip(node_name: str) -> str:
    """Return the IP to use for sending commands to a node worker."""
    if PI_ORCHESTRATOR_NODE is not None:
        if node_name == PI_ORCHESTRATOR_NODE:
            return "127.0.0.1"
        return NODES[node_name]["direct_ip"]
    return NODES[node_name]["lan_ip"]


def get_ssh_target(node_name: str) -> str | None:
    """Return SSH target IP, or None if node is local (Pi orchestrator mode)."""
    if PI_ORCHESTRATOR_NODE is not None:
        if node_name == PI_ORCHESTRATOR_NODE:
            return None
        return NODES[node_name]["direct_ip"]
    return NODES[node_name]["lan_ip"]


def is_local_node(node_name: str) -> bool:
    """Check if the given node is the local Pi orchestrator."""
    return PI_ORCHESTRATOR_NODE is not None and node_name == PI_ORCHESTRATOR_NODE
