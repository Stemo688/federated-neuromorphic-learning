"""Download and prepare Google Speech Commands dataset for the experiment.

Produces MFCC features shaped (49, 10, 1) matching ds_cnn_kws input expectations.
Handles the non-IID split: each node gets its exclusive class(es) plus the shared class.
"""

import logging
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

try:
    from scipy.io import wavfile
    from scipy.signal import stft
except ImportError:
    wavfile = None
    stft = None

from . import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_speech_commands(dest_dir: Path) -> Path:
    """Download and extract Google Speech Commands v0.02 if not already present."""
    dest_dir = Path(dest_dir)
    marker = dest_dir / ".extracted"
    if marker.exists():
        log.info("Speech Commands already extracted at %s", dest_dir)
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / "speech_commands_v0.02.tar.gz"

    if not archive.exists():
        log.info("Downloading Speech Commands v0.02 (~2.3 GB) ...")
        urllib.request.urlretrieve(config.SPEECH_COMMANDS_URL, archive)
        log.info("Download complete.")

    log.info("Extracting archive ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(dest_dir)
    marker.touch()
    log.info("Extraction complete at %s", dest_dir)
    return dest_dir


# ---------------------------------------------------------------------------
# Feature extraction — MFCC-like spectrogram (49 frames, 10 coefficients)
# ---------------------------------------------------------------------------

def _load_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load a 16-bit PCM WAV, return float32 array normalised to [-1, 1]."""
    sr, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    # Pad or trim to exactly 1 second
    target_len = target_sr
    if len(data) < target_len:
        data = np.pad(data, (0, target_len - len(data)))
    else:
        data = data[:target_len]
    return data


def extract_features(wav_path: Path) -> np.ndarray:
    """Extract (49, 10, 1) MFCC-like features from a WAV file.

    Uses a simplified log-mel spectrogram pipeline matching the ds_cnn_kws
    input format: 49 time frames x 10 frequency bins.
    """
    audio = _load_wav(wav_path)

    # STFT parameters to get ~49 frames from 16000 samples
    nperseg = 640   # 40 ms window
    noverlap = 320  # 20 ms hop → ~49 frames for 1s audio
    _, _, Zxx = stft(audio, fs=16000, nperseg=nperseg, noverlap=noverlap)
    power = np.abs(Zxx) ** 2

    # Mel-like binning: pick 10 log-spaced frequency bands
    n_freq = power.shape[0]
    mel_indices = np.linspace(0, n_freq - 1, 10 + 1).astype(int)
    mel_spec = np.zeros((10, power.shape[1]), dtype=np.float32)
    for i in range(10):
        mel_spec[i] = power[mel_indices[i]:mel_indices[i + 1] + 1].mean(axis=0)

    # Log compression
    mel_spec = np.log(mel_spec + 1e-6)

    # Transpose to (time, freq) = (frames, 10) and trim/pad to 49 frames
    features = mel_spec.T  # (frames, 10)
    if features.shape[0] < 49:
        features = np.pad(features, ((0, 49 - features.shape[0]), (0, 0)))
    else:
        features = features[:49]

    # Normalize to uint8 range [0, 255] for Akida
    fmin, fmax = features.min(), features.max()
    if fmax - fmin > 0:
        features = (features - fmin) / (fmax - fmin) * 255.0
    features = features.astype(np.uint8)

    return features.reshape(49, 10, 1)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def load_class_data(
    data_dir: Path,
    class_name: str,
    max_samples: int | None = None,
    seed: int | None = None,
    offset: int = 0,
) -> np.ndarray:
    """Load samples for a single class, return features array.

    Args:
        data_dir: Root directory containing class subdirectories.
        class_name: Name of the class subdirectory.
        max_samples: Maximum number of samples to load (after offset).
        seed: If provided, randomly shuffle wav files with this seed before slicing.
              If None, use sorted order (deterministic, legacy behavior).
        offset: Skip this many files before taking max_samples.
                Ensures non-overlapping splits when called multiple times with same seed.
    """
    class_dir = Path(data_dir) / class_name
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Class directory not found: {class_dir}")

    wav_files = sorted(class_dir.glob("*.wav"))
    if seed is not None:
        rng = np.random.RandomState(seed)
        rng.shuffle(wav_files)

    wav_files = wav_files[offset:]
    if max_samples is not None:
        wav_files = wav_files[:max_samples]

    features = []
    for wf in wav_files:
        try:
            feat = extract_features(wf)
            features.append(feat)
        except Exception as e:
            log.warning("Skipping %s: %s", wf, e)

    X = np.stack(features, axis=0)  # (N, 49, 10, 1)
    return X


def prepare_base_dataset(
    data_dir: Path,
    max_per_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load features and labels for the base (pre-training) classes."""
    all_X, all_y = [], []
    for idx, cls in enumerate(config.BASE_CLASSES):
        X = load_class_data(data_dir, cls, max_samples=max_per_class)
        y = np.full(len(X), idx, dtype=np.int32)
        all_X.append(X)
        all_y.append(y)
        log.info("Base class %s: %d samples", cls, len(X))

    return np.concatenate(all_X), np.concatenate(all_y)


def prepare_node_dataset(
    data_dir: Path,
    node_id: str,
    seed: int | None = None,
    novel_classes: list[str] | None = None,
    split: "config.FewShotSplit | None" = None,
    calibration_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare the few-shot dataset for a specific node.

    Labels use the GLOBAL NOVEL_CLASSES ordering so that both nodes share
    a consistent label space:
        backward=0, follow=1, forward=2

    Claudio trains on {backward(0), follow(1)}, Paolo on {backward(0), forward(2)}.

    Args:
        data_dir: Raw data directory with class subdirectories.
        node_id: "claudio" or "paolo".
        seed: Random seed for shuffling wav files before slicing.
        novel_classes: Override global NOVEL_CLASSES for alternative class sets.
        split: Override global SPLIT config.
        calibration_offset: Number of samples to skip (reserved for calibration set).

    Returns:
        X: features array (N, 49, 10, 1)
        y: labels array (N,) with GLOBAL label indices
        class_names: list of class names this node trains on
    """
    if novel_classes is None:
        novel_classes = config.NOVEL_CLASSES

    # Derive node-specific classes from novel_classes ordering:
    # novel_classes[0] = shared, novel_classes[1] = claudio_exclusive,
    # novel_classes[2] = paolo_exclusive
    if split is not None and split.shared_class in novel_classes:
        # Explicit split that matches the novel_classes
        shared_class = split.shared_class
        claudio_exclusive = split.claudio_exclusive
        paolo_exclusive = split.paolo_exclusive
    else:
        # Derive from novel_classes ordering (for custom class sets)
        shared_class = novel_classes[0]
        claudio_exclusive = novel_classes[1]
        paolo_exclusive = novel_classes[2]

    if node_id == "claudio":
        classes = [shared_class, claudio_exclusive]
    elif node_id == "paolo":
        classes = [shared_class, paolo_exclusive]
    else:
        raise ValueError(f"Unknown node_id: {node_id}")

    all_X, all_y = [], []
    for cls in classes:
        global_label = novel_classes.index(cls)
        samples_per_class = split.samples_per_class if split is not None else config.SPLIT.samples_per_class
        X = load_class_data(
            data_dir, cls,
            max_samples=samples_per_class,
            seed=seed,
            offset=calibration_offset,
        )
        y = np.full(len(X), global_label, dtype=np.int32)
        all_X.append(X)
        all_y.append(y)
        log.info("Node %s — class %s (global label %d): %d samples",
                 node_id, cls, global_label, len(X))

    return np.concatenate(all_X), np.concatenate(all_y), classes


def prepare_eval_dataset(
    data_dir: Path,
    max_per_class: int = 100,
    seed: int | None = None,
    novel_classes: list[str] | None = None,
    train_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare evaluation set with ALL novel classes (including ones each node didn't see).

    Args:
        data_dir: Raw data directory.
        max_per_class: Max eval samples per class.
        seed: Random seed for shuffling (same seed as training ensures consistent ordering).
        novel_classes: Override global NOVEL_CLASSES.
        train_offset: Number of training samples to skip past (calibration + training).
    """
    if novel_classes is None:
        novel_classes = config.NOVEL_CLASSES
    classes = novel_classes
    all_X, all_y = [], []

    for idx, cls in enumerate(classes):
        if seed is not None:
            # With seeded sampling, eval uses samples after the training offset
            X = load_class_data(
                data_dir, cls,
                max_samples=max_per_class,
                seed=seed,
                offset=train_offset,
            )
        else:
            # Legacy: use last samples after training slice
            X = load_class_data(data_dir, cls)
            if len(X) > max_per_class + config.SPLIT.samples_per_class:
                X = X[config.SPLIT.samples_per_class:
                       config.SPLIT.samples_per_class + max_per_class]
            elif len(X) > max_per_class:
                X = X[-max_per_class:]
        y = np.full(len(X), idx, dtype=np.int32)
        all_X.append(X)
        all_y.append(y)
        log.info("Eval class %s (label %d): %d samples", cls, idx, len(X))

    return np.concatenate(all_X), np.concatenate(all_y), classes


def prepare_calibration_dataset(
    data_dir: Path,
    classes: list[str],
    samples_per_class: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare calibration features from held-out samples for shared thresholds.

    Calibration samples come from offset=0 (the first `samples_per_class` samples
    in the seeded shuffle order). Training data must use offset=samples_per_class
    to avoid overlap.

    Returns:
        X: features array (N, 49, 10, 1)
        y: labels array (N,)
    """
    all_X, all_y = [], []
    for idx, cls in enumerate(classes):
        X = load_class_data(
            data_dir, cls,
            max_samples=samples_per_class,
            seed=seed,
            offset=0,
        )
        y = np.full(len(X), idx, dtype=np.int32)
        all_X.append(X)
        all_y.append(y)
        log.info("Calibration class %s: %d samples", cls, len(X))

    return np.concatenate(all_X), np.concatenate(all_y)
