#!/usr/bin/env python3
"""Fine-tune DS-CNN feature extractor on target keywords.

Runs on a Raspberry Pi using keras/quantizeml/cnn2snn.
Fine-tunes the pretrained DS-CNN model on a 3-class subset, then converts back
to Akida and extracts the headless feature extractor.

Supports:
  - Custom fine-tuning classes (--finetune-classes yes,no,stop for disjoint)
  - Wider feature dimensions (--feature-dim 128 or 256)
  - Custom output model name (--output-model feat_extractor_disjoint.fbz)

Usage (on Pi):
    source ~/akida-env/bin/activate
    cd ~/federated_experiment
    python finetune_dscnn.py --epochs 20 --deploy-to-paolo

Disjoint classes:
    python finetune_dscnn.py --finetune-classes yes,no,stop --output-model feat_extractor_disjoint.fbz

Wide features:
    python finetune_dscnn.py --feature-dim 128 --output-model feat_extractor_wide128.fbz

Quick test:
    python finetune_dscnn.py --epochs 1 --max-samples 50
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Default target classes for fine-tuning (same as experiment novel classes)
DEFAULT_TARGET_CLASSES = ["backward", "follow", "forward"]
INPUT_SHAPE = (49, 10, 1)
DEFAULT_OUTPUT_MODEL = "feat_extractor.fbz"


def load_wav_features(data_dir: Path, max_samples: int | None = None,
                      target_classes: list[str] | None = None) -> tuple:
    """Load WAV files and extract MFCC features for target classes.

    Reuses the same feature extraction pipeline as data_loader.py.

    Args:
        target_classes: List of class names to load. Defaults to DEFAULT_TARGET_CLASSES.

    Returns:
        X: (N, 49, 10, 1) uint8 features
        y: (N,) int32 labels
    """
    if target_classes is None:
        target_classes = DEFAULT_TARGET_CLASSES

    from scipy.io import wavfile
    from scipy.signal import stft

    all_X, all_y = [], []

    for label_idx, cls_name in enumerate(target_classes):
        cls_dir = data_dir / cls_name
        if not cls_dir.is_dir():
            log.warning("Class dir not found: %s", cls_dir)
            continue

        wav_files = sorted(cls_dir.glob("*.wav"))
        if max_samples is not None:
            wav_files = wav_files[:max_samples]

        for wf in wav_files:
            try:
                sr, audio = wavfile.read(str(wf))
                if audio.dtype == np.int16:
                    audio = audio.astype(np.float32) / 32768.0

                # Pad/trim to 1 second
                target_len = 16000
                if len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)))
                else:
                    audio = audio[:target_len]

                # STFT -> log-mel spectrogram
                _, _, Zxx = stft(audio, fs=16000, nperseg=640, noverlap=320)
                power = np.abs(Zxx) ** 2

                n_freq = power.shape[0]
                mel_indices = np.linspace(0, n_freq - 1, 11).astype(int)
                mel_spec = np.zeros((10, power.shape[1]), dtype=np.float32)
                for i in range(10):
                    mel_spec[i] = power[mel_indices[i]:mel_indices[i + 1] + 1].mean(axis=0)

                mel_spec = np.log(mel_spec + 1e-6)
                features = mel_spec.T  # (frames, 10)

                if features.shape[0] < 49:
                    features = np.pad(features, ((0, 49 - features.shape[0]), (0, 0)))
                else:
                    features = features[:49]

                fmin, fmax = features.min(), features.max()
                if fmax - fmin > 0:
                    features = (features - fmin) / (fmax - fmin) * 255.0
                features = features.astype(np.uint8)

                all_X.append(features.reshape(49, 10, 1))
                all_y.append(label_idx)
            except Exception as e:
                log.warning("Skipping %s: %s", wf, e)

        log.info("Class '%s': loaded %d samples",
                 cls_name, sum(1 for yy in all_y if yy == label_idx))

    X = np.stack(all_X, axis=0)
    y = np.array(all_y, dtype=np.int32)
    log.info("Total dataset: %d samples, shape=%s", len(X), X.shape)
    return X, y


def finetune(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 0.001,
    max_samples: int | None = None,
    deploy_to_paolo: bool = False,
    target_classes: list[str] | None = None,
    feature_dim: int = 64,
    output_model: str = DEFAULT_OUTPUT_MODEL,
):
    """Fine-tune DS-CNN on target keywords and produce a new feature extractor.

    Steps:
    1. Load audio data and extract features
    2. Load pretrained DS-CNN (already quantized), replace head
    3. Optionally add a wide projection layer (if feature_dim > 64)
    4. Freeze early layers, train with Adam via tf_keras
    5. Convert to Akida with cnn2snn
    6. Pop classification head (and keep wide projection if present)
    7. Save as output_model

    Args:
        target_classes: Classes to fine-tune on (default: backward,follow,forward).
        feature_dim: Output feature dimension (64=default, 128/256=wide projection).
        output_model: Output filename for the extractor (default: feat_extractor.fbz).
    """
    if target_classes is None:
        target_classes = DEFAULT_TARGET_CLASSES

    import tf_keras
    from quantizeml.layers.dense import QuantizedDense
    from quantizeml.layers.quantizer_layers import Dequantizer

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load data ---
    log.info("Loading audio data from %s for classes %s ...", data_dir, target_classes)
    X, y = load_wav_features(data_dir, max_samples=max_samples,
                             target_classes=target_classes)

    # Shuffle and split train/val
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    log.info("Train: %d, Val: %d", len(X_train), len(X_val))

    # --- 2. Build model ---
    log.info("Loading pretrained DS-CNN ...")
    from akida_models import ds_cnn_kws_pretrained
    model_keras = ds_cnn_kws_pretrained()

    # Get the global_avg pooling output (before the 33-class head)
    base_output = model_keras.get_layer("pw_separable_4/global_avg").output

    # Optionally add wide projection layer
    if feature_dim > 64:
        log.info("Adding wide projection layer: 64 -> %d", feature_dim)
        # Use Dequantizer between quantized backbone and regular Dense layers
        # to avoid per-channel vs per-tensor quantization mismatch
        x = Dequantizer(name="dequant_wide")(base_output)
        x = tf_keras.layers.Dense(feature_dim, name="dense_wide")(x)
        x = tf_keras.layers.Dense(len(target_classes), name="dense_3class")(x)
    else:
        # Standard: directly from 64-dim to num_classes
        x = QuantizedDense(len(target_classes), name="dense_3class")(base_output)
        x = Dequantizer(name="dequant_3class")(x)

    x = tf_keras.layers.Activation("softmax", name="softmax_3class")(x)
    finetune_model = tf_keras.Model(inputs=model_keras.input, outputs=x)

    # Freeze all layers except last 2 separable blocks + new head
    # Layer 10 is dw_separable_3, so i >= 10 unfreezes blocks 3, 4, global_avg, and new head
    freeze_until = None
    for i, layer in enumerate(finetune_model.layers):
        if "dw_separable_3" in layer.name:
            freeze_until = i
            break

    if freeze_until is None:
        freeze_until = int(len(finetune_model.layers) * 0.8)

    for i, layer in enumerate(finetune_model.layers):
        layer.trainable = i >= freeze_until

    trainable_count = sum(1 for l in finetune_model.layers if l.trainable)
    log.info("Trainable layers: %d / %d (freeze_until=%d)",
             trainable_count, len(finetune_model.layers), freeze_until)

    # --- 3. Train ---
    finetune_model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    log.info("Starting fine-tuning for %d epochs ...", epochs)
    t0 = time.time()
    history = finetune_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    train_time = time.time() - t0
    log.info("Fine-tuning complete in %.1f seconds", train_time)

    # Save training history
    hist_data = {
        "epochs": epochs,
        "train_time_seconds": train_time,
        "train_acc": [float(v) for v in history.history["accuracy"]],
        "val_acc": [float(v) for v in history.history["val_accuracy"]],
        "train_loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }
    with open(output_dir / "finetune_history.json", "w") as f:
        json.dump(hist_data, f, indent=2)

    final_val_acc = history.history["val_accuracy"][-1]
    log.info("Final val accuracy: %.1f%%", final_val_acc * 100)

    # --- 4. Convert to Akida ---
    feat_extractor_path = model_dir / output_model
    backup_path = model_dir / "feat_extractor_original.fbz"

    # Backup original if it exists and we're overwriting the default
    if output_model == DEFAULT_OUTPUT_MODEL:
        if feat_extractor_path.exists() and not backup_path.exists():
            import shutil
            shutil.copy2(str(feat_extractor_path), str(backup_path))
            log.info("Backed up original extractor to %s", backup_path)

    try:
        from cnn2snn import convert
        log.info("Converting to Akida ...")
        akida_model = convert(finetune_model)
        log.info("Conversion complete: %d Akida layers", len(akida_model.layers))
        for l in akida_model.layers:
            log.info("  %s: output_dims=%s", l.name, l.output_dims)
    except Exception as e:
        log.error("Akida conversion failed: %s", e)
        log.info("Keeping original feature extractor")
        return hist_data

    # --- 5. Pop classification head / save wide projection ---
    if feature_dim > 64:
        # Wide mode: cnn2snn converts only the quantized backbone (64-dim output).
        # The Dense projection layers are NOT in the Akida model.
        # Save backbone as .fbz + projection weights as .npy files.
        log.info("Wide mode: saving backbone (64-dim) + projection weights")

        # Extract Dense("dense_wide") weights from Keras model
        dense_wide_layer = finetune_model.get_layer("dense_wide")
        proj_W, proj_b = dense_wide_layer.get_weights()
        log.info("Projection weights: W=%s, b=%s", proj_W.shape, proj_b.shape)

        # Save projection weights alongside the model
        proj_stem = output_model.replace(".fbz", "")
        np.save(str(model_dir / f"{proj_stem}_proj_W.npy"), proj_W)
        np.save(str(model_dir / f"{proj_stem}_proj_b.npy"), proj_b)
        log.info("Saved projection weights to %s_proj_{{W,b}}.npy", proj_stem)

        # Save the Akida backbone (64-dim output, no popping needed)
        akida_model.save(str(feat_extractor_path))
        log.info("Saved backbone to %s", feat_extractor_path)

        out_dims = akida_model.layers[-1].output_dims
        # Sanity check backbone
        test_input = X_train[:1]
        backbone_out = akida_model.forward(test_input)
        backbone_features = backbone_out.reshape(1, -1).astype(np.float32)
        projected = backbone_features @ proj_W + proj_b
        log.info("Sanity: backbone %s -> projected %s (expected dim=%d)",
                 backbone_features.shape, projected.shape, feature_dim)
        assert projected.shape[-1] == feature_dim, f"Bad projected shape: {projected.shape}"
    else:
        # Standard mode: pop classification head to get 64-dim extractor
        layers_before = len(akida_model.layers)
        try:
            akida_model.pop_layer()  # dequant_3class
            log.info("Popped layer, last=%s", akida_model.layers[-1].name)
            akida_model.pop_layer()  # dense_3class
            log.info("Popped layer, last=%s", akida_model.layers[-1].name)
        except Exception as e:
            log.error("Error popping layers: %s", e)
            log.info("Keeping original feature extractor")
            return hist_data

        out_dims = akida_model.layers[-1].output_dims
        log.info("After popping: %d -> %d layers, output_dims=%s",
                 layers_before, len(akida_model.layers), out_dims)

        if out_dims[-1] != feature_dim:
            log.error("Unexpected output dims %s (expected last dim = %d)", out_dims, feature_dim)
            log.info("Keeping original feature extractor")
            return hist_data

        # --- 6. Save ---
        akida_model.save(str(feat_extractor_path))
        log.info("Saved fine-tuned feature extractor to %s", feat_extractor_path)

        # Quick sanity check
        test_input = X_train[:1]
        test_output = akida_model.forward(test_input)
        log.info("Sanity check: input %s -> output %s", test_input.shape, test_output.shape)
        assert test_output.shape[-1] == feature_dim, f"Bad output shape: {test_output.shape}"

    # --- 7. Deploy to paolo ---
    if deploy_to_paolo:
        paolo_ip = "10.0.0.2"
        remote_path = f"~/federated_experiment/models/{output_model}"
        log.info("Deploying %s to paolo (%s) ...", output_model, paolo_ip)
        try:
            subprocess.run([
                "scp", "-o", "StrictHostKeyChecking=no",
                str(feat_extractor_path),
                f"admin@{paolo_ip}:{remote_path}",
            ], check=True, timeout=60)
            log.info("Deployed to paolo successfully")
            # Also deploy projection weights for wide models
            if feature_dim > 64:
                for suffix in ["_proj_W.npy", "_proj_b.npy"]:
                    local = str(model_dir / f"{proj_stem}{suffix}")
                    remote = f"~/federated_experiment/models/{proj_stem}{suffix}"
                    subprocess.run([
                        "scp", "-o", "StrictHostKeyChecking=no",
                        local, f"admin@{paolo_ip}:{remote}",
                    ], check=True, timeout=60)
                log.info("Deployed projection weights to paolo")
        except Exception as e:
            log.error("Deploy to paolo failed: %s", e)

    hist_data["output_dims"] = list(out_dims)
    hist_data["feature_dim"] = feature_dim
    hist_data["target_classes"] = target_classes
    hist_data["output_model"] = output_model
    hist_data["feat_extractor_path"] = str(feat_extractor_path)
    return hist_data


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DS-CNN on target keywords")
    parser.add_argument("--data-dir", type=Path,
                        default=Path.home() / "federated_experiment" / "data" / "raw",
                        help="Directory with class subdirs of WAV files")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "federated_experiment",
                        help="Output directory for models and history")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per class (for quick testing)")
    parser.add_argument("--deploy-to-paolo", action="store_true",
                        help="SCP the new extractor to paolo via direct Ethernet")
    parser.add_argument("--finetune-classes", type=str, default=None,
                        help="Comma-separated classes (default: backward,follow,forward)")
    parser.add_argument("--feature-dim", type=int, default=64,
                        help="Feature dimension (64=standard, 128/256=wide projection)")
    parser.add_argument("--output-model", type=str, default=DEFAULT_OUTPUT_MODEL,
                        help="Output model filename (default: feat_extractor.fbz)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [finetune] %(levelname)s %(message)s",
    )

    target_classes = None
    if args.finetune_classes:
        target_classes = [c.strip() for c in args.finetune_classes.split(",")]
        log.info("Fine-tuning on custom classes: %s", target_classes)

    result = finetune(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        deploy_to_paolo=args.deploy_to_paolo,
        target_classes=target_classes,
        feature_dim=args.feature_dim,
        output_model=args.output_model,
    )

    if result:
        log.info("Fine-tuning result: val_acc=%.1f%%",
                 result.get("val_acc", [0])[-1] * 100)
    else:
        log.error("Fine-tuning failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
