#!/usr/bin/env python3
"""Comprehensive experiment sweep for federated neuromorphic few-shot learning.

Runs on claudio as Pi orchestrator, controlling both Pis over direct Ethernet.
Six phases covering fine-tuning, hyperparameter grid, binarization comparison,
disjoint feature extractors, wide feature dimensions, and multi-round federation.

Usage (on claudio):
    source ~/akida-env/bin/activate
    cd ~/federated_experiment
    python comprehensive_sweep.py

Run a single phase:
    python comprehensive_sweep.py --phase B

Quick test:
    python comprehensive_sweep.py --phase B --num-trials 1 --grid-subset 2

Resume (auto-detects completed work):
    python comprehensive_sweep.py --phase ALL
"""

import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase B: Extended grid
# ---------------------------------------------------------------------------
EXTENDED_GRID = {
    "num_weights": [10, 15, 20, 25, 30, 35, 40],
    "neurons_per_class": [25, 50, 75],
    "learning_competition": [0.1, 1.0],
}

# Phase C: Binarization methods
BINARIZATION_METHODS = ["mean", "median", "entropy"]

# Phase E: Wide feature grids
WIDE_128_NUM_WEIGHTS = [15, 20, 25, 30, 40, 50, 60]
WIDE_256_NUM_WEIGHTS = [20, 30, 40, 60, 80, 100]

# Phase G: Entropy + wide-256 combined
WIDE_256_ENTROPY_NUM_WEIGHTS = [20, 30, 40, 60, 80, 100]

# Phase F: Multi-round federation
MULTI_ROUND_ROUNDS = 5
RETRAINING_STRATEGIES = ["fedavg", "fedunion"]

# Top-N configs for focused experiments
TOP_N_FOR_BINARIZATION = 5
TOP_N_FOR_DISJOINT = 5
TOP_N_FOR_MULTI_ROUND = 3

# Default seeds
DEFAULT_NUM_TRIALS = 10
DEFAULT_FINETUNE_EPOCHS = 20


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_pi_orchestrator():
    """Configure the orchestrator to run on claudio in Pi-to-Pi mode."""
    sys.path.insert(0, str(Path.home()))
    from federated_experiment import config
    config.PI_ORCHESTRATOR_NODE = "claudio"
    config.MAC_PROJECT_DIR = config.PI_WORK_DIR
    return config


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _config_key(params: dict) -> str:
    """Create a stable hashable key for a config dict."""
    return json.dumps(params, sort_keys=True)


def _save_results(path: Path, results_dict: dict):
    """Atomic JSON save: write to temp file then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".sweep_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        os.replace(tmp_path, str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _get_top_configs(config_results: list, n: int) -> list:
    """Sort configs by mean_best_fed_acc descending, return top n."""
    ranked = sorted(
        config_results,
        key=lambda c: c.get("summary", {}).get("mean_best_fed_acc", 0),
        reverse=True,
    )
    return ranked[:n]


def _resume_state(existing_configs: list, num_trials: int):
    """Build resume maps: completed configs (skip) and partial configs (extend).

    Returns:
        (completed, partial) dicts keyed by config_key.
        - completed: configs with >= num_trials trials (skip entirely)
        - partial: configs with 1..num_trials-1 trials (extend with remaining seeds)
    """
    completed = {}
    partial = {}
    for cfg_result in existing_configs:
        ckey = _config_key(cfg_result["params"])
        n_existing = len(cfg_result.get("trials", []))
        if n_existing >= num_trials:
            completed[ckey] = cfg_result
        elif n_existing > 0:
            partial[ckey] = cfg_result
    return completed, partial


def _compute_trial_metrics(individual_results: dict, federated_results: dict) -> dict:
    """Compute summary metrics from individual and federated results.

    Returns dict with mean_individual_accuracy, best_federated_accuracy,
    and best_strategy.
    """
    # Mean individual accuracy across nodes
    ind_accs = [
        individual_results.get(n, {}).get("accuracy", 0)
        for n in ("claudio", "paolo")
    ]
    mean_ind = float(np.mean(ind_accs))

    # Best federated accuracy across all strategies and nodes
    best_fed = 0.0
    best_strategy = ""
    strategies = ("fedavg", "fedunion", "fedbest", "fedmajority", "fedselective")
    for strategy in strategies:
        for node in ("claudio", "paolo"):
            key = f"{strategy}_{node}"
            acc = federated_results.get(key, {}).get("accuracy", 0)
            if acc > best_fed:
                best_fed = acc
                best_strategy = strategy

    return {
        "mean_individual_accuracy": mean_ind,
        "best_federated_accuracy": float(best_fed),
        "best_strategy": best_strategy,
    }


def _collect_and_run_baselines(config_module, results_dir: Path,
                                num_classes: int) -> dict:
    """SCP features from both nodes, run baselines_pi on binary and int8 features.

    For claudio (local orchestrator), features are already on the local filesystem.
    For paolo, SCP from 10.0.0.2 over direct Ethernet.

    Returns dict with baselines_binary and baselines_int8 results.
    """
    from federated_experiment.orchestrator import send_command
    from federated_experiment.baselines_pi import run_all_baselines_pi, knn_baseline, knn_fedavg

    results_dir.mkdir(parents=True, exist_ok=True)
    remote_results = Path.home() / "federated_experiment" / "results"

    # Tell both nodes to save features (binary + int8)
    for name in ("claudio", "paolo"):
        cmd_ip = config_module.get_command_ip(name)
        resp = send_command(cmd_ip, {"action": "save_features"}, timeout=120)
        if resp.get("status") != "ok":
            log.error("save_features failed on %s: %s", name, resp)
            return {}
        log.info("Node %s saved features: train=%s, eval=%s",
                 name, resp.get("train_shape"), resp.get("eval_shape"))

    # SCP features from paolo (claudio's files are local)
    paolo_files = []
    for prefix in ("paolo_train_features_bin", "paolo_train_features_int8",
                    "paolo_train_labels", "eval_features_bin",
                    "eval_features_int8", "eval_labels"):
        paolo_files.append(f"{prefix}.npy")

    for fname in paolo_files:
        remote_path = f"~/federated_experiment/results/{fname}"
        local_path = str(results_dir / fname)
        try:
            subprocess.run([
                "scp", "-o", "StrictHostKeyChecking=no",
                f"admin@10.0.0.2:{remote_path}",
                local_path,
            ], check=True, timeout=60)
        except subprocess.CalledProcessError:
            log.warning("Could not SCP %s from paolo", fname)

    # Claudio features are already local; copy if results_dir differs
    for prefix in ("claudio_train_features_bin", "claudio_train_features_int8",
                    "claudio_train_labels", "eval_features_bin",
                    "eval_features_int8", "eval_labels"):
        src = remote_results / f"{prefix}.npy"
        dst = results_dir / f"{prefix}.npy"
        if src.exists() and src.resolve() != dst.resolve():
            import shutil
            shutil.copy2(str(src), str(dst))

    # Load binary features
    node_features_bin = {}
    node_features_int8 = {}
    for name in ("claudio", "paolo"):
        bin_path = results_dir / f"{name}_train_features_bin.npy"
        int8_path = results_dir / f"{name}_train_features_int8.npy"
        labels_path = results_dir / f"{name}_train_labels.npy"
        if bin_path.exists() and labels_path.exists():
            node_features_bin[name] = {
                "train_X": np.load(bin_path),
                "train_y": np.load(labels_path),
            }
        if int8_path.exists() and labels_path.exists():
            node_features_int8[name] = {
                "train_X": np.load(int8_path),
                "train_y": np.load(labels_path),
            }

    # Load eval features
    eval_bin_path = results_dir / "eval_features_bin.npy"
    eval_int8_path = results_dir / "eval_features_int8.npy"
    eval_labels_path = results_dir / "eval_labels.npy"

    result = {}

    # Run full baselines on binary features
    if (len(node_features_bin) >= 2
            and eval_bin_path.exists() and eval_labels_path.exists()):
        eval_X_bin = np.load(eval_bin_path)
        eval_y = np.load(eval_labels_path)
        result["baselines_binary"] = run_all_baselines_pi(
            node_features=node_features_bin,
            eval_features=eval_X_bin,
            eval_labels=eval_y,
            num_classes=num_classes,
            feature_type="binary",
        )

    # Run KNN baselines on int8 features (Experiment 2)
    if (len(node_features_int8) >= 2
            and eval_int8_path.exists() and eval_labels_path.exists()):
        eval_X_int8 = np.load(eval_int8_path)
        eval_y = np.load(eval_labels_path)

        int8_knn_individual = {}
        for name, data in node_features_int8.items():
            int8_knn_individual[name] = knn_baseline(
                data["train_X"], data["train_y"],
                eval_X_int8, eval_y,
                k=5, num_classes=num_classes, metric="euclidean",
            )

        node_ids = list(node_features_int8.keys())
        int8_knn_fed = knn_fedavg(
            node_features_int8[node_ids[0]]["train_X"],
            node_features_int8[node_ids[0]]["train_y"],
            node_features_int8[node_ids[1]]["train_X"],
            node_features_int8[node_ids[1]]["train_y"],
            eval_X_int8, eval_y,
            k=5, num_classes=num_classes, metric="euclidean",
        )

        result["baselines_int8"] = {
            "knn_individual": int8_knn_individual,
            "knn_fedavg": int8_knn_fed,
        }

    return result


def _run_single_trial(
    seed: int,
    params: dict,
    config_module,
    data_dir: Path,
    results_dir: Path,
    extra_build_params: dict | None = None,
    run_baselines: bool = True,
    num_classes: int = 3,
) -> dict:
    """Run a single trial with the given STDP hyperparameters.

    Args:
        seed: Random seed for data split.
        params: STDP hyperparameters (num_weights, neurons_per_class,
                learning_competition).
        config_module: The config module (with get_command_ip, etc.).
        data_dir: Directory for prepared data.
        results_dir: Directory for results and features.
        extra_build_params: Additional params merged into build_model command
                           (e.g. binarization_method, feat_extractor_name).
        run_baselines: Whether to collect features and run baselines.
        num_classes: Number of classes.

    Returns:
        Trial result dict.
    """
    from federated_experiment.orchestrator import (
        prepare_data,
        deploy_to_nodes,
        send_command,
        run_local_training,
        run_individual_evaluation,
        run_weight_exchange,
        run_federated_evaluation,
    )

    trial_start = time.time()

    # 1. Prepare data with this seed
    prepare_data(data_dir, seed=seed, use_calibration=False)

    # 2. Deploy data to nodes
    deploy_to_nodes(data_dir)

    # 3. Build edge models with sweep hyperparameters
    build_params = dict(params)
    if extra_build_params:
        build_params.update(extra_build_params)

    for name in ("claudio", "paolo"):
        resp = send_command(config_module.get_command_ip(name), {
            "action": "build_model",
            "params": build_params,
        })
        if resp.get("status") != "ok":
            log.error("build_model failed on %s: %s", name, resp)
            return {"error": f"build_model failed on {name}", "seed": seed,
                    "params": params}

    # 4. Train (with optional binarization_method override)
    train_cmd_extra = {}
    if extra_build_params and "binarization_method" in extra_build_params:
        train_cmd_extra["binarization_method"] = extra_build_params["binarization_method"]

    training_results = {}
    for name in ("claudio", "paolo"):
        cmd = {"action": "train"}
        cmd.update(train_cmd_extra)
        result = send_command(
            config_module.get_command_ip(name), cmd, timeout=120)
        training_results[name] = result.get("metrics", {})
    log.info("Training complete: %s",
             {n: m.get("learning_time_ms", "?") for n, m in training_results.items()})

    # 5. Individual evaluation
    individual_results = run_individual_evaluation()

    # 6. Save features + run baselines
    baseline_results = {}
    if run_baselines:
        try:
            baseline_results = _collect_and_run_baselines(
                config_module, results_dir, num_classes)
        except Exception as e:
            log.error("Baseline collection failed: %s", e)
            baseline_results = {"error": str(e)}

    # 7. Weight exchange + federated evaluation
    exchange_data = run_weight_exchange()
    federated_results = run_federated_evaluation(exchange_data)

    trial_time = time.time() - trial_start

    # Summarize federated results into a clean dict
    federated_clean = {}
    for key, metrics in federated_results.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            federated_clean[key] = {
                "accuracy": metrics["accuracy"],
                "correct": metrics.get("correct"),
                "total": metrics.get("total"),
            }

    # Compute summary metrics
    summary = _compute_trial_metrics(individual_results, federated_results)

    trial_result = {
        "seed": seed,
        "params": params,
        "trial_time_seconds": trial_time,
        "individual": individual_results,
        "federated": federated_clean,
        "mean_individual_accuracy": summary["mean_individual_accuracy"],
        "best_federated_accuracy": summary["best_federated_accuracy"],
        "best_strategy": summary["best_strategy"],
    }

    if baseline_results.get("baselines_binary"):
        trial_result["baselines_binary"] = baseline_results["baselines_binary"]
    if baseline_results.get("baselines_int8"):
        trial_result["baselines_int8"] = baseline_results["baselines_int8"]

    return trial_result


def _aggregate_config_results(params: dict, trials: list) -> dict:
    """Aggregate trial results into a config result with summary statistics."""
    valid_trials = [t for t in trials if "error" not in t]

    ind_accs = [t["mean_individual_accuracy"] for t in valid_trials]
    best_accs = [t["best_federated_accuracy"] for t in valid_trials]

    # Mean federated accuracy across all strategies and nodes
    fed_accs = []
    for t in valid_trials:
        accs = [m.get("accuracy", 0) for m in t.get("federated", {}).values()
                if isinstance(m, dict) and "accuracy" in m]
        if accs:
            fed_accs.append(float(np.mean(accs)))

    return {
        "params": params,
        "trials": trials,
        "summary": {
            "mean_individual_acc": float(np.mean(ind_accs)) if ind_accs else 0.0,
            "std_individual_acc": float(np.std(ind_accs)) if ind_accs else 0.0,
            "mean_federated_acc": float(np.mean(fed_accs)) if fed_accs else 0.0,
            "std_federated_acc": float(np.std(fed_accs)) if fed_accs else 0.0,
            "mean_best_fed_acc": float(np.mean(best_accs)) if best_accs else 0.0,
            "std_best_fed_acc": float(np.std(best_accs)) if best_accs else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# Phase A: Fine-tuning
# ---------------------------------------------------------------------------

def run_phase_a(finetune_epochs: int, output_dir: Path):
    """Phase A: Fine-tune feature extractors.

    A1: Target classes (backward/follow/forward) -> feat_extractor.fbz
    A2: Disjoint classes (yes/no/stop) -> feat_extractor_disjoint.fbz
    A3: Wide 128-dim projection -> feat_extractor_wide128.fbz
    A4: Wide 256-dim projection -> feat_extractor_wide256.fbz
    """
    log.info("=" * 70)
    log.info("PHASE A: Fine-tuning feature extractors")
    log.info("=" * 70)

    model_dir = Path.home() / "federated_experiment" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    finetune_configs = [
        {
            "name": "A1_target",
            "output_model": "feat_extractor.fbz",
            "extra_flags": [],
            "description": "Target classes (backward/follow/forward)",
        },
        {
            "name": "A2_disjoint",
            "output_model": "feat_extractor_disjoint.fbz",
            "extra_flags": ["--finetune-classes", "yes,no,stop"],
            "description": "Disjoint classes (yes/no/stop)",
        },
        {
            "name": "A3_wide128",
            "output_model": "feat_extractor_wide128.fbz",
            "extra_flags": ["--feature-dim", "128"],
            "description": "Wide 128-dim projection",
        },
        {
            "name": "A4_wide256",
            "output_model": "feat_extractor_wide256.fbz",
            "extra_flags": ["--feature-dim", "256"],
            "description": "Wide 256-dim projection",
        },
    ]

    results = {}
    for ft_cfg in finetune_configs:
        model_path = model_dir / ft_cfg["output_model"]
        if model_path.exists():
            log.info("  %s: SKIP (already exists: %s)",
                     ft_cfg["name"], model_path)
            results[ft_cfg["name"]] = {"status": "skipped", "path": str(model_path)}
            continue

        log.info("  %s: %s", ft_cfg["name"], ft_cfg["description"])

        cmd = [
            sys.executable, "finetune_dscnn.py",
            "--epochs", str(finetune_epochs),
            "--deploy-to-paolo",
            "--output-model", ft_cfg["output_model"],
        ] + ft_cfg["extra_flags"]

        log.info("  Running: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, timeout=1800)
            results[ft_cfg["name"]] = {"status": "ok", "path": str(model_path)}
        except subprocess.CalledProcessError as e:
            log.error("  %s FAILED: %s", ft_cfg["name"], e)
            results[ft_cfg["name"]] = {"status": "error", "error": str(e)}
            continue
        except subprocess.TimeoutExpired:
            log.error("  %s TIMED OUT", ft_cfg["name"])
            results[ft_cfg["name"]] = {"status": "timeout"}
            continue

        # SCP the extractor to paolo (finetune_dscnn.py --deploy-to-paolo
        # should handle this, but ensure it's there)
        if model_path.exists():
            try:
                subprocess.run([
                    "scp", "-o", "StrictHostKeyChecking=no",
                    str(model_path),
                    "admin@10.0.0.2:~/federated_experiment/models/",
                ], check=True, timeout=60)
                log.info("  SCP %s to paolo OK", ft_cfg["output_model"])
            except subprocess.CalledProcessError as e:
                log.warning("  SCP to paolo failed: %s", e)

    return results


# ---------------------------------------------------------------------------
# Phase B: Full main sweep
# ---------------------------------------------------------------------------

def run_phase_b(
    config_module,
    num_trials: int,
    data_dir: Path,
    results_dir: Path,
    results_path: Path,
    all_results: dict,
    grid_subset: int | None = None,
):
    """Phase B: Extended hyperparameter grid sweep.

    42 configs x num_trials trials = total runs.
    """
    from federated_experiment.orchestrator import send_command

    log.info("=" * 70)
    log.info("PHASE B: Extended hyperparameter grid sweep")
    log.info("=" * 70)

    seeds = [42 + i for i in range(num_trials)]

    # Generate grid
    keys = sorted(EXTENDED_GRID.keys())
    combos = list(itertools.product(*(EXTENDED_GRID[k] for k in keys)))
    configs = [dict(zip(keys, combo)) for combo in combos]

    if grid_subset is not None:
        configs = configs[:grid_subset]

    log.info("Phase B: %d configs x %d trials = %d total runs",
             len(configs), num_trials, len(configs) * num_trials)

    # Resume: check completed configs
    completed = {}
    for cfg_result in all_results.get("phase_b", {}).get("configs", []):
        ckey = _config_key(cfg_result["params"])
        if len(cfg_result.get("trials", [])) >= num_trials:
            completed[ckey] = cfg_result

    config_results = list(completed.values())

    for cfg_idx, params in enumerate(configs):
        ckey = _config_key(params)
        if ckey in completed:
            log.info("[B %d/%d] SKIP (done): %s", cfg_idx + 1, len(configs), params)
            continue

        log.info("=" * 60)
        log.info("[B %d/%d] Config: %s", cfg_idx + 1, len(configs), params)
        log.info("=" * 60)

        trials = []
        for trial_idx, seed in enumerate(seeds):
            log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, num_trials, seed)
            trial_result = _run_single_trial(
                seed=seed,
                params=params,
                config_module=config_module,
                data_dir=data_dir,
                results_dir=results_dir,
                run_baselines=True,
                num_classes=config_module.NUM_NEW_CLASSES,
            )
            trials.append(trial_result)

        config_result = _aggregate_config_results(params, trials)
        config_results.append(config_result)

        # Incremental save
        all_results["phase_b"] = {
            "grid": EXTENDED_GRID,
            "num_configs": len(configs),
            "num_trials": num_trials,
            "configs": config_results,
        }
        _save_results(results_path, all_results)

        s = config_result["summary"]
        log.info("  Summary: ind=%.1f%%+/-%.1f%%, best_fed=%.1f%%+/-%.1f%%",
                 s["mean_individual_acc"] * 100, s["std_individual_acc"] * 100,
                 s["mean_best_fed_acc"] * 100, s["std_best_fed_acc"] * 100)

    # Final update
    all_results["phase_b"] = {
        "grid": EXTENDED_GRID,
        "num_configs": len(configs),
        "num_trials": num_trials,
        "configs": config_results,
    }
    _save_results(results_path, all_results)

    return config_results


# ---------------------------------------------------------------------------
# Phase C: Binarization comparison
# ---------------------------------------------------------------------------

def run_phase_c(
    config_module,
    num_trials: int,
    data_dir: Path,
    results_dir: Path,
    results_path: Path,
    all_results: dict,
):
    """Phase C: Binarization method comparison.

    Top 5 configs from Phase B x 3 methods x num_trials trials = 150 runs.
    """
    log.info("=" * 70)
    log.info("PHASE C: Binarization comparison")
    log.info("=" * 70)

    phase_b_configs = all_results.get("phase_b", {}).get("configs", [])
    if not phase_b_configs:
        log.warning("Phase C requires Phase B results. Skipping.")
        return []

    top_configs = _get_top_configs(phase_b_configs, TOP_N_FOR_BINARIZATION)
    seeds = [42 + i for i in range(num_trials)]

    log.info("Phase C: %d configs x %d methods x %d trials = %d runs",
             len(top_configs), len(BINARIZATION_METHODS), num_trials,
             len(top_configs) * len(BINARIZATION_METHODS) * num_trials)

    # Resume: support both skip (completed) and extend (partial) configs
    existing_configs = all_results.get("phase_c", {}).get("configs", [])
    completed, partial = _resume_state(existing_configs, num_trials)

    config_results = list(completed.values())

    for rank_idx, top_cfg in enumerate(top_configs):
        base_params = top_cfg["params"]

        for method in BINARIZATION_METHODS:
            params = dict(base_params)
            params["binarization_method"] = method
            ckey = _config_key(params)

            if ckey in completed:
                log.info("[C %d/%d] SKIP (done): %s + %s",
                         rank_idx + 1, len(top_configs), base_params, method)
                continue

            # Start from existing trials if partially completed
            existing_trials = []
            if ckey in partial:
                existing_trials = partial[ckey].get("trials", [])
                existing_seeds = {t["seed"] for t in existing_trials}
                log.info("[C top-%d] EXTEND (%d→%d): %s + %s",
                         rank_idx + 1, len(existing_trials), num_trials,
                         base_params, method)
            else:
                existing_seeds = set()
                log.info("[C top-%d] %s + binarization=%s",
                         rank_idx + 1, base_params, method)

            trials = list(existing_trials)
            for trial_idx, seed in enumerate(seeds):
                if seed in existing_seeds:
                    continue
                log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, num_trials, seed)
                trial_result = _run_single_trial(
                    seed=seed,
                    params=base_params,
                    config_module=config_module,
                    data_dir=data_dir,
                    results_dir=results_dir,
                    extra_build_params={"binarization_method": method},
                    run_baselines=False,
                    num_classes=config_module.NUM_NEW_CLASSES,
                )
                trials.append(trial_result)

            config_result = _aggregate_config_results(params, trials)
            config_results.append(config_result)

            all_results["phase_c"] = {
                "description": "Binarization method comparison",
                "top_n": TOP_N_FOR_BINARIZATION,
                "methods": BINARIZATION_METHODS,
                "num_trials": num_trials,
                "configs": config_results,
            }
            _save_results(results_path, all_results)

    all_results["phase_c"] = {
        "description": "Binarization method comparison",
        "top_n": TOP_N_FOR_BINARIZATION,
        "methods": BINARIZATION_METHODS,
        "num_trials": num_trials,
        "configs": config_results,
    }
    _save_results(results_path, all_results)

    return config_results


# ---------------------------------------------------------------------------
# Phase D: Disjoint extractor
# ---------------------------------------------------------------------------

def run_phase_d(
    config_module,
    num_trials: int,
    data_dir: Path,
    results_dir: Path,
    results_path: Path,
    all_results: dict,
):
    """Phase D: Disjoint feature extractor.

    Top 5 configs from Phase B x num_trials trials = 50 runs.
    Uses feat_extractor_disjoint.fbz (fine-tuned on yes/no/stop).
    """
    log.info("=" * 70)
    log.info("PHASE D: Disjoint feature extractor")
    log.info("=" * 70)

    phase_b_configs = all_results.get("phase_b", {}).get("configs", [])
    if not phase_b_configs:
        log.warning("Phase D requires Phase B results. Skipping.")
        return []

    # Check that the disjoint extractor exists
    disjoint_path = Path.home() / "federated_experiment" / "models" / "feat_extractor_disjoint.fbz"
    if not disjoint_path.exists():
        log.warning("Phase D requires feat_extractor_disjoint.fbz. Run Phase A first. Skipping.")
        return []

    top_configs = _get_top_configs(phase_b_configs, TOP_N_FOR_DISJOINT)
    seeds = [42 + i for i in range(num_trials)]

    log.info("Phase D: %d configs x %d trials = %d runs",
             len(top_configs), num_trials, len(top_configs) * num_trials)

    # Resume
    completed = {}
    for cfg_result in all_results.get("phase_d", {}).get("configs", []):
        ckey = _config_key(cfg_result["params"])
        if len(cfg_result.get("trials", [])) >= num_trials:
            completed[ckey] = cfg_result

    config_results = list(completed.values())

    for rank_idx, top_cfg in enumerate(top_configs):
        base_params = top_cfg["params"]
        params = dict(base_params)
        params["feat_extractor_name"] = "feat_extractor_disjoint.fbz"
        ckey = _config_key(params)

        if ckey in completed:
            log.info("[D %d/%d] SKIP (done): %s",
                     rank_idx + 1, len(top_configs), params)
            continue

        log.info("[D top-%d] %s + disjoint extractor", rank_idx + 1, base_params)

        trials = []
        for trial_idx, seed in enumerate(seeds):
            log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, num_trials, seed)
            trial_result = _run_single_trial(
                seed=seed,
                params=base_params,
                config_module=config_module,
                data_dir=data_dir,
                results_dir=results_dir,
                extra_build_params={
                    "feat_extractor_name": "feat_extractor_disjoint.fbz",
                },
                run_baselines=False,
                num_classes=config_module.NUM_NEW_CLASSES,
            )
            trials.append(trial_result)

        config_result = _aggregate_config_results(params, trials)
        config_results.append(config_result)

        all_results["phase_d"] = {
            "description": "Disjoint feature extractor (yes/no/stop)",
            "top_n": TOP_N_FOR_DISJOINT,
            "num_trials": num_trials,
            "configs": config_results,
        }
        _save_results(results_path, all_results)

    all_results["phase_d"] = {
        "description": "Disjoint feature extractor (yes/no/stop)",
        "top_n": TOP_N_FOR_DISJOINT,
        "num_trials": num_trials,
        "configs": config_results,
    }
    _save_results(results_path, all_results)

    return config_results


# ---------------------------------------------------------------------------
# Phase E: Wide features
# ---------------------------------------------------------------------------

def run_phase_e(
    config_module,
    num_trials: int,
    data_dir: Path,
    results_dir: Path,
    results_path: Path,
    all_results: dict,
):
    """Phase E: Wide feature extractors (128 and 256 dimensions).

    E1: feat_extractor_wide128.fbz with num_weights grid, best npc+lc -> 7x10=70
    E2: feat_extractor_wide256.fbz with num_weights grid, best npc+lc -> 6x10=60
    Total: 130 runs.
    """
    log.info("=" * 70)
    log.info("PHASE E: Wide feature extractors")
    log.info("=" * 70)

    phase_b_configs = all_results.get("phase_b", {}).get("configs", [])
    if not phase_b_configs:
        log.warning("Phase E requires Phase B results. Skipping.")
        return []

    # Get best npc and lc from Phase B top-1
    top1 = _get_top_configs(phase_b_configs, 1)[0]
    best_npc = top1["params"]["neurons_per_class"]
    best_lc = top1["params"]["learning_competition"]
    log.info("Phase E using best npc=%d, lc=%.1f from Phase B", best_npc, best_lc)

    seeds = [42 + i for i in range(num_trials)]

    # Resume: support both skip (completed) and extend (partial) configs
    existing_configs = all_results.get("phase_e", {}).get("configs", [])
    completed, partial = _resume_state(existing_configs, num_trials)

    config_results = list(completed.values())

    # E1: Wide 128
    wide128_path = Path.home() / "federated_experiment" / "models" / "feat_extractor_wide128.fbz"
    if wide128_path.exists():
        log.info("--- E1: feat_extractor_wide128.fbz ---")
        for nw_idx, nw in enumerate(WIDE_128_NUM_WEIGHTS):
            params = {
                "num_weights": nw,
                "neurons_per_class": best_npc,
                "learning_competition": best_lc,
                "feat_extractor_name": "feat_extractor_wide128.fbz",
            }
            ckey = _config_key(params)

            if ckey in completed:
                log.info("[E1 %d/%d] SKIP (done): nw=%d",
                         nw_idx + 1, len(WIDE_128_NUM_WEIGHTS), nw)
                continue

            base_params = {
                "num_weights": nw,
                "neurons_per_class": best_npc,
                "learning_competition": best_lc,
            }

            # Start from existing trials if partially completed
            existing_trials = []
            if ckey in partial:
                existing_trials = partial[ckey].get("trials", [])
                existing_seeds = {t["seed"] for t in existing_trials}
                log.info("[E1 %d/%d] EXTEND (%d→%d): nw=%d",
                         nw_idx + 1, len(WIDE_128_NUM_WEIGHTS),
                         len(existing_trials), num_trials, nw)
            else:
                existing_seeds = set()
                log.info("[E1 %d/%d] nw=%d, npc=%d, lc=%.1f + wide128",
                         nw_idx + 1, len(WIDE_128_NUM_WEIGHTS), nw, best_npc, best_lc)

            trials = list(existing_trials)
            for trial_idx, seed in enumerate(seeds):
                if seed in existing_seeds:
                    continue
                log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, num_trials, seed)
                trial_result = _run_single_trial(
                    seed=seed,
                    params=base_params,
                    config_module=config_module,
                    data_dir=data_dir,
                    results_dir=results_dir,
                    extra_build_params={
                        "feat_extractor_name": "feat_extractor_wide128.fbz",
                    },
                    run_baselines=False,
                    num_classes=config_module.NUM_NEW_CLASSES,
                )
                trials.append(trial_result)

            config_result = _aggregate_config_results(params, trials)
            config_results.append(config_result)

            all_results["phase_e"] = {
                "description": "Wide feature extractors",
                "best_npc": best_npc,
                "best_lc": best_lc,
                "num_trials": num_trials,
                "configs": config_results,
            }
            _save_results(results_path, all_results)
    else:
        log.warning("E1: feat_extractor_wide128.fbz not found, skipping. Run Phase A first.")

    # E2: Wide 256
    wide256_path = Path.home() / "federated_experiment" / "models" / "feat_extractor_wide256.fbz"
    if wide256_path.exists():
        log.info("--- E2: feat_extractor_wide256.fbz ---")
        for nw_idx, nw in enumerate(WIDE_256_NUM_WEIGHTS):
            params = {
                "num_weights": nw,
                "neurons_per_class": best_npc,
                "learning_competition": best_lc,
                "feat_extractor_name": "feat_extractor_wide256.fbz",
            }
            ckey = _config_key(params)

            if ckey in completed:
                log.info("[E2 %d/%d] SKIP (done): nw=%d",
                         nw_idx + 1, len(WIDE_256_NUM_WEIGHTS), nw)
                continue

            base_params = {
                "num_weights": nw,
                "neurons_per_class": best_npc,
                "learning_competition": best_lc,
            }

            # Start from existing trials if partially completed
            existing_trials = []
            if ckey in partial:
                existing_trials = partial[ckey].get("trials", [])
                existing_seeds = {t["seed"] for t in existing_trials}
                log.info("[E2 %d/%d] EXTEND (%d→%d): nw=%d",
                         nw_idx + 1, len(WIDE_256_NUM_WEIGHTS),
                         len(existing_trials), num_trials, nw)
            else:
                existing_seeds = set()
                log.info("[E2 %d/%d] nw=%d, npc=%d, lc=%.1f + wide256",
                         nw_idx + 1, len(WIDE_256_NUM_WEIGHTS), nw, best_npc, best_lc)

            trials = list(existing_trials)
            for trial_idx, seed in enumerate(seeds):
                if seed in existing_seeds:
                    continue
                log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, num_trials, seed)
                trial_result = _run_single_trial(
                    seed=seed,
                    params=base_params,
                    config_module=config_module,
                    data_dir=data_dir,
                    results_dir=results_dir,
                    extra_build_params={
                        "feat_extractor_name": "feat_extractor_wide256.fbz",
                    },
                    run_baselines=False,
                    num_classes=config_module.NUM_NEW_CLASSES,
                )
                trials.append(trial_result)

            config_result = _aggregate_config_results(params, trials)
            config_results.append(config_result)

            all_results["phase_e"] = {
                "description": "Wide feature extractors",
                "best_npc": best_npc,
                "best_lc": best_lc,
                "num_trials": num_trials,
                "configs": config_results,
            }
            _save_results(results_path, all_results)
    else:
        log.warning("E2: feat_extractor_wide256.fbz not found, skipping. Run Phase A first.")

    # Final update
    all_results["phase_e"] = {
        "description": "Wide feature extractors",
        "best_npc": best_npc,
        "best_lc": best_lc,
        "num_trials": num_trials,
        "configs": config_results,
    }
    _save_results(results_path, all_results)

    return config_results


# ---------------------------------------------------------------------------
# Phase F: Multi-round federation
# ---------------------------------------------------------------------------

def run_phase_f(
    config_module,
    num_trials: int,
    data_dir: Path,
    results_dir: Path,
    results_path: Path,
    all_results: dict,
):
    """Phase F: Multi-round federation with retraining.

    Top 3 configs from Phase B x 2 strategies x num_trials trials x 5 rounds.
    """
    from federated_experiment.orchestrator import (
        prepare_data,
        deploy_to_nodes,
        send_command,
        run_local_training,
        run_individual_evaluation,
        run_weight_exchange,
        run_multi_round_federation,
    )

    log.info("=" * 70)
    log.info("PHASE F: Multi-round federation")
    log.info("=" * 70)

    phase_b_configs = all_results.get("phase_b", {}).get("configs", [])
    if not phase_b_configs:
        log.warning("Phase F requires Phase B results. Skipping.")
        return []

    top_configs = _get_top_configs(phase_b_configs, TOP_N_FOR_MULTI_ROUND)
    seeds = [42 + i for i in range(num_trials)]

    log.info("Phase F: %d configs x %d strategies x %d trials x %d rounds",
             len(top_configs), len(RETRAINING_STRATEGIES), num_trials,
             MULTI_ROUND_ROUNDS)

    # Resume
    completed = {}
    for cfg_result in all_results.get("phase_f", {}).get("configs", []):
        ckey = _config_key(cfg_result["params"])
        if len(cfg_result.get("trials", [])) >= num_trials:
            completed[ckey] = cfg_result

    config_results = list(completed.values())

    for rank_idx, top_cfg in enumerate(top_configs):
        base_params = top_cfg["params"]

        for retrain_strategy in RETRAINING_STRATEGIES:
            params = dict(base_params)
            params["retraining_strategy"] = retrain_strategy
            ckey = _config_key(params)

            if ckey in completed:
                log.info("[F %d/%d] SKIP (done): %s + %s",
                         rank_idx + 1, len(top_configs),
                         base_params, retrain_strategy)
                continue

            log.info("[F top-%d] %s + retrain=%s",
                     rank_idx + 1, base_params, retrain_strategy)

            trials = []
            for trial_idx, seed in enumerate(seeds):
                log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, num_trials, seed)

                trial_start = time.time()

                # Prepare data
                prepare_data(data_dir, seed=seed, use_calibration=False)
                deploy_to_nodes(data_dir)

                # Build edge models with these params
                for name in ("claudio", "paolo"):
                    resp = send_command(config_module.get_command_ip(name), {
                        "action": "build_model",
                        "params": base_params,
                    })
                    if resp.get("status") != "ok":
                        log.error("build_model failed on %s: %s", name, resp)

                # Train + individual eval
                run_local_training()
                individual_results = run_individual_evaluation()

                # Weight exchange
                exchange_data = run_weight_exchange()

                # Multi-round federation with specified retraining strategy
                round_results = run_multi_round_federation(
                    MULTI_ROUND_ROUNDS,
                    exchange_data,
                    retraining_strategy=retrain_strategy,
                )

                trial_time = time.time() - trial_start

                trial_result = {
                    "seed": seed,
                    "params": base_params,
                    "retraining_strategy": retrain_strategy,
                    "trial_time_seconds": trial_time,
                    "individual": individual_results,
                    "rounds": round_results,
                }

                # Compute summary from individual results
                ind_accs = [
                    individual_results.get(n, {}).get("accuracy", 0)
                    for n in ("claudio", "paolo")
                ]
                trial_result["mean_individual_accuracy"] = float(np.mean(ind_accs))

                # Best accuracy from final round
                final_round_key = f"round_{MULTI_ROUND_ROUNDS}"
                final_round = round_results.get(final_round_key, {})
                best_fed = 0.0
                best_strat = ""
                for strategy, strat_data in final_round.items():
                    if isinstance(strat_data, dict):
                        for node_name, metrics in strat_data.items():
                            if isinstance(metrics, dict):
                                acc = metrics.get("accuracy", 0)
                                if acc > best_fed:
                                    best_fed = acc
                                    best_strat = strategy
                trial_result["best_federated_accuracy"] = float(best_fed)
                trial_result["best_strategy"] = best_strat

                trials.append(trial_result)

            config_result = _aggregate_config_results(params, trials)
            config_results.append(config_result)

            all_results["phase_f"] = {
                "description": "Multi-round federation",
                "top_n": TOP_N_FOR_MULTI_ROUND,
                "num_rounds": MULTI_ROUND_ROUNDS,
                "retraining_strategies": RETRAINING_STRATEGIES,
                "num_trials": num_trials,
                "configs": config_results,
            }
            _save_results(results_path, all_results)

    # Final update
    all_results["phase_f"] = {
        "description": "Multi-round federation",
        "top_n": TOP_N_FOR_MULTI_ROUND,
        "num_rounds": MULTI_ROUND_ROUNDS,
        "retraining_strategies": RETRAINING_STRATEGIES,
        "num_trials": num_trials,
        "configs": config_results,
    }
    _save_results(results_path, all_results)

    return config_results


# ---------------------------------------------------------------------------
# Phase G: Entropy binarization + wide-256 features (combined)
# ---------------------------------------------------------------------------

def run_phase_g(
    config_module,
    num_trials: int,
    data_dir: Path,
    results_dir: Path,
    results_path: Path,
    all_results: dict,
):
    """Phase G: Entropy binarization on wide-256 features.

    Combines the two most effective mitigations for binarization loss:
    - Wide 256-dim features (from Phase E)
    - Entropy binarization (from Phase C)

    Uses same nw grid as Phase E wide-256, with best npc+lc from Phase B.
    6 configs x num_trials trials = 60 runs.
    """
    log.info("=" * 70)
    log.info("PHASE G: Entropy binarization + wide-256 features")
    log.info("=" * 70)

    phase_b_configs = all_results.get("phase_b", {}).get("configs", [])
    if not phase_b_configs:
        log.warning("Phase G requires Phase B results. Skipping.")
        return []

    # Check that the wide-256 extractor exists
    wide256_path = (Path.home() / "federated_experiment" / "models"
                    / "feat_extractor_wide256.fbz")
    if not wide256_path.exists():
        log.warning("Phase G requires feat_extractor_wide256.fbz. "
                     "Run Phase A first. Skipping.")
        return []

    # Get best npc and lc from Phase B top-1
    top1 = _get_top_configs(phase_b_configs, 1)[0]
    best_npc = top1["params"]["neurons_per_class"]
    best_lc = top1["params"]["learning_competition"]
    log.info("Phase G using best npc=%d, lc=%.1f from Phase B", best_npc, best_lc)

    seeds = [42 + i for i in range(num_trials)]

    log.info("Phase G: %d configs x %d trials = %d runs",
             len(WIDE_256_ENTROPY_NUM_WEIGHTS), num_trials,
             len(WIDE_256_ENTROPY_NUM_WEIGHTS) * num_trials)

    # Resume — support extending partial configs (e.g. 10→30 trials)
    existing_configs = all_results.get("phase_g", {}).get("configs", [])
    completed, partial = _resume_state(existing_configs, num_trials)

    config_results = list(completed.values())

    for nw_idx, nw in enumerate(WIDE_256_ENTROPY_NUM_WEIGHTS):
        params = {
            "num_weights": nw,
            "neurons_per_class": best_npc,
            "learning_competition": best_lc,
            "feat_extractor_name": "feat_extractor_wide256.fbz",
            "binarization_method": "entropy",
        }
        ckey = _config_key(params)

        if ckey in completed:
            log.info("[G %d/%d] SKIP (done): nw=%d",
                     nw_idx + 1, len(WIDE_256_ENTROPY_NUM_WEIGHTS), nw)
            continue

        # Check for partial results to extend
        existing_trials = []
        existing_seeds = set()
        if ckey in partial:
            existing_trials = list(partial[ckey].get("trials", []))
            existing_seeds = {t["seed"] for t in existing_trials}
            log.info("[G %d/%d] Extending nw=%d from %d to %d trials",
                     nw_idx + 1, len(WIDE_256_ENTROPY_NUM_WEIGHTS),
                     nw, len(existing_trials), num_trials)
        else:
            log.info("[G %d/%d] nw=%d, npc=%d, lc=%.1f + wide256 + entropy",
                     nw_idx + 1, len(WIDE_256_ENTROPY_NUM_WEIGHTS),
                     nw, best_npc, best_lc)

        base_params = {
            "num_weights": nw,
            "neurons_per_class": best_npc,
            "learning_competition": best_lc,
        }

        trials = list(existing_trials)
        for trial_idx, seed in enumerate(seeds):
            if seed in existing_seeds:
                continue
            log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, num_trials, seed)
            trial_result = _run_single_trial(
                seed=seed,
                params=base_params,
                config_module=config_module,
                data_dir=data_dir,
                results_dir=results_dir,
                extra_build_params={
                    "feat_extractor_name": "feat_extractor_wide256.fbz",
                    "binarization_method": "entropy",
                },
                run_baselines=False,
                num_classes=config_module.NUM_NEW_CLASSES,
            )
            trials.append(trial_result)

        config_result = _aggregate_config_results(params, trials)
        config_results.append(config_result)

        all_results["phase_g"] = {
            "description": "Entropy binarization + wide-256 features",
            "best_npc": best_npc,
            "best_lc": best_lc,
            "num_trials": num_trials,
            "configs": config_results,
        }
        _save_results(results_path, all_results)

    # Final update
    all_results["phase_g"] = {
        "description": "Entropy binarization + wide-256 features",
        "best_npc": best_npc,
        "best_lc": best_lc,
        "num_trials": num_trials,
        "configs": config_results,
    }
    _save_results(results_path, all_results)

    return config_results


# ---------------------------------------------------------------------------
# Phase H: N-node simulation
# ---------------------------------------------------------------------------

# Default configs for Phase H
PHASE_H_NUM_NODES = 4
PHASE_H_TOP_N = 3
PHASE_H_N2_STRATEGIES = ["fedunion", "fedavg"]


def _prepare_n_node_data(
    data_dir: Path,
    raw_dir: Path,
    seed: int,
    num_nodes: int,
    novel_classes: list[str],
    samples_per_class: int = 50,
):
    """Prepare data splits for N virtual nodes.

    With N=4 and 50 samples/class total:
      VN0 (claudio-type): backward[0:25] + follow[0:25]
      VN1 (claudio-type): backward[25:50] + follow[25:50]
      VN2 (paolo-type):   backward[50:75] + forward[0:25]
      VN3 (paolo-type):   backward[75:100] + forward[25:50]

    Each virtual node gets samples_per_class//(num_nodes//2) samples per class.
    Returns list of (node_type, data_files_prefix) pairs for physical execution.
    """
    from federated_experiment.data_loader import load_class_data, prepare_eval_dataset

    n_per_type = num_nodes // 2  # Half claudio-type, half paolo-type
    samples_per_vn = samples_per_class // n_per_type

    shared_class = novel_classes[0]      # backward
    claudio_exclusive = novel_classes[1]  # follow
    paolo_exclusive = novel_classes[2]    # forward

    vn_configs = []
    # First half: claudio-type (shared + claudio_exclusive)
    for i in range(n_per_type):
        vn_configs.append({
            "vn_id": i,
            "node_type": "claudio",
            "classes": [shared_class, claudio_exclusive],
            "shared_offset": i * samples_per_vn,
            "exclusive_offset": i * samples_per_vn,
        })
    # Second half: paolo-type (shared + paolo_exclusive)
    for i in range(n_per_type):
        vn_configs.append({
            "vn_id": n_per_type + i,
            "node_type": "paolo",
            "classes": [shared_class, paolo_exclusive],
            "shared_offset": (n_per_type + i) * samples_per_vn,
            "exclusive_offset": i * samples_per_vn,
        })

    # Generate data files for each virtual node
    for vn in vn_configs:
        all_X, all_y = [], []
        for cls_name in vn["classes"]:
            global_label = novel_classes.index(cls_name)
            if cls_name == shared_class:
                offset = vn["shared_offset"]
            else:
                offset = vn["exclusive_offset"]
            X = load_class_data(
                raw_dir, cls_name,
                max_samples=samples_per_vn,
                seed=seed,
                offset=offset,
            )
            y = np.full(len(X), global_label, dtype=np.int32)
            all_X.append(X)
            all_y.append(y)
            log.info("VN%d class %s (label %d): %d samples (offset=%d)",
                     vn["vn_id"], cls_name, global_label, len(X), offset)

        X_concat = np.concatenate(all_X)
        y_concat = np.concatenate(all_y)

        # Save as physical node name for the pair it will run on
        prefix = f"vn{vn['vn_id']}"
        np.save(data_dir / f"{prefix}_X_train.npy", X_concat)
        np.save(data_dir / f"{prefix}_y_train.npy", y_concat)

        import json as _json
        with open(data_dir / f"{prefix}_classes.json", "w") as f:
            _json.dump(vn["classes"], f)

    # Prepare eval set (same as standard — offset past all training data)
    train_offset = samples_per_class * 2  # Max samples used across all VNs
    X_eval, y_eval, _ = prepare_eval_dataset(
        raw_dir, seed=seed, novel_classes=novel_classes,
        train_offset=train_offset,
    )
    np.save(data_dir / "eval_X.npy", X_eval)
    np.save(data_dir / "eval_y.npy", y_eval)

    return vn_configs


def run_phase_h(
    config_module,
    num_trials: int,
    data_dir: Path,
    results_dir: Path,
    results_path: Path,
    all_results: dict,
    num_nodes: int = PHASE_H_NUM_NODES,
):
    """Phase H: N-node simulation.

    Simulates N virtual nodes using 2 physical nodes in series.
    For N=4: runs 2 sequential pair experiments, collects 4 weight sets,
    then federates across all N using FedUnion-N and FedAvg-N.

    Also runs a within-trial N=2 baseline (VN0+VN2 merge) for direct comparison.

    Uses top configs from Phase B.
    """
    from federated_experiment.orchestrator import (
        send_command,
        deploy_to_nodes,
        run_local_training,
    )
    from federated_experiment.federation import (
        NodeWeights,
        fedunion_n,
        fedavg_n,
    )

    log.info("=" * 70)
    log.info("PHASE H: N-node simulation (N=%d)", num_nodes)
    log.info("=" * 70)

    phase_b_configs = all_results.get("phase_b", {}).get("configs", [])
    if not phase_b_configs:
        log.warning("Phase H requires Phase B results. Skipping.")
        return []

    if num_nodes % 2 != 0:
        log.error("num_nodes must be even (got %d). Skipping.", num_nodes)
        return []

    top_configs = _get_top_configs(phase_b_configs, PHASE_H_TOP_N)
    seeds = [42 + i for i in range(num_trials)]

    n_pairs = num_nodes // 2
    log.info("Phase H: %d configs x %d trials x %d pairs = %d physical runs",
             len(top_configs), num_trials, n_pairs,
             len(top_configs) * num_trials * n_pairs)

    # Resume
    existing_configs = all_results.get("phase_h", {}).get("configs", [])
    completed, partial = _resume_state(existing_configs, num_trials)

    config_results = list(completed.values())

    raw_dir = data_dir / "raw"

    for rank_idx, top_cfg in enumerate(top_configs):
        base_params = top_cfg["params"]
        params = dict(base_params)
        params["num_nodes"] = num_nodes
        ckey = _config_key(params)

        if ckey in completed:
            log.info("[H %d/%d] SKIP (done): %s",
                     rank_idx + 1, len(top_configs), base_params)
            continue

        # Support trial extension
        existing_trials = []
        if ckey in partial:
            existing_trials = partial[ckey].get("trials", [])
            existing_seeds = {t["seed"] for t in existing_trials}
            log.info("[H %d/%d] EXTEND (%d→%d): %s",
                     rank_idx + 1, len(top_configs),
                     len(existing_trials), num_trials, base_params)
        else:
            existing_seeds = set()
            log.info("[H %d/%d] Config: %s (N=%d nodes)",
                     rank_idx + 1, len(top_configs), base_params, num_nodes)

        trials = list(existing_trials)

        for trial_idx, seed in enumerate(seeds):
            if seed in existing_seeds:
                continue

            log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, num_trials, seed)
            trial_start = time.time()

            # 1. Prepare N-way data splits
            vn_configs = _prepare_n_node_data(
                data_dir, raw_dir, seed, num_nodes,
                config_module.NOVEL_CLASSES,
                samples_per_class=config_module.SPLIT.samples_per_class,
            )

            # 2. Run sequential pair experiments on 2 physical nodes
            all_node_weights = []
            pair_results = {}

            for pair_idx in range(n_pairs):
                vn_a = vn_configs[pair_idx * 2]      # Goes on claudio
                vn_b = vn_configs[pair_idx * 2 + 1]  # Goes on paolo

                log.info("  Pair %d/%d: VN%d on claudio, VN%d on paolo",
                         pair_idx + 1, n_pairs, vn_a["vn_id"], vn_b["vn_id"])

                # Deploy VN data to physical nodes
                import shutil
                for phys_name, vn in [("claudio", vn_a), ("paolo", vn_b)]:
                    prefix = f"vn{vn['vn_id']}"
                    for suffix in ("_X_train.npy", "_y_train.npy", "_classes.json"):
                        src = data_dir / f"{prefix}{suffix}"
                        dst = data_dir / f"{phys_name}{suffix}"
                        shutil.copy2(str(src), str(dst))

                deploy_to_nodes(data_dir)

                # Build models on both nodes
                for name in ("claudio", "paolo"):
                    resp = send_command(config_module.get_command_ip(name), {
                        "action": "build_model",
                        "params": base_params,
                    })
                    if resp.get("status") != "ok":
                        log.error("build_model failed on %s: %s", name, resp)

                # Train on both nodes
                for name in ("claudio", "paolo"):
                    resp = send_command(
                        config_module.get_command_ip(name),
                        {"action": "train"}, timeout=120)
                    log.info("  VN%d (%s) trained: %s",
                             vn_a["vn_id"] if name == "claudio" else vn_b["vn_id"],
                             name, resp.get("metrics", {}).get("learning_time_ms", "?"))

                # Extract weights from both physical nodes
                for name, vn in [("claudio", vn_a), ("paolo", vn_b)]:
                    resp = send_command(
                        config_module.get_command_ip(name),
                        {"action": "get_weights"})
                    if resp.get("status") == "ok":
                        payload = resp["payload"]
                        w = np.array(payload["weights"], dtype=np.int8)
                        cm = {int(k): v
                              for k, v in payload["class_map"].items()}
                        nw = NodeWeights(
                            weights=w,
                            class_map=cm,
                            node_id=f"vn{vn['vn_id']}",
                            classes_seen=payload["classes_seen"],
                        )
                        all_node_weights.append(nw)
                    else:
                        log.error("get_weights failed on %s: %s", name, resp)

                # Individual evaluation on this pair
                for name in ("claudio", "paolo"):
                    resp = send_command(
                        config_module.get_command_ip(name),
                        {"action": "evaluate",
                         "num_classes": config_module.NUM_NEW_CLASSES},
                        timeout=120)
                    vn_id = vn_a["vn_id"] if name == "claudio" else vn_b["vn_id"]
                    pair_results[f"vn{vn_id}_individual"] = resp.get("metrics", {})

            # 3. N-node federation
            if len(all_node_weights) < num_nodes:
                log.error("Only got %d/%d weight sets, skipping federation",
                          len(all_node_weights), num_nodes)
                trials.append({"seed": seed, "error": "insufficient weights",
                                "params": base_params})
                continue

            federation_results = {}

            # N-node strategies
            for strategy in PHASE_H_N2_STRATEGIES:
                log.info("  Federation: %s-N%d", strategy, num_nodes)
                if strategy == "fedunion":
                    merged = fedunion_n(all_node_weights,
                                       config_module.NOVEL_CLASSES)
                else:
                    merged = fedavg_n(all_node_weights,
                                      config_module.NOVEL_CLASSES)

                # Evaluate on claudio (inject merged weights)
                resp = send_command(
                    config_module.get_command_ip("claudio"),
                    {"action": "set_weights", "weights": merged.tolist()})
                eval_resp = send_command(
                    config_module.get_command_ip("claudio"),
                    {"action": "evaluate",
                     "num_classes": config_module.NUM_NEW_CLASSES},
                    timeout=120)
                federation_results[f"{strategy}_n{num_nodes}"] = (
                    eval_resp.get("metrics", {}))
                log.info("    %s-N%d: %.1f%%", strategy, num_nodes,
                         eval_resp.get("metrics", {}).get("accuracy", 0) * 100)

            # 4. N=2 baseline within this trial (VN0 + VN2 — one claudio-type
            #    and one paolo-type from different pairs)
            from federated_experiment.federation import fedunion, fedavg
            for strategy_fn, strategy_name in [(fedunion, "fedunion"),
                                                (fedavg, "fedavg")]:
                merged_2 = strategy_fn(all_node_weights[0],
                                       all_node_weights[n_pairs],
                                       config_module.NOVEL_CLASSES)
                resp = send_command(
                    config_module.get_command_ip("claudio"),
                    {"action": "set_weights", "weights": merged_2.tolist()})
                eval_resp = send_command(
                    config_module.get_command_ip("claudio"),
                    {"action": "evaluate",
                     "num_classes": config_module.NUM_NEW_CLASSES},
                    timeout=120)
                federation_results[f"{strategy_name}_n2_baseline"] = (
                    eval_resp.get("metrics", {}))
                log.info("    %s-N2 baseline: %.1f%%", strategy_name,
                         eval_resp.get("metrics", {}).get("accuracy", 0) * 100)

            trial_time = time.time() - trial_start

            # Compute summary metrics
            ind_accs = [
                pair_results.get(f"vn{i}_individual", {}).get("accuracy", 0)
                for i in range(num_nodes)
            ]
            mean_ind = float(np.mean(ind_accs)) if ind_accs else 0.0

            # Best N-node federated accuracy
            best_fed_n = 0.0
            best_strat_n = ""
            for k, v in federation_results.items():
                if f"_n{num_nodes}" in k:
                    acc = v.get("accuracy", 0)
                    if acc > best_fed_n:
                        best_fed_n = acc
                        best_strat_n = k

            trial_result = {
                "seed": seed,
                "params": base_params,
                "num_nodes": num_nodes,
                "trial_time_seconds": trial_time,
                "individual": pair_results,
                "federation": federation_results,
                "mean_individual_accuracy": mean_ind,
                "best_federated_accuracy": float(best_fed_n),
                "best_strategy": best_strat_n,
                "node_weights_shapes": [
                    {"vn_id": nw.node_id, "shape": list(nw.weights.shape)}
                    for nw in all_node_weights
                ],
            }
            trials.append(trial_result)

        config_result = _aggregate_config_results(params, trials)
        config_results.append(config_result)

        all_results["phase_h"] = {
            "description": f"N-node simulation (N={num_nodes})",
            "num_nodes": num_nodes,
            "top_n": PHASE_H_TOP_N,
            "num_trials": num_trials,
            "configs": config_results,
        }
        _save_results(results_path, all_results)

    # Final update
    all_results["phase_h"] = {
        "description": f"N-node simulation (N={num_nodes})",
        "num_nodes": num_nodes,
        "top_n": PHASE_H_TOP_N,
        "num_trials": num_trials,
        "configs": config_results,
    }
    _save_results(results_path, all_results)

    return config_results


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def _print_phase_b_summary(config_results: list):
    """Print a summary table of Phase B results."""
    if not config_results:
        return

    print("\n" + "=" * 80)
    print("PHASE B: EXTENDED GRID SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'nw':>4} {'npc':>4} {'lc':>5}  "
          f"{'Ind Acc':>10} {'Fed Acc':>10} {'Best Fed':>10}")
    print("-" * 55)

    ranked = sorted(
        config_results,
        key=lambda c: c.get("summary", {}).get("mean_best_fed_acc", 0),
        reverse=True,
    )

    for cfg in ranked:
        p = cfg["params"]
        s = cfg["summary"]
        print(
            f"{p.get('num_weights', '?'):>4} "
            f"{p.get('neurons_per_class', '?'):>4} "
            f"{p.get('learning_competition', '?'):>5.1f}  "
            f"{s['mean_individual_acc']*100:>6.1f}%+/-{s['std_individual_acc']*100:>4.1f}  "
            f"{s['mean_federated_acc']*100:>6.1f}%+/-{s['std_federated_acc']*100:>4.1f}  "
            f"{s['mean_best_fed_acc']*100:>6.1f}%+/-{s['std_best_fed_acc']*100:>4.1f}"
        )
    print("=" * 80)


def _print_overall_summary(all_results: dict):
    """Print a summary of all phases."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SWEEP SUMMARY")
    print("=" * 80)

    for phase_key, phase_name in [
        ("phase_a", "A: Fine-tuning"),
        ("phase_b", "B: Extended grid"),
        ("phase_c", "C: Binarization comparison"),
        ("phase_d", "D: Disjoint extractor"),
        ("phase_e", "E: Wide features"),
        ("phase_f", "F: Multi-round federation"),
        ("phase_g", "G: Entropy + wide-256"),
        ("phase_h", "H: N-node simulation"),
    ]:
        phase_data = all_results.get(phase_key, {})
        if not phase_data:
            print(f"  Phase {phase_name}: NOT RUN")
            continue

        if phase_key == "phase_a":
            statuses = phase_data if isinstance(phase_data, dict) else {}
            ok_count = sum(1 for v in statuses.values()
                          if isinstance(v, dict) and v.get("status") == "ok")
            skip_count = sum(1 for v in statuses.values()
                            if isinstance(v, dict) and v.get("status") == "skipped")
            print(f"  Phase {phase_name}: {ok_count} completed, {skip_count} skipped")
        else:
            configs = phase_data.get("configs", [])
            if configs:
                best = _get_top_configs(configs, 1)
                if best:
                    b = best[0]
                    s = b.get("summary", {})
                    print(f"  Phase {phase_name}: {len(configs)} configs, "
                          f"best fed acc = {s.get('mean_best_fed_acc', 0)*100:.1f}% "
                          f"({b['params']})")
                else:
                    print(f"  Phase {phase_name}: {len(configs)} configs")
            else:
                print(f"  Phase {phase_name}: 0 configs")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Main sweep orchestration
# ---------------------------------------------------------------------------

def run_comprehensive_sweep(
    phases: list[str],
    num_trials: int = DEFAULT_NUM_TRIALS,
    finetune_epochs: int = DEFAULT_FINETUNE_EPOCHS,
    grid_subset: int | None = None,
    output_dir: Path | None = None,
    skip_finetune: bool = False,
):
    """Run the comprehensive experiment sweep.

    Args:
        phases: List of phase letters to run (A, B, C, D, E, F) or ["ALL"].
        num_trials: Number of trials per config.
        finetune_epochs: Epochs for Phase A fine-tuning.
        grid_subset: Limit number of configs in Phase B (for testing).
        output_dir: Output directory for results.
        skip_finetune: Skip Phase A even if included in phases.
    """
    config_module = setup_pi_orchestrator()

    from federated_experiment.orchestrator import (
        launch_workers,
        shutdown_workers,
    )

    if output_dir is None:
        output_dir = Path.home() / "federated_experiment" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path.home() / "federated_experiment" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "comprehensive_sweep.json"

    # Resolve phase list
    run_all = "ALL" in [p.upper() for p in phases]
    phase_set = set(p.upper() for p in phases)

    # Load existing results for resume
    all_results = {}
    if results_path.exists():
        try:
            with open(results_path) as f:
                all_results = json.load(f)
            log.info("Loaded existing results from %s", results_path)
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("Could not load existing results: %s", e)
            all_results = {}

    all_results["experiment_config"] = {
        "num_trials": num_trials,
        "finetune_epochs": finetune_epochs,
        "phases_requested": phases,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    sweep_start = time.time()

    # --- Phase A: Fine-tuning (no workers needed) ---
    if (run_all or "A" in phase_set) and not skip_finetune:
        try:
            phase_a_results = run_phase_a(finetune_epochs, output_dir)
            all_results["phase_a"] = phase_a_results
            _save_results(results_path, all_results)
        except Exception as e:
            log.error("Phase A failed: %s", e, exc_info=True)
            all_results["phase_a"] = {"error": str(e)}
            _save_results(results_path, all_results)

    # --- Phases B-F require workers ---
    need_workers = any(
        (run_all or p in phase_set)
        for p in ("B", "C", "D", "E", "F", "G", "H")
    )

    if need_workers:
        log.info("Launching workers for phases B-F ...")
        launch_workers()

        try:
            # --- Phase B: Extended grid ---
            if run_all or "B" in phase_set:
                try:
                    run_phase_b(
                        config_module=config_module,
                        num_trials=num_trials,
                        data_dir=data_dir,
                        results_dir=output_dir,
                        results_path=results_path,
                        all_results=all_results,
                        grid_subset=grid_subset,
                    )
                except Exception as e:
                    log.error("Phase B failed: %s", e, exc_info=True)
                    all_results.setdefault("phase_b", {})["error"] = str(e)
                    _save_results(results_path, all_results)

            # --- Phase C: Binarization comparison ---
            if run_all or "C" in phase_set:
                try:
                    run_phase_c(
                        config_module=config_module,
                        num_trials=num_trials,
                        data_dir=data_dir,
                        results_dir=output_dir,
                        results_path=results_path,
                        all_results=all_results,
                    )
                except Exception as e:
                    log.error("Phase C failed: %s", e, exc_info=True)
                    all_results.setdefault("phase_c", {})["error"] = str(e)
                    _save_results(results_path, all_results)

            # --- Phase D: Disjoint extractor ---
            if run_all or "D" in phase_set:
                try:
                    run_phase_d(
                        config_module=config_module,
                        num_trials=num_trials,
                        data_dir=data_dir,
                        results_dir=output_dir,
                        results_path=results_path,
                        all_results=all_results,
                    )
                except Exception as e:
                    log.error("Phase D failed: %s", e, exc_info=True)
                    all_results.setdefault("phase_d", {})["error"] = str(e)
                    _save_results(results_path, all_results)

            # --- Phase E: Wide features ---
            if run_all or "E" in phase_set:
                try:
                    run_phase_e(
                        config_module=config_module,
                        num_trials=num_trials,
                        data_dir=data_dir,
                        results_dir=output_dir,
                        results_path=results_path,
                        all_results=all_results,
                    )
                except Exception as e:
                    log.error("Phase E failed: %s", e, exc_info=True)
                    all_results.setdefault("phase_e", {})["error"] = str(e)
                    _save_results(results_path, all_results)

            # --- Phase F: Multi-round federation ---
            if run_all or "F" in phase_set:
                try:
                    run_phase_f(
                        config_module=config_module,
                        num_trials=num_trials,
                        data_dir=data_dir,
                        results_dir=output_dir,
                        results_path=results_path,
                        all_results=all_results,
                    )
                except Exception as e:
                    log.error("Phase F failed: %s", e, exc_info=True)
                    all_results.setdefault("phase_f", {})["error"] = str(e)
                    _save_results(results_path, all_results)

            # --- Phase G: Entropy + wide-256 combined ---
            if run_all or "G" in phase_set:
                try:
                    run_phase_g(
                        config_module=config_module,
                        num_trials=num_trials,
                        data_dir=data_dir,
                        results_dir=output_dir,
                        results_path=results_path,
                        all_results=all_results,
                    )
                except Exception as e:
                    log.error("Phase G failed: %s", e, exc_info=True)
                    all_results.setdefault("phase_g", {})["error"] = str(e)
                    _save_results(results_path, all_results)

            # --- Phase H: N-node simulation ---
            if run_all or "H" in phase_set:
                try:
                    run_phase_h(
                        config_module=config_module,
                        num_trials=num_trials,
                        data_dir=data_dir,
                        results_dir=output_dir,
                        results_path=results_path,
                        all_results=all_results,
                    )
                except Exception as e:
                    log.error("Phase H failed: %s", e, exc_info=True)
                    all_results.setdefault("phase_h", {})["error"] = str(e)
                    _save_results(results_path, all_results)

        finally:
            shutdown_workers()

    # Final save and summary
    elapsed = time.time() - sweep_start
    all_results["total_elapsed_seconds"] = elapsed
    all_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _save_results(results_path, all_results)

    log.info("Comprehensive sweep complete in %.1f seconds (%.1f hours)",
             elapsed, elapsed / 3600)
    log.info("Results saved to %s", results_path)

    # Print summaries
    phase_b_configs = all_results.get("phase_b", {}).get("configs", [])
    if phase_b_configs:
        _print_phase_b_summary(phase_b_configs)

    _print_overall_summary(all_results)

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive federated neuromorphic experiment sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases:
  A  Fine-tune feature extractors (A1-A4)
  B  Extended hyperparameter grid (42 configs x trials)
  C  Binarization comparison (top 5 x 3 methods x trials)
  D  Disjoint feature extractor (top 5 x trials)
  E  Wide feature dimensions (128 + 256 dim)
  F  Multi-round federation (top 3 x 2 strategies x trials x 5 rounds)
  G  Entropy + wide-256 combined (6 configs x trials)
  H  N-node simulation (top 3 x trials, 4 virtual nodes via 2 physical)

Trial extension:
  Running with --num-trials 30 on a phase that already has 10 trials
  will extend those configs by running the additional 20 trials.

Examples:
  python comprehensive_sweep.py --phase ALL
  python comprehensive_sweep.py --phase H
  python comprehensive_sweep.py --phase C --phase E --num-trials 30
  python comprehensive_sweep.py --phase B --num-trials 1 --grid-subset 2
  python comprehensive_sweep.py --phase C --phase D
  python comprehensive_sweep.py --phase B --skip-finetune
        """,
    )
    parser.add_argument(
        "--phase", action="append", default=None,
        help="Phase(s) to run: A, B, C, D, E, F, or ALL (default: ALL). "
             "Can be specified multiple times.",
    )
    parser.add_argument(
        "--num-trials", type=int, default=DEFAULT_NUM_TRIALS,
        help=f"Number of trials per config (default: {DEFAULT_NUM_TRIALS})",
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=DEFAULT_FINETUNE_EPOCHS,
        help=f"Fine-tuning epochs for Phase A (default: {DEFAULT_FINETUNE_EPOCHS})",
    )
    parser.add_argument(
        "--grid-subset", type=int, default=None,
        help="Only run first N grid configs in Phase B (for testing)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: ~/federated_experiment/results/)",
    )
    parser.add_argument(
        "--skip-finetune", action="store_true",
        help="Skip Phase A fine-tuning even if included",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [sweep] %(levelname)s %(message)s",
    )

    phases = args.phase if args.phase else ["ALL"]

    run_comprehensive_sweep(
        phases=phases,
        num_trials=args.num_trials,
        finetune_epochs=args.finetune_epochs,
        grid_subset=args.grid_subset,
        output_dir=args.output_dir,
        skip_finetune=args.skip_finetune,
    )


if __name__ == "__main__":
    main()
