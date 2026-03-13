#!/usr/bin/env python3
"""Hyperparameter sweep over STDP edge learning parameters.

Runs on claudio as Pi orchestrator, controlling both Pis over direct Ethernet.
Grid search: num_weights × neurons_per_class × learning_competition.
Each config runs multiple trials with different seeds.

Usage (on claudio):
    source ~/akida-env/bin/activate
    cd ~/federated_experiment
    python hyperparam_sweep.py

Quick test:
    python hyperparam_sweep.py --num-trials 1 --grid-subset 1
"""

import argparse
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------
GRID = {
    "num_weights": [10, 15, 20, 25],
    "neurons_per_class": [25, 50, 75],
    "learning_competition": [0.1, 1.0],
}

DEFAULT_NUM_TRIALS = 10
DEFAULT_SEEDS = [42 + i for i in range(DEFAULT_NUM_TRIALS)]
MULTI_ROUND_ROUNDS = 5
TOP_N_FOR_MULTI_ROUND = 3


def setup_pi_orchestrator():
    """Configure the orchestrator to run on claudio in Pi-to-Pi mode."""
    sys.path.insert(0, str(Path.home()))
    from federated_experiment import config
    config.PI_ORCHESTRATOR_NODE = "claudio"
    config.MAC_PROJECT_DIR = config.PI_WORK_DIR
    return config


def run_single_trial(
    seed: int,
    params: dict,
    config,
    data_dir: Path,
    results_dir: Path,
    skip_deploy: bool = False,
) -> dict:
    """Run one trial with the given STDP hyperparameters.

    Returns dict with individual and federated accuracies.
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

    # Prepare data with this seed
    prepare_data(data_dir, seed=seed, use_calibration=False)

    # Deploy data to nodes
    if not skip_deploy:
        deploy_to_nodes(data_dir)

    # Build edge models with sweep hyperparameters
    for name in ("claudio", "paolo"):
        resp = send_command(config.get_command_ip(name), {
            "action": "build_model",
            "params": params,
        })
        if resp.get("status") != "ok":
            log.error("build_model failed on %s: %s", name, resp)
            return {"error": f"build_model failed on {name}"}

    # Train
    training_metrics = run_local_training()

    # Individual evaluation
    individual_results = run_individual_evaluation()

    # Weight exchange + federated evaluation
    exchange_data = run_weight_exchange()
    federated_results = run_federated_evaluation(exchange_data)

    trial_time = time.time() - trial_start

    # Extract key metrics
    result = {
        "seed": seed,
        "params": params,
        "trial_time_seconds": trial_time,
        "individual": individual_results,
        "federated": {},
    }

    # Summarize federated results
    for key, metrics in federated_results.items():
        if isinstance(metrics, dict) and "accuracy" in metrics:
            result["federated"][key] = {
                "accuracy": metrics["accuracy"],
                "correct": metrics.get("correct"),
                "total": metrics.get("total"),
            }

    # Compute mean accuracies across nodes
    ind_accs = [
        individual_results.get(n, {}).get("accuracy", 0)
        for n in ("claudio", "paolo")
    ]
    result["mean_individual_accuracy"] = float(np.mean(ind_accs))

    fed_accs = []
    for strategy in ("fedavg", "fedunion", "fedbest", "fedmajority", "fedselective"):
        for node in ("claudio", "paolo"):
            key = f"{strategy}_{node}"
            acc = federated_results.get(key, {}).get("accuracy", 0)
            if acc > 0:
                fed_accs.append(acc)
    result["mean_federated_accuracy"] = float(np.mean(fed_accs)) if fed_accs else 0.0

    # Best federated accuracy (across all strategies)
    best_fed = 0.0
    best_strategy = ""
    for strategy in ("fedavg", "fedunion", "fedbest", "fedmajority", "fedselective"):
        for node in ("claudio", "paolo"):
            key = f"{strategy}_{node}"
            acc = federated_results.get(key, {}).get("accuracy", 0)
            if acc > best_fed:
                best_fed = acc
                best_strategy = strategy
    result["best_federated_accuracy"] = best_fed
    result["best_strategy"] = best_strategy

    return result


def run_sweep(
    num_trials: int = DEFAULT_NUM_TRIALS,
    grid_subset: int | None = None,
    resume_path: Path | None = None,
    output_dir: Path | None = None,
    skip_multi_round: bool = False,
):
    """Run the full hyperparameter sweep."""
    config = setup_pi_orchestrator()
    from federated_experiment.orchestrator import (
        launch_workers,
        shutdown_workers,
        send_command,
        run_local_training,
        run_individual_evaluation,
        run_weight_exchange,
        run_federated_evaluation,
        run_multi_round_federation,
        prepare_data,
        deploy_to_nodes,
    )

    if output_dir is None:
        output_dir = config.PI_WORK_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = config.PI_WORK_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    seeds = DEFAULT_SEEDS[:num_trials]
    results_path = output_dir / "sweep_results.json"

    # Generate grid configurations
    keys = sorted(GRID.keys())
    combos = list(itertools.product(*(GRID[k] for k in keys)))
    configs = [dict(zip(keys, combo)) for combo in combos]

    if grid_subset is not None:
        configs = configs[:grid_subset]

    log.info("Sweep: %d configs x %d trials = %d total runs",
             len(configs), num_trials, len(configs) * num_trials)

    # Resume support
    completed_configs = {}
    if resume_path and resume_path.exists():
        with open(resume_path) as f:
            prev_results = json.load(f)
        for cfg_result in prev_results.get("configs", []):
            cfg_key = _config_key(cfg_result["params"])
            completed_configs[cfg_key] = cfg_result
        log.info("Resuming: %d configs already completed", len(completed_configs))
    elif results_path.exists():
        # Auto-resume from output path
        try:
            with open(results_path) as f:
                prev_results = json.load(f)
            for cfg_result in prev_results.get("configs", []):
                cfg_key = _config_key(cfg_result["params"])
                if len(cfg_result.get("trials", [])) >= num_trials:
                    completed_configs[cfg_key] = cfg_result
            log.info("Auto-resume: %d configs already completed", len(completed_configs))
        except (json.JSONDecodeError, KeyError):
            pass

    # Launch workers once
    launch_workers()

    all_config_results = list(completed_configs.values())
    sweep_start = time.time()

    try:
        for cfg_idx, params in enumerate(configs):
            cfg_key = _config_key(params)
            if cfg_key in completed_configs:
                log.info("[%d/%d] Skipping (already done): %s",
                         cfg_idx + 1, len(configs), params)
                continue

            log.info("=" * 60)
            log.info("[%d/%d] Config: %s", cfg_idx + 1, len(configs), params)
            log.info("=" * 60)

            trials = []
            for trial_idx, seed in enumerate(seeds):
                log.info("  Trial %d/%d (seed=%d)", trial_idx + 1, num_trials, seed)
                trial_result = run_single_trial(
                    seed=seed,
                    params=params,
                    config=config,
                    data_dir=data_dir,
                    results_dir=output_dir,
                )
                trials.append(trial_result)

            # Aggregate config results
            ind_accs = [t["mean_individual_accuracy"] for t in trials if "error" not in t]
            fed_accs = [t["mean_federated_accuracy"] for t in trials if "error" not in t]
            best_accs = [t["best_federated_accuracy"] for t in trials if "error" not in t]

            config_result = {
                "params": params,
                "num_trials": len(trials),
                "trials": trials,
                "summary": {
                    "mean_individual_acc": float(np.mean(ind_accs)) if ind_accs else 0,
                    "std_individual_acc": float(np.std(ind_accs)) if ind_accs else 0,
                    "mean_federated_acc": float(np.mean(fed_accs)) if fed_accs else 0,
                    "std_federated_acc": float(np.std(fed_accs)) if fed_accs else 0,
                    "mean_best_fed_acc": float(np.mean(best_accs)) if best_accs else 0,
                    "std_best_fed_acc": float(np.std(best_accs)) if best_accs else 0,
                },
            }
            all_config_results.append(config_result)

            # Incremental save
            _save_results(results_path, all_config_results, sweep_start)
            log.info("  Config summary: ind=%.1f%% +/- %.1f%%, fed=%.1f%% +/- %.1f%%",
                     config_result["summary"]["mean_individual_acc"] * 100,
                     config_result["summary"]["std_individual_acc"] * 100,
                     config_result["summary"]["mean_federated_acc"] * 100,
                     config_result["summary"]["std_federated_acc"] * 100)

        # --- Phase 2: Multi-round runs for top configs ---
        if not skip_multi_round and len(all_config_results) >= TOP_N_FOR_MULTI_ROUND:
            log.info("=" * 60)
            log.info("PHASE 2: Multi-round federation for top %d configs", TOP_N_FOR_MULTI_ROUND)
            log.info("=" * 60)

            # Sort by best federated accuracy
            ranked = sorted(
                all_config_results,
                key=lambda c: c["summary"].get("mean_best_fed_acc", 0),
                reverse=True,
            )
            top_configs = ranked[:TOP_N_FOR_MULTI_ROUND]

            for rank_idx, cfg_result in enumerate(top_configs):
                params = cfg_result["params"]
                log.info("[Top %d] Config: %s (acc=%.1f%%)",
                         rank_idx + 1, params,
                         cfg_result["summary"]["mean_best_fed_acc"] * 100)

                multi_round_trials = []
                for trial_idx, seed in enumerate(seeds):
                    log.info("  Multi-round trial %d/%d (seed=%d)",
                             trial_idx + 1, num_trials, seed)

                    # Prepare data
                    prepare_data(data_dir, seed=seed, use_calibration=False)
                    deploy_to_nodes(data_dir)

                    # Build with these params
                    for name in ("claudio", "paolo"):
                        send_command(config.get_command_ip(name), {
                            "action": "build_model",
                            "params": params,
                        })

                    # Train + evaluate
                    run_local_training()
                    individual_results = run_individual_evaluation()
                    exchange_data = run_weight_exchange()

                    # Multi-round federation
                    round_results = run_multi_round_federation(
                        MULTI_ROUND_ROUNDS, exchange_data)

                    multi_round_trials.append({
                        "seed": seed,
                        "individual": individual_results,
                        "rounds": round_results,
                    })

                cfg_result["multi_round"] = {
                    "num_rounds": MULTI_ROUND_ROUNDS,
                    "trials": multi_round_trials,
                }

                _save_results(results_path, all_config_results, sweep_start)

        # Final save
        _save_results(results_path, all_config_results, sweep_start)
        log.info("Sweep complete! Results at %s", results_path)

        # Print summary table
        _print_summary(all_config_results)

    finally:
        shutdown_workers()


def _config_key(params: dict) -> str:
    """Create a hashable key for a config dict."""
    return json.dumps(params, sort_keys=True)


def _save_results(path: Path, config_results: list, sweep_start: float):
    """Save incremental results to disk."""
    output = {
        "sweep_grid": GRID,
        "elapsed_seconds": time.time() - sweep_start,
        "num_configs_completed": len(config_results),
        "configs": config_results,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)


def _print_summary(config_results: list):
    """Print a summary table of all sweep results."""
    print("\n" + "=" * 80)
    print("HYPERPARAMETER SWEEP SUMMARY")
    print("=" * 80)
    print(f"{'nw':>4} {'npc':>4} {'lc':>5}  "
          f"{'Ind Acc':>10} {'Fed Acc':>10} {'Best Fed':>10}")
    print("-" * 55)

    ranked = sorted(
        config_results,
        key=lambda c: c["summary"].get("mean_best_fed_acc", 0),
        reverse=True,
    )

    for cfg in ranked:
        p = cfg["params"]
        s = cfg["summary"]
        print(f"{p['num_weights']:>4} {p['neurons_per_class']:>4} {p['learning_competition']:>5.1f}  "
              f"{s['mean_individual_acc']*100:>6.1f}%+/-{s['std_individual_acc']*100:>4.1f}  "
              f"{s['mean_federated_acc']*100:>6.1f}%+/-{s['std_federated_acc']*100:>4.1f}  "
              f"{s['mean_best_fed_acc']*100:>6.1f}%+/-{s['std_best_fed_acc']*100:>4.1f}")

    print("=" * 80)

    if ranked and "multi_round" in ranked[0]:
        print("\nTop configs with multi-round federation:")
        for cfg in ranked[:TOP_N_FOR_MULTI_ROUND]:
            if "multi_round" not in cfg:
                continue
            p = cfg["params"]
            mr = cfg["multi_round"]
            print(f"  nw={p['num_weights']}, npc={p['neurons_per_class']}, "
                  f"lc={p['learning_competition']}: "
                  f"{mr['num_rounds']} rounds, {len(mr['trials'])} trials")


def main():
    parser = argparse.ArgumentParser(description="STDP hyperparameter sweep")
    parser.add_argument("--num-trials", type=int, default=DEFAULT_NUM_TRIALS,
                        help=f"Trials per config (default: {DEFAULT_NUM_TRIALS})")
    parser.add_argument("--grid-subset", type=int, default=None,
                        help="Only run first N grid configs (for testing)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume from a previous sweep_results.json")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: ~/federated_experiment/results/)")
    parser.add_argument("--skip-multi-round", action="store_true",
                        help="Skip phase 2 (multi-round runs for top configs)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [sweep] %(levelname)s %(message)s",
    )

    run_sweep(
        num_trials=args.num_trials,
        grid_subset=args.grid_subset,
        resume_path=args.resume,
        output_dir=args.output_dir,
        skip_multi_round=args.skip_multi_round,
    )


if __name__ == "__main__":
    main()
