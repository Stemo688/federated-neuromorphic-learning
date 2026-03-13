#!/usr/bin/env python3
"""Main entry point for the federated neuromorphic few-shot learning experiment.

Run from the Mac:
    cd /Users/steven/Documents/GitHub/neuromorphic_bridge
    python -m federated_experiment.run_experiment

Or directly:
    python federated_experiment/run_experiment.py

Multi-trial experiment with all improvements:
    python -m federated_experiment.run_experiment \
        --num-trials 10 --num-rounds 5 \
        --shared-thresholds --run-baselines

Analysis only:
    python -m federated_experiment.analyze_results
"""

import argparse
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Federated Neuromorphic Few-Shot Learning Experiment",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Local directory for dataset (default: federated_experiment/data/)",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Skip deploying code/data to Pis (if already deployed)",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only download and prepare data, don't run experiment",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials with different random seeds (default: 1)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
        help="Number of federation rounds with retraining (default: 1)",
    )
    parser.add_argument(
        "--shared-thresholds",
        action="store_true",
        help="Use shared binarization thresholds from calibration set",
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Run software baselines (linear + KNN) for comparison",
    )
    parser.add_argument(
        "--class-set",
        type=str,
        default=None,
        choices=["default", "movement", "digits", "pretrained"],
        help="Which set of novel classes to use (default: backward/follow/forward)",
    )
    parser.add_argument(
        "--compare-retraining",
        action="store_true",
        help="Run a second multi-round experiment with FedUnion-based retraining",
    )
    parser.add_argument(
        "--comparison-mode",
        action="store_true",
        help="Run both local and shared threshold regimes for comparison",
    )
    parser.add_argument(
        "--pi-orchestrator",
        choices=["claudio", "paolo"],
        default=None,
        help="Run orchestrator on a Pi instead of Mac (uses direct Ethernet for Pi-to-Pi)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    # Allow running as script or module
    try:
        from federated_experiment.orchestrator import (
            run_full_experiment, run_comparison_experiment, prepare_data,
        )
        from federated_experiment import config
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from federated_experiment.orchestrator import (
            run_full_experiment, run_comparison_experiment, prepare_data,
        )
        from federated_experiment import config

    # Pi orchestrator mode: override paths and set global config
    if args.pi_orchestrator:
        config.PI_ORCHESTRATOR_NODE = args.pi_orchestrator
        config.MAC_PROJECT_DIR = config.PI_WORK_DIR
        logging.info("Pi orchestrator mode: node=%s, project_dir=%s",
                     args.pi_orchestrator, config.MAC_PROJECT_DIR)

    data_dir = args.data_dir or config.MAC_PROJECT_DIR / "data"

    if args.data_only:
        logging.info("Preparing data only ...")
        prepare_data(data_dir)
        logging.info("Data preparation complete.")
        return

    logging.info("Starting federated neuromorphic few-shot learning experiment")
    logging.info("Nodes: %s", list(config.NODES.keys()))
    logging.info("Novel classes: %s", config.NOVEL_CLASSES)
    logging.info("Samples per class: %d", config.SPLIT.samples_per_class)
    logging.info("Federation strategies: %s", config.FEDERATION_STRATEGIES)
    logging.info("Num trials: %d, Num rounds: %d", args.num_trials, args.num_rounds)
    logging.info("Shared thresholds: %s, Baselines: %s",
                 args.shared_thresholds, args.run_baselines)
    if args.class_set:
        logging.info("Class set: %s", args.class_set)

    if args.comparison_mode:
        logging.info("Running comparison experiment (local vs shared thresholds)")
        results = run_comparison_experiment(
            data_dir=data_dir,
            num_trials=args.num_trials,
            num_rounds=args.num_rounds,
            run_software_baselines=args.run_baselines,
            class_set=args.class_set,
            compare_retraining=args.compare_retraining,
        )
    else:
        results = run_full_experiment(
            data_dir,
            skip_data=args.skip_deploy,
            skip_deploy=args.skip_deploy,
            num_trials=args.num_trials,
            num_rounds=args.num_rounds,
            use_shared_thresholds=args.shared_thresholds,
            run_software_baselines=args.run_baselines,
            class_set=args.class_set,
            compare_retraining=args.compare_retraining,
        )

    if results:
        logging.info("Experiment completed successfully!")
    else:
        logging.error("Experiment failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
