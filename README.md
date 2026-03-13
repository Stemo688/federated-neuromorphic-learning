# Federated Few-Shot Learning on Neuromorphic Hardware

Code for the paper: *Federated Few-Shot Learning on Neuromorphic Hardware: An Empirical Study Across Physical Edge Nodes* (Motta & Nanni, 2026).

This repository contains the full experimental framework for federated few-shot class learning using BrainChip Akida AKD1000 neuromorphic processors with on-chip STDP. Two physical Raspberry Pi 5 nodes, each equipped with an Akida PCIe accelerator, learn novel keyword classes from limited samples and exchange neuron-level weight maps over a direct Ethernet link.

## Hardware Requirements

- 2x Raspberry Pi 5 (8 GB recommended)
- 2x BrainChip Akida AKD1000 (PCIe HAT)
- Direct Ethernet link between Pis (10.0.0.x subnet)
- Orchestrator machine (Mac or Linux) with SSH access to both Pis

## Software Requirements

On each Raspberry Pi:
- Python 3.12 with virtualenv at `~/akida-env/`
- `akida`, `akida-models`, `cnn2snn` (BrainChip SDK)
- `numpy`, `scipy`, `scikit-learn`

On the orchestrator:
- Python 3.10+
- `numpy`, `scipy`, `scikit-learn`, `paramiko`

## Repository Structure

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters, network topology, and experimental grid definitions |
| `run_experiment.py` | Main entry point for running experiments |
| `orchestrator.py` | Coordinates multi-node experiments from the orchestrator machine |
| `node_worker.py` | Worker process running on each Raspberry Pi |
| `federation.py` | Federation strategies: FedUnion, FedAvg, FedBest, FedMajority, FedSelective |
| `finetune_dscnn.py` | Feature extractor fine-tuning (DS-CNN backbone) with wide feature support |
| `data_loader.py` | Google Speech Commands v0.02 dataset loading and MFCC extraction |
| `baselines.py` | Software baselines (k-NN, linear, MLP) for comparison |
| `baselines_pi.py` | On-device software baselines running on Raspberry Pi |
| `comprehensive_sweep.py` | Full Phase A-F experimental sweep (binarization, disjoint, wide features, multi-round) |
| `hyperparam_sweep.py` | STDP hyperparameter grid search |
| `analyze_results.py` | Statistical analysis, bootstrap CIs, Cohen's d, result tables |
| `setup_claudio.sh` | Pi setup script |
| `run_autonomous.sh` | Autonomous experiment runner |
| `run_autonomous_v2.sh` | Updated autonomous runner with comprehensive sweep |

## Quick Start

### 1. Prepare Data

Download Google Speech Commands v0.02 and extract MFCC features:

```bash
python -m run_experiment --data-only
```

### 2. Deploy to Pis

Ensure SSH keys are configured for both Pis, then:

```bash
python -m run_experiment --num-trials 1 --num-rounds 1
```

This deploys code and data to both Pis, runs a single trial, and collects results.

### 3. Full Experiment

To reproduce the paper results (10 trials, 5 federation rounds, all baselines):

```bash
python -m run_experiment \
    --num-trials 10 \
    --num-rounds 5 \
    --shared-thresholds \
    --run-baselines
```

### 4. Comprehensive Sweep (Phases A-F)

To run the full experimental sweep including binarization comparison, disjoint validation, wide features, and multi-round analysis:

```bash
python comprehensive_sweep.py
```

### 5. Analysis

Generate statistical analysis from collected results:

```bash
python analyze_results.py
```

## Federation Strategies

- **FedUnion**: Concatenate neuron populations from all nodes (neuron-level aggregation)
- **FedAvg**: Average weight vectors across matching neurons (weight-level aggregation)
- **FedBest**: Select the single best-performing node's model
- **FedMajority**: Ensemble majority vote across node predictions
- **FedSelective**: Selective neuron merging based on activation thresholds

## Experimental Phases

The comprehensive sweep (`comprehensive_sweep.py`) runs six phases:

| Phase | Description | Runs |
|-------|-------------|------|
| A | Baseline (pre-fine-tuning) | -- |
| B | 42-config hyperparameter sweep (7 nw x 3 npc x 2 lc) | 420 |
| C | Binarization comparison (mean/median/entropy) on top configs | 150 |
| D | Disjoint class validation (extractor never sees target classes) | 50 |
| E | Wide feature scaling (128-dim and 256-dim) | 130 |
| F | Multi-round federation stability analysis | 60 |

Total: approximately 800 experimental runs on physical hardware.

## Citation

If you use this code, please cite:

```bibtex
@article{motta2026federated,
  title={Federated Few-Shot Learning on Neuromorphic Hardware: An Empirical Study Across Physical Edge Nodes},
  author={Motta, Steven and Nanni, Gioele},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
