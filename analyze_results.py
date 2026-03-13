#!/usr/bin/env python3
"""Analyze multi-trial experiment results: statistics, plots, and LaTeX tables.

Usage:
    python -m federated_experiment.analyze_results [--results PATH] [--output-dir PATH]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(path: Path) -> dict:
    """Load multi-trial results JSON."""
    with open(path) as f:
        return json.load(f)


def extract_accuracies(results: dict) -> dict:
    """Extract accuracy arrays from multi-trial results for easy analysis.

    Returns nested dict:
        {metric_name: np.ndarray of per-trial values}
    """
    trials = results.get("trials", [])
    n = len(trials)
    if n == 0:
        return {}

    accs = {}

    # Individual accuracies
    for node in ("claudio", "paolo"):
        key = f"individual_{node}"
        accs[key] = np.array([
            t["individual"].get(node, {}).get("accuracy", np.nan)
            for t in trials
        ])

    # Per-class individual accuracies
    for node in ("claudio", "paolo"):
        for cls in ("0", "1", "2"):
            key = f"individual_{node}_class{cls}"
            accs[key] = np.array([
                t["individual"].get(node, {}).get("per_class_accuracy", {}).get(cls, np.nan)
                for t in trials
            ])

    # Federated accuracies (all rounds)
    max_rounds = max(
        len(t.get("federated", {})) for t in trials
    ) if trials else 1

    all_strategies = ("fedavg", "fedunion", "fedbest", "fedmajority", "fedselective")

    for round_num in range(1, max_rounds + 1):
        rkey = f"round_{round_num}"
        for strategy in all_strategies:
            for node in ("claudio", "paolo"):
                key = f"{strategy}_{node}_{rkey}"
                accs[key] = np.array([
                    t.get("federated", {}).get(rkey, {}).get(strategy, {}).get(
                        node, {}).get("accuracy", np.nan)
                    for t in trials
                ])
                # Per-class
                for cls in ("0", "1", "2"):
                    ckey = f"{strategy}_{node}_{rkey}_class{cls}"
                    accs[ckey] = np.array([
                        t.get("federated", {}).get(rkey, {}).get(strategy, {}).get(
                            node, {}).get("per_class_accuracy", {}).get(cls, np.nan)
                        for t in trials
                    ])

    # Baselines
    for bl_type in ("linear_individual", "linear_fedavg", "mlp_individual",
                    "mlp_fedavg", "knn_individual", "knn_fedavg"):
        vals = []
        for t in trials:
            bl = t.get("baselines", {}).get(bl_type, {})
            if isinstance(bl, dict) and "accuracy" in bl:
                vals.append(bl["accuracy"])
            elif isinstance(bl, dict):
                # Per-node: average the node accuracies
                node_accs = [v["accuracy"] for v in bl.values()
                             if isinstance(v, dict) and "accuracy" in v]
                vals.append(np.mean(node_accs) if node_accs else np.nan)
            else:
                vals.append(np.nan)
        accs[f"baseline_{bl_type}"] = np.array(vals)

    # FedUnion-retraining multi-round data (if present)
    has_union_retrain = any("federated_union_retrain" in t for t in trials)
    if has_union_retrain:
        max_ur_rounds = max(
            len(t.get("federated_union_retrain", {})) for t in trials
        ) if trials else 0
        for round_num in range(1, max_ur_rounds + 1):
            rkey = f"round_{round_num}"
            for strategy in all_strategies:
                for node in ("claudio", "paolo"):
                    key = f"ur_{strategy}_{node}_{rkey}"
                    accs[key] = np.array([
                        t.get("federated_union_retrain", {}).get(rkey, {}).get(
                            strategy, {}).get(node, {}).get("accuracy", np.nan)
                        for t in trials
                    ])

    return accs


# ---------------------------------------------------------------------------
# Bootstrap CIs and effect sizes
# ---------------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000,
                 ci: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Compute bootstrap confidence interval (percentile method).

    Args:
        data: 1-D array of observations.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    data = data[~np.isnan(data)]
    if len(data) < 2:
        m = np.mean(data) if len(data) == 1 else 0.0
        return m, m
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_bootstrap)
    ])
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d for paired samples.

    Uses the standard deviation of the differences as the denominator,
    appropriate for within-subject (paired) designs.

    Args:
        a, b: Paired 1-D arrays of observations.

    Returns:
        Cohen's d effect size.
    """
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    diff = a[:n] - b[:n]
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def compute_effect_sizes(accs: dict) -> dict:
    """Compute Cohen's d and bootstrap CIs for key comparisons.

    Returns dict of {comparison_name: {cohens_d, ci_a, ci_b}}.
    """
    results = {}

    def _add(name, key_a, key_b):
        a = accs.get(key_a)
        b = accs.get(key_b)
        if a is None or b is None:
            return
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) < 3 or len(b) < 3:
            return

        results[name] = {
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "ci_a": list(bootstrap_ci(a)),
            "ci_b": list(bootstrap_ci(b)),
            "cohens_d": cohens_d(a, b),
        }

    # Individual vs federated strategies
    for strategy in ("fedavg", "fedunion", "fedbest", "fedmajority"):
        for node in ("claudio", "paolo"):
            _add(f"transfer_{node}_{strategy}",
                 f"individual_{node}",
                 f"{strategy}_{node}_round_1")

    # Baselines vs neuromorphic
    for bl_type in ("linear_fedavg", "mlp_fedavg", "knn_fedavg"):
        for node in ("claudio", "paolo"):
            _add(f"{bl_type}_vs_fedavg_{node}",
                 f"baseline_{bl_type}",
                 f"fedavg_{node}_round_1")

    return results


# ---------------------------------------------------------------------------
# Comprehensive analysis for sweep results
# ---------------------------------------------------------------------------

def analyze_comprehensive_sweep(results: dict, output_dir: Path):
    """Analyze comprehensive sweep results with all experiments.

    Expected structure: results = {"phase_B": {"configs": [...]}, "phase_C": {...}, ...}
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = {}

    # Phase B: Main sweep analysis
    phase_b = results.get("phase_B", {})
    if phase_b and "configs" in phase_b:
        configs = phase_b["configs"]
        analysis["phase_B"] = _analyze_sweep_configs(configs, "Main Sweep")

    # Phase C: Binarization comparison
    phase_c = results.get("phase_C", {})
    if phase_c and "configs" in phase_c:
        analysis["phase_C"] = _analyze_binarization(phase_c["configs"])

    # Phase D: Disjoint extractor
    phase_d = results.get("phase_D", {})
    if phase_d and "configs" in phase_d:
        analysis["phase_D"] = _analyze_sweep_configs(
            phase_d["configs"], "Disjoint Extractor")

    # Phase E: Wide features
    for sub in ("phase_E1", "phase_E2"):
        phase_e = results.get(sub, {})
        if phase_e and "configs" in phase_e:
            analysis[sub] = _analyze_sweep_configs(
                phase_e["configs"], f"Wide Features ({sub})")

    # Phase F: Multi-round
    phase_f = results.get("phase_F", {})
    if phase_f:
        analysis["phase_F"] = phase_f  # Pass through, complex structure

    # Save analysis
    with open(output_dir / "comprehensive_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Generate summary table
    _print_comprehensive_summary(analysis)

    return analysis


def _analyze_sweep_configs(configs: list, label: str) -> dict:
    """Analyze a list of config results, returning summary with CIs."""
    ranked = sorted(
        configs,
        key=lambda c: c.get("summary", {}).get("mean_best_fed_acc", 0),
        reverse=True,
    )

    summaries = []
    for cfg in ranked:
        s = cfg.get("summary", {})
        trials = cfg.get("trials", [])

        # Compute bootstrap CIs for best federated accuracy
        best_accs = np.array([
            t.get("best_federated_accuracy", 0) for t in trials
            if "error" not in t
        ])

        entry = {
            "params": cfg["params"],
            "mean_best_fed_acc": s.get("mean_best_fed_acc", 0),
            "std_best_fed_acc": s.get("std_best_fed_acc", 0),
            "mean_individual_acc": s.get("mean_individual_acc", 0),
        }
        if len(best_accs) >= 3:
            entry["ci_best_fed"] = list(bootstrap_ci(best_accs))
        summaries.append(entry)

    return {"label": label, "num_configs": len(configs), "ranked": summaries}


def _analyze_binarization(configs: list) -> dict:
    """Analyze binarization method comparison results."""
    by_method = {}
    for cfg in configs:
        method = cfg.get("params", {}).get("binarization_method", "mean")
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(cfg)

    summary = {}
    for method, cfgs in by_method.items():
        all_best = []
        for cfg in cfgs:
            for t in cfg.get("trials", []):
                if "error" not in t:
                    all_best.append(t.get("best_federated_accuracy", 0))
        arr = np.array(all_best)
        summary[method] = {
            "mean": float(np.mean(arr)) if len(arr) > 0 else 0,
            "std": float(np.std(arr)) if len(arr) > 0 else 0,
            "n": len(arr),
        }
        if len(arr) >= 3:
            summary[method]["ci"] = list(bootstrap_ci(arr))

    return {"by_method": summary}


def _print_comprehensive_summary(analysis: dict):
    """Print a human-readable summary of all experiments."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EXPERIMENT ANALYSIS")
    print("=" * 80)

    for phase_key in ("phase_B", "phase_C", "phase_D", "phase_E1", "phase_E2"):
        phase = analysis.get(phase_key, {})
        if not phase:
            continue

        label = phase.get("label", phase_key)
        print(f"\n--- {label} ({phase.get('num_configs', 0)} configs) ---")

        if phase_key == "phase_C":
            # Binarization
            for method, s in phase.get("by_method", {}).items():
                ci_str = ""
                if "ci" in s:
                    ci_str = f" CI=[{s['ci'][0]*100:.1f}, {s['ci'][1]*100:.1f}]"
                print(f"  {method}: {s['mean']*100:.1f}% +/- {s['std']*100:.1f}%{ci_str}")
        else:
            ranked = phase.get("ranked", [])
            for i, entry in enumerate(ranked[:5]):
                p = entry["params"]
                ci_str = ""
                if "ci_best_fed" in entry:
                    ci = entry["ci_best_fed"]
                    ci_str = f" CI=[{ci[0]*100:.1f}, {ci[1]*100:.1f}]"
                print(f"  #{i+1} nw={p.get('num_weights')}, "
                      f"npc={p.get('neurons_per_class')}, "
                      f"lc={p.get('learning_competition')}: "
                      f"best_fed={entry['mean_best_fed_acc']*100:.1f}%"
                      f"{ci_str}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def compute_statistics(accs: dict) -> dict:
    """Compute mean, std, and paired tests for key comparisons.

    Returns dict of {comparison_name: {mean_a, std_a, mean_b, std_b, p_value, test}}.
    """
    from scipy import stats

    results = {}

    def _add_comparison(name, key_a, key_b, test="wilcoxon"):
        a = accs.get(key_a)
        b = accs.get(key_b)
        if a is None or b is None:
            return
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        n = min(len(a), len(b))
        if n < 3:
            return

        a, b = a[:n], b[:n]

        if test == "wilcoxon":
            try:
                stat, p = stats.wilcoxon(a, b)
            except ValueError:
                # All differences are zero
                stat, p = 0.0, 1.0
        else:
            stat, p = stats.ttest_rel(a, b)

        results[name] = {
            "mean_a": float(np.mean(a)),
            "std_a": float(np.std(a)),
            "mean_b": float(np.mean(b)),
            "std_b": float(np.std(b)),
            "statistic": float(stat),
            "p_value": float(p),
            "test": test,
            "n": n,
        }

    # Knowledge transfer: individual vs each strategy (round 1)
    for strategy in ("fedavg", "fedunion", "fedbest", "fedmajority"):
        for node in ("claudio", "paolo"):
            _add_comparison(
                f"transfer_{node}_{strategy}",
                f"individual_{node}",
                f"{strategy}_{node}_round_1",
            )

    # Baselines vs neuromorphic
    for node in ("claudio", "paolo"):
        _add_comparison(
            f"neuromorphic_vs_linear_{node}",
            f"individual_{node}",
            "baseline_linear_individual",
        )
        _add_comparison(
            f"neuromorphic_vs_mlp_{node}",
            f"individual_{node}",
            "baseline_mlp_individual",
        )
        _add_comparison(
            f"fedavg_vs_linear_fedavg_{node}",
            f"fedavg_{node}_round_1",
            "baseline_linear_fedavg",
        )
        _add_comparison(
            f"fedavg_vs_mlp_fedavg_{node}",
            f"fedavg_{node}_round_1",
            "baseline_mlp_fedavg",
        )

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(results: dict, output_dir: Path):
    """Plot accuracy vs federation round for each strategy.

    If FedUnion-retrain data is present, adds dashed lines for the
    alternative retraining regime.
    """
    import matplotlib.pyplot as plt

    trials = results.get("trials", [])
    if not trials:
        return

    max_rounds = max(len(t.get("federated", {})) for t in trials)
    if max_rounds < 2:
        log.info("Only 1 round, skipping convergence plot")
        return

    has_union_retrain = any("federated_union_retrain" in t for t in trials)
    max_ur_rounds = 0
    if has_union_retrain:
        max_ur_rounds = max(
            len(t.get("federated_union_retrain", {})) for t in trials
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, node in zip(axes, ("claudio", "paolo")):
        rounds = list(range(1, max_rounds + 1))

        for strategy, color in [("fedavg", "C0"), ("fedunion", "C1"), ("fedbest", "C2"),
                                 ("fedmajority", "C3")]:
            means = []
            stds = []
            for r in rounds:
                rkey = f"round_{r}"
                accs = [
                    t.get("federated", {}).get(rkey, {}).get(strategy, {}).get(
                        node, {}).get("accuracy", np.nan)
                    for t in trials
                ]
                accs = [a for a in accs if not np.isnan(a)]
                means.append(np.mean(accs) if accs else 0)
                stds.append(np.std(accs) if accs else 0)

            means = np.array(means)
            stds = np.array(stds)
            ax.plot(rounds, means * 100, "-o", color=color, label=strategy)
            ax.fill_between(rounds, (means - stds) * 100, (means + stds) * 100,
                            alpha=0.2, color=color)

        # FedUnion-retrain overlay (dashed lines)
        if has_union_retrain and max_ur_rounds >= 2:
            ur_rounds = list(range(1, max_ur_rounds + 1))
            for strategy, color in [("fedunion", "C1"), ("fedavg", "C0")]:
                means = []
                stds = []
                for r in ur_rounds:
                    rkey = f"round_{r}"
                    accs = [
                        t.get("federated_union_retrain", {}).get(rkey, {}).get(
                            strategy, {}).get(node, {}).get("accuracy", np.nan)
                        for t in trials
                    ]
                    accs = [a for a in accs if not np.isnan(a)]
                    means.append(np.mean(accs) if accs else 0)
                    stds.append(np.std(accs) if accs else 0)

                means = np.array(means)
                stds = np.array(stds)
                ax.plot(ur_rounds, means * 100, "--s", color=color,
                        label=f"{strategy} (UR)", alpha=0.8)
                ax.fill_between(ur_rounds, (means - stds) * 100,
                                (means + stds) * 100, alpha=0.1, color=color)

        # Add individual baseline
        ind_accs = [t["individual"].get(node, {}).get("accuracy", 0) for t in trials]
        ax.axhline(np.mean(ind_accs) * 100, color="gray", linestyle="--",
                    label="individual")

        ax.set_xlabel("Federation Round")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Node: {node}")
        ax.legend(fontsize=7)
        ax.set_xticks(rounds)

    fig.suptitle("Accuracy vs Federation Round")
    fig.tight_layout()
    fig.savefig(output_dir / "convergence.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved convergence plot")


def plot_comparison_bars(accs: dict, output_dir: Path):
    """Bar chart comparing individual, federated, and baseline accuracies."""
    import matplotlib.pyplot as plt

    categories = []
    means = []
    stds = []
    colors = []

    color_map = {
        "individual": "C0",
        "fedavg": "C1",
        "fedunion": "C2",
        "fedbest": "C3",
        "baseline": "C4",
    }

    for node in ("claudio", "paolo"):
        key = f"individual_{node}"
        if key in accs and len(accs[key]) > 0:
            categories.append(f"Indiv.\n{node}")
            means.append(np.nanmean(accs[key]))
            stds.append(np.nanstd(accs[key]))
            colors.append(color_map["individual"])

    color_map["fedmajority"] = "C5"

    for strategy in ("fedavg", "fedunion", "fedbest", "fedmajority"):
        for node in ("claudio", "paolo"):
            key = f"{strategy}_{node}_round_1"
            if key in accs and len(accs[key]) > 0:
                categories.append(f"{strategy}\n{node}")
                means.append(np.nanmean(accs[key]))
                stds.append(np.nanstd(accs[key]))
                colors.append(color_map.get(strategy, "C7"))

    for bl_type in ("linear_fedavg", "mlp_fedavg", "knn_fedavg"):
        key = f"baseline_{bl_type}"
        if key in accs and not np.all(np.isnan(accs[key])):
            categories.append(bl_type.replace("_", "\n"))
            means.append(np.nanmean(accs[key]))
            stds.append(np.nanstd(accs[key]))
            colors.append(color_map["baseline"])

    if not categories:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.8), 5))
    x = np.arange(len(categories))
    ax.bar(x, np.array(means) * 100, yerr=np.array(stds) * 100,
           color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Method Comparison (mean +/- std across trials)")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved comparison bar chart")


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_table(accs: dict, stats: dict) -> str:
    """Generate a LaTeX results table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Federated neuromorphic few-shot learning results "
                 r"(mean $\pm$ std over $N$ trials).}")
    lines.append(r"\label{tab:results}")
    lines.append(r"\begin{tabular}{llcc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Node & Accuracy (\%) & $p$-value \\")
    lines.append(r"\midrule")

    # Individual
    for node in ("claudio", "paolo"):
        key = f"individual_{node}"
        if key in accs:
            m = np.nanmean(accs[key]) * 100
            s = np.nanstd(accs[key]) * 100
            lines.append(f"Individual & {node} & ${m:.1f} \\pm {s:.1f}$ & -- \\\\")

    lines.append(r"\midrule")

    # Federated
    for strategy in ("fedavg", "fedunion", "fedbest", "fedmajority"):
        for node in ("claudio", "paolo"):
            key = f"{strategy}_{node}_round_1"
            if key in accs:
                m = np.nanmean(accs[key]) * 100
                s = np.nanstd(accs[key]) * 100
                stat_key = f"transfer_{node}_{strategy}"
                p_str = "--"
                if stat_key in stats:
                    p = stats[stat_key]["p_value"]
                    if p < 0.001:
                        p_str = "$<0.001$"
                    elif p < 0.05:
                        p_str = f"${p:.3f}$"
                    else:
                        p_str = f"${p:.3f}$"
                sname = strategy.replace("fed", "Fed")
                lines.append(f"{sname} & {node} & ${m:.1f} \\pm {s:.1f}$ & {p_str} \\\\")

    # Baselines
    has_baselines = any(k.startswith("baseline_") for k in accs
                        if not np.all(np.isnan(accs[k])))
    if has_baselines:
        lines.append(r"\midrule")
        for bl_type, label in [("linear_individual", "Linear (indiv.)"),
                                ("linear_fedavg", "Linear (FedAvg)"),
                                ("mlp_individual", "MLP (indiv.)"),
                                ("mlp_fedavg", "MLP (FedAvg)"),
                                ("knn_individual", "KNN (indiv.)"),
                                ("knn_fedavg", r"KNN (pooled$^\dagger$)")]:
            key = f"baseline_{bl_type}"
            if key in accs and not np.all(np.isnan(accs[key])):
                m = np.nanmean(accs[key]) * 100
                s = np.nanstd(accs[key]) * 100
                lines.append(f"{label} & -- & ${m:.1f} \\pm {s:.1f}$ & -- \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_per_class_table(accs: dict) -> str:
    """Generate a LaTeX per-class accuracy table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-class accuracy breakdown (mean $\pm$ std).}")
    lines.append(r"\label{tab:per_class}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & Node & Class 0 (\%) & Class 1 (\%) & Class 2 (\%) \\")
    lines.append(r"\midrule")

    methods = [("individual", "Individual"), ("fedavg", "FedAvg"),
               ("fedunion", "FedUnion"), ("fedmajority", "FedMajority")]
    for i, (method_prefix, label) in enumerate(methods):
        for node in ("claudio", "paolo"):
            cells = []
            for cls in ("0", "1", "2"):
                if method_prefix == "individual":
                    key = f"individual_{node}_class{cls}"
                else:
                    key = f"{method_prefix}_{node}_round_1_class{cls}"
                if key in accs:
                    m = np.nanmean(accs[key]) * 100
                    s = np.nanstd(accs[key]) * 100
                    cells.append(f"${m:.1f} \\pm {s:.1f}$")
                else:
                    cells.append("--")
            lines.append(f"{label} & {node} & {' & '.join(cells)} \\\\")
        if i < len(methods) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_pretrained_per_class_table(accs: dict,
                                        class_names: list[str] | None = None) -> str:
    """Generate a LaTeX per-class table for pretrained class set results.

    Args:
        accs: Accuracy dict from extract_accuracies() on pretrained results.
        class_names: Display names for the 3 classes (e.g. ["yes", "no", "stop"]).
    """
    if class_names is None:
        class_names = ["Class 0", "Class 1", "Class 2"]
    # Escape underscores for LaTeX
    safe_names = [n.replace("_", r"\_") for n in class_names]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-class accuracy for pretrained-vocabulary classes "
                 r"(mean $\pm$ std, 10 trials).}")
    lines.append(r"\label{tab:pretrained_per_class}")
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(f"Method & Node & {safe_names[0]} (\\%) "
                 f"& {safe_names[1]} (\\%) & {safe_names[2]} (\\%) \\\\")
    lines.append(r"\midrule")

    methods = [("individual", "Individual"), ("fedavg", "FedAvg"),
               ("fedunion", "FedUnion"), ("fedmajority", "FedMajority")]
    for i, (method_prefix, label) in enumerate(methods):
        for node in ("claudio", "paolo"):
            cells = []
            for cls in ("0", "1", "2"):
                if method_prefix == "individual":
                    key = f"individual_{node}_class{cls}"
                else:
                    key = f"{method_prefix}_{node}_round_1_class{cls}"
                if key in accs:
                    m = np.nanmean(accs[key]) * 100
                    s = np.nanstd(accs[key]) * 100
                    cells.append(f"${m:.1f} \\pm {s:.1f}$")
                else:
                    cells.append("--")
            lines.append(f"{label} & {node} & {' & '.join(cells)} \\\\")
        if i < len(methods) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze multi-trial experiment results")
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "multi_trial_results.json")
    parser.add_argument("--comprehensive-results", type=Path, default=None,
                        help="Path to comprehensive_sweep.json for extended analysis")
    parser.add_argument("--pretrained-results", type=Path, default=None,
                        help="Path to pretrained class set results JSON (for per-class table)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "analysis")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Comprehensive sweep analysis ---
    if args.comprehensive_results and args.comprehensive_results.exists():
        log.info("Analyzing comprehensive sweep results ...")
        comp_results = load_results(args.comprehensive_results)
        analysis = analyze_comprehensive_sweep(comp_results, args.output_dir)
        log.info("Comprehensive analysis complete")

    # --- Standard multi-trial analysis ---
    if not args.results.exists():
        if args.comprehensive_results:
            return  # Only comprehensive analysis was requested
        log.error("Results file not found: %s", args.results)
        sys.exit(1)

    results = load_results(args.results)
    accs = extract_accuracies(results)

    log.info("Loaded %d trials", len(results.get("trials", [])))

    # Statistical tests
    try:
        stats = compute_statistics(accs)
        stats_path = args.output_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        log.info("Statistics saved to %s", stats_path)

        for name, s in stats.items():
            log.info("  %s: %.3f vs %.3f, p=%.4f (%s)",
                     name, s["mean_a"], s["mean_b"], s["p_value"], s["test"])
    except ImportError:
        log.warning("scipy not available, skipping statistical tests")
        stats = {}

    # Effect sizes (bootstrap CIs + Cohen's d)
    effect_sizes = compute_effect_sizes(accs)
    if effect_sizes:
        es_path = args.output_dir / "effect_sizes.json"
        with open(es_path, "w") as f:
            json.dump(effect_sizes, f, indent=2)
        log.info("Effect sizes saved to %s", es_path)

        print("\n=== Effect Sizes (Cohen's d + 95%% Bootstrap CIs) ===")
        for name, es in effect_sizes.items():
            ci_a = es["ci_a"]
            ci_b = es["ci_b"]
            print(f"  {name}: d={es['cohens_d']:.3f}, "
                  f"A={es['mean_a']:.3f} [{ci_a[0]:.3f},{ci_a[1]:.3f}], "
                  f"B={es['mean_b']:.3f} [{ci_b[0]:.3f},{ci_b[1]:.3f}]")

    # Plots
    try:
        plot_convergence(results, args.output_dir)
        plot_comparison_bars(accs, args.output_dir)
    except ImportError:
        log.warning("matplotlib not available, skipping plots")

    # LaTeX tables
    main_table = generate_latex_table(accs, stats)
    table_path = args.output_dir / "results_table.tex"
    with open(table_path, "w") as f:
        f.write(main_table)
    log.info("Main results table saved to %s", table_path)

    per_class_table = generate_per_class_table(accs)
    pc_path = args.output_dir / "per_class_table.tex"
    with open(pc_path, "w") as f:
        f.write(per_class_table)
    log.info("Per-class table saved to %s", pc_path)

    # Print tables
    print("\n=== Main Results Table ===")
    print(main_table)
    print("\n=== Per-Class Table ===")
    print(per_class_table)

    if stats:
        print("\n=== Statistical Tests ===")
        for name, s in stats.items():
            sig = "*" if s["p_value"] < 0.05 else ""
            print(f"  {name}: {s['mean_a']:.3f} vs {s['mean_b']:.3f}, "
                  f"p={s['p_value']:.4f}{sig}")

    # Pretrained per-class table (if pretrained results available)
    if args.pretrained_results and args.pretrained_results.exists():
        pre_results = load_results(args.pretrained_results)
        pre_accs = extract_accuracies(pre_results)
        pre_class_table = generate_pretrained_per_class_table(
            pre_accs, class_names=["yes", "no", "stop"])
        pre_path = args.output_dir / "pretrained_per_class_table.tex"
        with open(pre_path, "w") as f:
            f.write(pre_class_table)
        log.info("Pretrained per-class table saved to %s", pre_path)
        print("\n=== Pretrained Per-Class Table ===")
        print(pre_class_table)


if __name__ == "__main__":
    main()
