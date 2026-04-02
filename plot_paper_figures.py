#!/usr/bin/env python3
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd


PAPER_RESULTS_DIR = os.path.join("results", "paper")
PLOTS_DIR = "plots"

SMALL_INSTANCE_CSV = os.path.join(PAPER_RESULTS_DIR, "small_instance_1h_coverage.csv")
BASELINE_SWEEP_CSV = os.path.join(PAPER_RESULTS_DIR, "coverage_horizon_baseline.csv")

HORIZON_ORDER = ["10min", "30min", "1h", "6h", "12h"]
HORIZON_TO_HOURS = {
    "10min": 10 / 60,
    "30min": 30 / 60,
    "1h": 1,
    "6h": 6,
    "12h": 12,
}

LABELS = {
    "maxflow_preflow_push": "Max-flow",
    "milp_small": "MILP",
    "energy_first": "Energy-first EDF",
    "edf": "Deadline EDF",
    "random": "Random",
}

COLORS = {
    "maxflow_preflow_push": "#355C7D",
    "milp_small": "#6C5B7B",
    "energy_first": "#2A9D8F",
    "edf": "#F4A261",
    "random": "#E76F51",
}

MARKERS = {
    "maxflow_preflow_push": "o",
    "energy_first": "s",
    "edf": "^",
    "random": "D",
}


def save(fig, stem: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    png = os.path.join(PLOTS_DIR, f"{stem}.png")
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def setup_axes(ax):
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)


def add_bar_labels(ax, values, y_positions, x_offset: float) -> None:
    for value, y_pos in zip(values, y_positions):
        ax.text(value + x_offset, y_pos, f"{value:.3f}", va="center", ha="left", fontsize=12)


def plot_small_instance_coverage():
    df = pd.read_csv(SMALL_INSTANCE_CSV)
    order = ["maxflow_preflow_push", "milp_small", "energy_first", "edf", "random"]
    df = df[df["algorithm"].isin(order)].copy()
    df["algorithm"] = pd.Categorical(df["algorithm"], categories=order, ordered=True)
    df = df.sort_values("algorithm")

    labels = [LABELS[a] for a in df["algorithm"]]
    colors = [COLORS[a] for a in df["algorithm"]]
    values = df["coverage_ratio"].to_list()
    y = range(len(df))

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    fig.patch.set_facecolor("#FFFFFF")
    setup_axes(ax)
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(list(y), labels, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel("Coverage Ratio", fontsize=15)
    ax.set_xlim(0, max(values) * 1.18)
    ax.set_title("Small Instance Coverage Comparison (1h)", fontsize=17, pad=10)
    ax.tick_params(axis="x", labelsize=13)

    add_bar_labels(ax, values, y, max(values) * 0.015)

    fig.tight_layout()
    save(fig, "paper_small_instance_coverage")


def _plot_horizon_sweep(df: pd.DataFrame, metric: str, ylabel: str, title: str, stem: str):
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    setup_axes(ax)

    for algorithm in ["maxflow_preflow_push", "energy_first", "edf", "random"]:
        sub = df[df["algorithm"] == algorithm].copy()
        if sub.empty:
            continue
        sub["horizon_label"] = pd.Categorical(sub["horizon_label"], categories=HORIZON_ORDER, ordered=True)
        sub = sub.sort_values("horizon_label")
        ax.plot(
            [HORIZON_TO_HOURS[h] for h in sub["horizon_label"]],
            sub[metric],
            linewidth=2.2,
            marker=MARKERS[algorithm],
            markersize=7,
            color=COLORS[algorithm],
            label=LABELS[algorithm],
        )

    ax.set_xlabel("Scheduling Horizon", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xticks([HORIZON_TO_HOURS[h] for h in HORIZON_ORDER], HORIZON_ORDER)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(frameon=False, fontsize=12, loc="best")
    ax.set_title(title, fontsize=17, pad=10)
    fig.tight_layout()
    save(fig, stem)


def plot_baseline_horizon_sweep():
    df = pd.read_csv(BASELINE_SWEEP_CSV)
    _plot_horizon_sweep(
        df,
        "coverage_ratio",
        "Coverage Ratio",
        "Coverage vs. Horizon (Baseline)",
        "paper_coverage_vs_horizon_baseline",
    )

def plot_baseline_completed_horizon_sweep():
    df = pd.read_csv(BASELINE_SWEEP_CSV)
    _plot_horizon_sweep(
        df,
        "completed_job_ratio",
        "Completed Job Ratio",
        "Completed Jobs vs. Horizon (Baseline)",
        "paper_completed_jobs_vs_horizon_baseline",
    )


def plot_paper_panel():
    small_df = pd.read_csv(SMALL_INSTANCE_CSV)
    baseline_df = pd.read_csv(BASELINE_SWEEP_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    for ax in axes:
        setup_axes(ax)

    order = ["maxflow_preflow_push", "milp_small", "energy_first", "edf", "random"]
    small_df = small_df[small_df["algorithm"].isin(order)].copy()
    small_df["algorithm"] = pd.Categorical(small_df["algorithm"], categories=order, ordered=True)
    small_df = small_df.sort_values("algorithm")
    y = range(len(small_df))
    values = small_df["coverage_ratio"].to_list()
    axes[0].barh(y, values, color=[COLORS[a] for a in small_df["algorithm"]], height=0.62)
    axes[0].set_yticks(list(y), [LABELS[a] for a in small_df["algorithm"]], fontsize=12)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(values) * 1.18)
    axes[0].set_xlabel("Coverage Ratio", fontsize=14)
    axes[0].set_title("(a) 1h Small Instance", fontsize=15, pad=8)
    axes[0].tick_params(axis="x", labelsize=12)
    add_bar_labels(axes[0], values, y, max(values) * 0.015)

    for algorithm in ["maxflow_preflow_push", "energy_first", "edf", "random"]:
        sub = baseline_df[baseline_df["algorithm"] == algorithm].copy()
        sub["horizon_label"] = pd.Categorical(sub["horizon_label"], categories=HORIZON_ORDER, ordered=True)
        sub = sub.sort_values("horizon_label")
        axes[1].plot(
            [HORIZON_TO_HOURS[h] for h in sub["horizon_label"]],
            sub["coverage_ratio"],
            linewidth=2.1,
            marker=MARKERS[algorithm],
            markersize=6.5,
            color=COLORS[algorithm],
            label=LABELS[algorithm],
        )
    axes[1].set_xticks([HORIZON_TO_HOURS[h] for h in HORIZON_ORDER], HORIZON_ORDER)
    axes[1].set_xlabel("Horizon", fontsize=14)
    axes[1].set_ylabel("Coverage Ratio", fontsize=14)
    axes[1].tick_params(axis="both", labelsize=12)
    axes[1].set_title("(b) Baseline Horizon Sweep", fontsize=15, pad=8)
    axes[1].legend(frameon=False, fontsize=11, loc="lower right")
    fig.tight_layout()
    save(fig, "paper_algorithm_comparison_panel")


def main():
    plot_small_instance_coverage()
    plot_baseline_horizon_sweep()
    plot_baseline_completed_horizon_sweep()
    plot_paper_panel()
    print(f"Saved paper figures to {os.path.abspath(PLOTS_DIR)}")


if __name__ == "__main__":
    main()
