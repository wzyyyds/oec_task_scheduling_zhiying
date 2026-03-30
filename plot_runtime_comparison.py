#!/usr/bin/env python3
import argparse
import glob
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_GLOB = os.path.join("results", "runtime", "maxflow_runtime_*.csv")
DEFAULT_OUTPUT_PNG = os.path.join("plots", "runtime_vs_horizon.png")
DEFAULT_OUTPUT_PDF = os.path.join("plots", "runtime_vs_horizon.pdf")
HORIZON_ORDER = ["10min", "1h", "6h", "12h", "24h"]
HORIZON_TO_HOURS = {
    "10min": 10 / 60,
    "1h": 1,
    "6h": 6,
    "12h": 12,
    "24h": 24,
}
ALGORITHM_ORDER = ["edmonds_karp", "ford_fulkerson", "preflow_push"]
ALGORITHM_LABELS = {
    "edmonds_karp": "Edmonds-Karp",
    "ford_fulkerson": "Ford-Fulkerson",
    "preflow_push": "Push-Relabel",
}
ALGORITHM_STYLES = {
    "edmonds_karp": {"color": "#C44E52", "marker": "o"},
    "ford_fulkerson": {"color": "#55A868", "marker": "^"},
    "preflow_push": {"color": "#4C72B0", "marker": "s"},
}
RUNTIME_LABELS = {
    "total_runtime_sec": "Total runtime (seconds)",
    "maxflow_sec": "Max-flow solve time (seconds)",
    "graph_build_sec": "Graph build time (seconds)",
}


def load_runtime_csvs(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files matched: {pattern}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if {"label", "flow_algorithm"}.issubset(df.columns):
            frames.append(df)

    if not frames:
        raise ValueError("No benchmark CSV with label/flow_algorithm columns was found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["label", "flow_algorithm"])
    df = df.drop_duplicates(subset=["label", "flow_algorithm"], keep="last")
    df["horizon_hours"] = df["label"].map(HORIZON_TO_HOURS)
    df = df[df["horizon_hours"].notna()].copy()
    return df


def annotate_speedups(ax, df: pd.DataFrame, runtime_col: str) -> None:
    baseline = df[df["flow_algorithm"] == "preflow_push"].set_index("label")
    compare_algorithms = ["edmonds_karp", "ford_fulkerson"]

    for label in HORIZON_ORDER:
        if label not in baseline.index:
            continue

        x = HORIZON_TO_HOURS[label]
        baseline_runtime = float(baseline.at[label, runtime_col])
        if baseline_runtime <= 0:
            continue

        for idx, algorithm in enumerate(compare_algorithms):
            comp = df[df["flow_algorithm"] == algorithm].set_index("label")
            if label not in comp.index:
                continue

            comp_runtime = float(comp.at[label, runtime_col])
            if comp_runtime <= 0:
                continue

            y = max(comp_runtime, baseline_runtime)
            speedup = comp_runtime / baseline_runtime

            ax.annotate(
                f"{speedup:.1f}x",
                xy=(x, y),
                xytext=(0, 10 + idx * 12),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#333333",
            )


def plot_runtime(df: pd.DataFrame, output_png: str, output_pdf: str, runtime_col: str) -> None:
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for algorithm in ALGORITHM_ORDER:
        sub = df[df["flow_algorithm"] == algorithm].copy()
        if sub.empty:
            continue

        sub["label"] = pd.Categorical(sub["label"], categories=HORIZON_ORDER, ordered=True)
        sub = sub.sort_values("label")
        style = ALGORITHM_STYLES[algorithm]

        ax.plot(
            sub["horizon_hours"],
            sub[runtime_col],
            marker=style["marker"],
            linewidth=2.3,
            markersize=7,
            color=style["color"],
            label=ALGORITHM_LABELS[algorithm],
        )

    annotate_speedups(ax, df, runtime_col)

    ax.set_xlabel("Scheduling horizon", fontsize=14)
    ax.set_ylabel(RUNTIME_LABELS[runtime_col], fontsize=14)
    ax.set_xticks([HORIZON_TO_HOURS[label] for label in HORIZON_ORDER])
    ax.set_xticklabels(HORIZON_ORDER, fontsize=12)
    ax.set_yscale("log")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, which="major", linewidth=0.7, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot runtime vs horizon for max-flow algorithms.")
    parser.add_argument(
        "--input-glob",
        default=DEFAULT_INPUT_GLOB,
        help="Glob for benchmark CSV files.",
    )
    parser.add_argument(
        "--output-png",
        default=DEFAULT_OUTPUT_PNG,
        help="Output path for the PNG image.",
    )
    parser.add_argument(
        "--output-pdf",
        default=DEFAULT_OUTPUT_PDF,
        help="Output path for the PDF figure.",
    )
    parser.add_argument(
        "--runtime-col",
        default="total_runtime_sec",
        choices=["total_runtime_sec", "maxflow_sec", "graph_build_sec"],
        help="Which runtime metric to plot.",
    )
    args = parser.parse_args()

    df = load_runtime_csvs(args.input_glob)
    plot_runtime(df, args.output_png, args.output_pdf, args.runtime_col)
    print(f"Saved plot to {os.path.abspath(args.output_png)}")
    print(f"Saved plot to {os.path.abspath(args.output_pdf)}")


if __name__ == "__main__":
    main()
