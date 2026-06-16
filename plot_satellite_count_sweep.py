#!/usr/bin/env python3
import argparse
import csv
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib.pyplot as plt


DEFAULT_INPUT = os.path.join("results", "rebuttal", "satellite_count_sweep.csv")
DEFAULT_OUTPUT_PNG = os.path.join("plots", "rebuttal_satellite_count_sweep.png")
DEFAULT_OUTPUT_PDF = os.path.join("plots", "rebuttal_satellite_count_sweep.pdf")

ALGORITHM_ORDER = ["maxflow_preflow_push", "energy_first", "edf", "random"]
LABELS = {
    "maxflow_preflow_push": "ECoFlow",
    "energy_first": "Energy-first EDF",
    "edf": "EDF",
    "random": "Random",
}
STYLES = {
    "maxflow_preflow_push": {"color": "#355C7D", "marker": "o"},
    "energy_first": {"color": "#2A9D8F", "marker": "s"},
    "edf": {"color": "#F4A261", "marker": "^"},
    "random": {"color": "#E76F51", "marker": "D"},
}


def load_rows(path: str):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["satellite_count", "coverage_ratio"]:
            row[key] = float(row[key])
    return rows


def setup_axes(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot satellite-count sweep for ECoFlow.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input sweep CSV.")
    parser.add_argument("--output-png", default=DEFAULT_OUTPUT_PNG, help="Output PNG path.")
    parser.add_argument("--output-pdf", default=DEFAULT_OUTPUT_PDF, help="Output PDF path.")
    args = parser.parse_args()

    rows = load_rows(args.input)
    x_values = sorted({row["satellite_count"] for row in rows})

    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    setup_axes(ax)

    for algorithm in ALGORITHM_ORDER:
        sub = [row for row in rows if row.get("algorithm") == algorithm]
        sub = sorted(sub, key=lambda row: row["satellite_count"])
        if not sub:
            continue
        style = STYLES[algorithm]
        ax.plot(
            [row["satellite_count"] for row in sub],
            [row["coverage_ratio"] for row in sub],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=7,
            label=LABELS[algorithm],
        )

    ax.set_xlabel("Number of satellites", fontsize=17)
    ax.set_ylabel("Coverage Ratio", fontsize=17)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_values)
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(frameon=False, fontsize=14, loc="best")

    fig.tight_layout()
    fig.savefig(args.output_png, dpi=240, bbox_inches="tight")
    fig.savefig(args.output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved satellite-count sweep plot to {os.path.abspath(args.output_png)}")
    print(f"Saved satellite-count sweep plot to {os.path.abspath(args.output_pdf)}")


if __name__ == "__main__":
    main()
