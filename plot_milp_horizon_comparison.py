#!/usr/bin/env python3
import argparse
import csv
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib.pyplot as plt


DEFAULT_INPUT = os.path.join("results", "rebuttal", "milp_horizon_comparison.csv")
DEFAULT_RUNTIME_PNG = os.path.join("plots", "rebuttal_milp_runtime_vs_horizon.png")
DEFAULT_RUNTIME_PDF = os.path.join("plots", "rebuttal_milp_runtime_vs_horizon.pdf")
DEFAULT_COVERAGE_PNG = os.path.join("plots", "rebuttal_milp_coverage_vs_horizon.png")
DEFAULT_COVERAGE_PDF = os.path.join("plots", "rebuttal_milp_coverage_vs_horizon.pdf")

HORIZON_ORDER = ["10min", "1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h"]
HORIZON_TO_XPOS = {
    "10min": 0.6,
    "1h": 1,
    "2h": 2,
    "3h": 3,
    "4h": 4,
    "5h": 5,
    "6h": 6,
    "7h": 7,
    "8h": 8,
    "9h": 9,
    "10h": 10,
    "11h": 11,
    "12h": 12,
}
ALGORITHM_ORDER = ["ecoflow", "milp"]
LABELS = {
    "ecoflow": "ECoFlow",
    "milp": "MILP",
}
STYLES = {
    "ecoflow": {"color": "#355C7D", "marker": "o"},
    "milp": {"color": "#C44E52", "marker": "s"},
}


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path: str):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ["coverage_ratio", "wall_clock_sec"]:
            row[key] = as_float(row.get(key))
    return rows


def setup_axes(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linewidth=0.7, alpha=0.7)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.35)


def plot_metric(rows, metric: str, ylabel: str, output_png: str, output_pdf: str, log_y: bool = False) -> None:
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    setup_axes(ax)

    for algorithm in ALGORITHM_ORDER:
        sub = [row for row in rows if row.get("algorithm") == algorithm]
        points = []
        for row in sub:
            x = HORIZON_TO_XPOS.get(row.get("horizon_label"))
            y = row.get(metric)
            if x is None or y is None:
                continue
            if metric == "wall_clock_sec" and y <= 0:
                continue
            points.append((x, y))
        points.sort(key=lambda item: item[0])
        if not points:
            continue

        style = STYLES[algorithm]
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker=style["marker"],
            linewidth=2.3,
            markersize=7,
            color=style["color"],
            label=LABELS[algorithm],
        )

    ax.set_xlabel("Scheduling horizon", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks([HORIZON_TO_XPOS[label] for label in HORIZON_ORDER], HORIZON_ORDER)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_xlim(0, 12.5)
    if log_y:
        ax.set_yscale("log")
    if metric == "coverage_ratio":
        ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=12, loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ECoFlow vs MILP horizon comparison.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input comparison CSV.")
    parser.add_argument("--runtime-png", default=DEFAULT_RUNTIME_PNG)
    parser.add_argument("--runtime-pdf", default=DEFAULT_RUNTIME_PDF)
    parser.add_argument("--coverage-png", default=DEFAULT_COVERAGE_PNG)
    parser.add_argument("--coverage-pdf", default=DEFAULT_COVERAGE_PDF)
    args = parser.parse_args()

    rows = load_rows(args.input)
    plot_metric(
        rows,
        "wall_clock_sec",
        "Runtime (seconds)",
        args.runtime_png,
        args.runtime_pdf,
        log_y=True,
    )
    plot_metric(
        rows,
        "coverage_ratio",
        "Coverage ratio",
        args.coverage_png,
        args.coverage_pdf,
        log_y=False,
    )
    print(f"Saved runtime plot to {os.path.abspath(args.runtime_png)}")
    print(f"Saved runtime plot to {os.path.abspath(args.runtime_pdf)}")
    print(f"Saved coverage plot to {os.path.abspath(args.coverage_png)}")
    print(f"Saved coverage plot to {os.path.abspath(args.coverage_pdf)}")


if __name__ == "__main__":
    main()
