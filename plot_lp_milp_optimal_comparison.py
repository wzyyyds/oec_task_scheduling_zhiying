#!/usr/bin/env python3
import argparse
import csv
import os
import tempfile
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib.pyplot as plt


DEFAULT_INPUT = os.path.join("results", "rebuttal", "lp_milp_optimal_comparison.csv")
DEFAULT_RUNTIME_PNG = os.path.join("plots", "rebuttal_lp_milp_runtime.png")
DEFAULT_RUNTIME_PDF = os.path.join("plots", "rebuttal_lp_milp_runtime.pdf")
DEFAULT_COVERAGE_PNG = os.path.join("plots", "rebuttal_lp_milp_coverage.png")
DEFAULT_COVERAGE_PDF = os.path.join("plots", "rebuttal_lp_milp_coverage.pdf")

ALGORITHM_ORDER = ["ecoflow", "lp_optimal"]
ALGORITHM_LABELS = {
    "ecoflow": "ECoFlow (max-flow)",
    "lp_optimal": "LP optimal",
}
ALGORITHM_STYLES = {
    "ecoflow": {"color": "#355C7D", "marker": "o"},
    "lp_optimal": {"color": "#C44E52", "marker": "s"},
}
X_LABELS = {
    "slots": "Time slots",
    "num_nodes": "Graph nodes",
    "num_edges": "Graph edges",
    "num_candidate_assignments": "MILP candidate assignments",
}


def as_float(value):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_rows(path: str) -> List[Dict[str, object]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    numeric_columns = [
        "slots",
        "num_nodes",
        "num_edges",
        "num_candidate_assignments",
        "coverage_ratio",
        "wall_clock_sec",
    ]
    for row in rows:
        for col in numeric_columns:
            row[col] = as_float(row.get(col))
    return rows


def x_value(row: Dict[str, object], x_axis: str):
    if x_axis in {"slots", "num_candidate_assignments"}:
        return row.get(x_axis)
    if row.get(x_axis) is not None:
        return row.get(x_axis)

    # LP rows do not build the time-expanded max-flow graph, so mirror the
    # ECoFlow graph size for the same horizon when the caller asks for graph size.
    return row.get(f"ecoflow_{x_axis}")


def attach_ecoflow_graph_sizes(rows: List[Dict[str, object]]) -> None:
    by_horizon = {
        row["horizon_label"]: row
        for row in rows
        if row.get("algorithm") == "ecoflow"
    }
    for row in rows:
        ecoflow_row = by_horizon.get(row["horizon_label"])
        if not ecoflow_row:
            continue
        row["ecoflow_num_nodes"] = ecoflow_row.get("num_nodes")
        row["ecoflow_num_edges"] = ecoflow_row.get("num_edges")


def setup_axes(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.45)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.25)


def plot_metric(
    rows: List[Dict[str, object]],
    x_axis: str,
    metric: str,
    ylabel: str,
    output_png: str,
    output_pdf: str,
    log_y: bool = False,
) -> None:
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    setup_axes(ax)

    for algorithm in ALGORITHM_ORDER:
        sub = [row for row in rows if row.get("algorithm") == algorithm]
        points = []
        for row in sub:
            x = x_value(row, x_axis)
            y = row.get(metric)
            if x is None or y is None:
                continue
            if metric == "wall_clock_sec" and y <= 0:
                continue
            points.append((x, y))
        points.sort(key=lambda item: item[0])
        if not points:
            continue

        style = ALGORITHM_STYLES[algorithm]
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker=style["marker"],
            linewidth=2.3,
            markersize=7,
            color=style["color"],
            label=ALGORITHM_LABELS[algorithm],
        )

    ax.set_xlabel(X_LABELS[x_axis], fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
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
    parser = argparse.ArgumentParser(description="Plot ECoFlow vs LP/MILP optimal rebuttal figures.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input comparison CSV.")
    parser.add_argument(
        "--x-axis",
        default="slots",
        choices=["slots", "num_nodes", "num_edges", "num_candidate_assignments"],
        help="X axis for both figures.",
    )
    parser.add_argument("--runtime-png", default=DEFAULT_RUNTIME_PNG)
    parser.add_argument("--runtime-pdf", default=DEFAULT_RUNTIME_PDF)
    parser.add_argument("--coverage-png", default=DEFAULT_COVERAGE_PNG)
    parser.add_argument("--coverage-pdf", default=DEFAULT_COVERAGE_PDF)
    args = parser.parse_args()

    rows = load_rows(args.input)
    attach_ecoflow_graph_sizes(rows)
    plot_metric(
        rows,
        args.x_axis,
        "wall_clock_sec",
        "Runtime (seconds)",
        args.runtime_png,
        args.runtime_pdf,
        log_y=True,
    )
    plot_metric(
        rows,
        args.x_axis,
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
