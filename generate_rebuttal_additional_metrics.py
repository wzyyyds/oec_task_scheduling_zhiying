#!/usr/bin/env python3
import argparse
import csv
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np

import additional_metrics as metrics
import alternative_algorithms as alt
from run_experiments import BASE_CONFIG, build_case_from_config


OUT_DIR = os.path.join("results", "rebuttal")
PLOTS_DIR = "plots"
OUT_CSV = os.path.join(OUT_DIR, "additional_metrics_quick.csv")
TRAJ_CSV = os.path.join(OUT_DIR, "battery_trajectory_representative.csv")

HORIZON_OPTIONS: List[Tuple[str, int]] = [
    ("10min", 10 * 60),
    ("1h", 1 * 3600),
    ("2h", 2 * 3600),
    ("3h", 3 * 3600),
    ("6h", 6 * 3600),
    ("12h", 12 * 3600),
]
HORIZON_LOOKUP = dict(HORIZON_OPTIONS)
HORIZON_DISPLAY_LABELS = {
    "10min": "10 min",
    "1h": "1 h",
    "2h": "2 h",
    "3h": "3 h",
    "6h": "6 h",
    "12h": "12 h",
}
QUICK_LABELS = ["10min", "1h", "2h", "3h"]
METHODS = ["ecoflow", "energy_first", "edf", "random", "milp"]
PLOT_METHODS = ["ecoflow", "energy_first", "edf", "random"]
METHOD_LABELS = {
    "ecoflow": "ECoFlow",
    "energy_first": "Energy-first EDF",
    "edf": "EDF",
    "random": "Random",
    "milp": "MILP",
}
STYLES = {
    "ecoflow": {"color": "#355C7D", "marker": "o"},
    "energy_first": {"color": "#2A9D8F", "marker": "s"},
    "edf": {"color": "#F4A261", "marker": "^"},
    "random": {"color": "#E76F51", "marker": "D"},
    "milp": {"color": "#C44E52", "marker": "x"},
}


def truncate_case(A: np.ndarray, e_jk: np.ndarray, slot_len: float, horizon_sec: int):
    slots = int(horizon_sec / slot_len)
    return A[:, :, :slots], e_jk[:, :slots], slots


def run_method(method: str, tasks, A, e_jk, slot_len, tau_b, psi, phi, horizon_sec, milp_time_limit: int):
    start = time.perf_counter()
    if method == "ecoflow":
        result = alt.feasibility_test(
            tasks=tasks,
            A=A,
            e_jk=e_jk,
            psi=psi,
            phi=phi,
            tau_b=tau_b,
            slot_len=slot_len,
            horizon_sec=horizon_sec,
            return_flow=False,
            return_schedule=True,
            debug=False,
            flow_algorithm="preflow_push",
        )
    elif method == "energy_first":
        result = alt.heuristic_most_energy_first(
            tasks=tasks,
            A=A,
            e_jk=e_jk,
            psi=psi,
            phi=phi,
            tau_b=tau_b,
            slot_len=slot_len,
            horizon_sec=horizon_sec,
            return_schedule=True,
            debug=False,
        )
    elif method == "edf":
        result = alt.heuristic_edf(
            tasks=tasks,
            A=A,
            e_jk=e_jk,
            psi=psi,
            phi=phi,
            tau_b=tau_b,
            slot_len=slot_len,
            horizon_sec=horizon_sec,
            return_schedule=True,
            debug=False,
        )
    elif method == "random":
        result = alt.heuristic_random_assignment(
            tasks=tasks,
            A=A,
            e_jk=e_jk,
            psi=psi,
            phi=phi,
            tau_b=tau_b,
            slot_len=slot_len,
            horizon_sec=horizon_sec,
            random_seed=42,
            return_schedule=True,
            debug=False,
        )
    elif method == "milp":
        result = alt.milp_small_instance(
            tasks=tasks,
            A=A,
            e_jk=e_jk,
            psi=psi,
            phi=phi,
            tau_b=tau_b,
            slot_len=slot_len,
            horizon_sec=horizon_sec,
            time_limit_sec=milp_time_limit,
            max_feasible_assignments=60000,
            objective_mode="throughput",
            return_schedule=True,
            debug=False,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    result["wall_clock_sec"] = time.perf_counter() - start
    return result


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    excluded = {"avg_battery_ratio_trajectory", "min_battery_ratio_trajectory"}
    fieldnames = [key for key in rows[0].keys() if key not in excluded]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{k: v for k, v in row.items() if k in fieldnames} for row in rows])


def write_trajectory_csv(path: str, representative_label: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    traj_rows = [row for row in rows if row["horizon_label"] == representative_label]
    if not traj_rows:
        return
    max_len = max(len(row["avg_battery_ratio_trajectory"]) for row in traj_rows)
    avg_by_method = {row["method"]: row["avg_battery_ratio_trajectory"] for row in traj_rows}
    min_by_method = {row["method"]: row["min_battery_ratio_trajectory"] for row in traj_rows}
    with open(path, "w", newline="") as f:
        fieldnames = ["slot"]
        for method in avg_by_method:
            fieldnames.extend([f"avg_{method}", f"min_{method}"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for slot in range(max_len):
            out = {"slot": slot}
            for method, trajectory in avg_by_method.items():
                out[f"avg_{method}"] = trajectory[slot] if slot < len(trajectory) else ""
                min_trajectory = min_by_method[method]
                out[f"min_{method}"] = min_trajectory[slot] if slot < len(min_trajectory) else ""
            writer.writerow(out)


def setup_axes(ax) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)


def plot_metric(
    rows,
    metric_key: str,
    ylabel: str,
    stem: str,
    labels: List[str],
    title: str = "",
    legend_outside: bool = False,
    legend_anchor: Optional[Tuple[float, float]] = None,
) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    x_positions = {label: idx for idx, label in enumerate(labels)}
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    setup_axes(ax)
    for method in PLOT_METHODS:
        sub = [row for row in rows if row["method"] == method and row["horizon_label"] in x_positions]
        if not sub:
            continue
        sub.sort(key=lambda row: x_positions[row["horizon_label"]])
        style = STYLES[method]
        ax.plot(
            [x_positions[row["horizon_label"]] for row in sub],
            [float(row[metric_key]) for row in sub],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=7,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel("Scheduling horizon", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xticks(
        [x_positions[label] for label in labels],
        [HORIZON_DISPLAY_LABELS.get(label, label) for label in labels],
    )
    if title:
        ax.set_title(title, fontsize=15, pad=10)
    ax.tick_params(axis="both", labelsize=13)
    if legend_outside:
        ax.legend(frameon=False, fontsize=11, bbox_to_anchor=(1.02, 0.5), loc="center left")
    elif legend_anchor is not None:
        ax.legend(frameon=False, fontsize=11, bbox_to_anchor=legend_anchor, loc="center")
    else:
        ax.legend(frameon=False, fontsize=11, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.png"), dpi=240, bbox_inches="tight")
    fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_battery_trajectory(
    rows,
    representative_label: str,
    trajectory_key: str,
    ylabel: str,
    stem: str,
    slot_len: float,
) -> None:
    traj_rows = [row for row in rows if row["horizon_label"] == representative_label]
    if not traj_rows:
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    setup_axes(ax)
    for method in PLOT_METHODS:
        matching = [row for row in traj_rows if row["method"] == method]
        if not matching:
            continue
        row = matching[0]
        method = row["method"]
        trajectory = row[trajectory_key]
        horizon_sec = float(row["horizon_sec"])
        if horizon_sec >= 3600.0:
            x_values = np.arange(len(trajectory), dtype=float) * slot_len / 3600.0
            xlabel = "Elapsed time (h)"
        else:
            x_values = np.arange(len(trajectory), dtype=float) * slot_len / 60.0
            xlabel = "Elapsed time (min)"
        style = STYLES[method]
        ax.plot(
            x_values,
            trajectory,
            color=style["color"],
            linewidth=2.0,
            label=METHOD_LABELS[method],
        )
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(frameon=False, fontsize=11, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.png"), dpi=240, bbox_inches="tight")
    fig.savefig(os.path.join(PLOTS_DIR, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: List[Dict[str, object]]) -> None:
    print("\nSummary (coverage / job miss / battery depletion / Jain):")
    print("method,horizon,coverage,job_miss,min_battery_ratio,low_battery_slot_ratio_5,avg_battery_ratio,jain")
    for row in rows:
        print(
            f"{row['method']},{row['horizon_label']},"
            f"{row['coverage_ratio']:.4f},{row['job_deadline_miss_ratio']:.4f},"
            f"{row['min_battery_ratio']:.4f},{row['low_battery_slot_ratio_5']:.4f},"
            f"{row['avg_battery_ratio']:.4f},"
            f"{row['jain_fairness_index']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reviewer-requested additional metrics.")
    parser.add_argument("--quick_rebuttal_metrics", action="store_true", help="Run the quick rebuttal subset.")
    parser.add_argument("--labels", nargs="*", default=None, help="Horizon labels to run.")
    parser.add_argument("--methods", nargs="*", default=METHODS, help="Methods to run.")
    parser.add_argument("--out", default=OUT_CSV, help="Output metrics CSV.")
    parser.add_argument("--trajectory-out", default=TRAJ_CSV, help="Output battery trajectory CSV.")
    parser.add_argument("--representative-label", default="1h", help="Horizon used for battery trajectory plot.")
    parser.add_argument("--milp-time-limit", type=int, default=300)
    args = parser.parse_args()

    labels = args.labels if args.labels is not None else (QUICK_LABELS if args.quick_rebuttal_metrics else QUICK_LABELS)
    invalid_labels = [label for label in labels if label not in HORIZON_LOOKUP]
    if invalid_labels:
        raise ValueError(f"Unknown horizon labels: {', '.join(invalid_labels)}")
    invalid_methods = [method for method in args.methods if method not in METHODS]
    if invalid_methods:
        raise ValueError(f"Unknown methods: {', '.join(invalid_methods)}")

    tasks, A_full, e_full, slot_len, tau_b, psi, phi, _ = build_case_from_config(dict(BASE_CONFIG))
    rows: List[Dict[str, object]] = []
    for label in labels:
        horizon_sec = HORIZON_LOOKUP[label]
        A, e_jk, slots = truncate_case(A_full, e_full, slot_len, horizon_sec)
        print(f"[Case:{label}] slots={slots}")
        for method in args.methods:
            result = run_method(method, tasks, A, e_jk, slot_len, tau_b, psi, phi, horizon_sec, args.milp_time_limit)
            if result.get("solver_status") == "skipped_too_large":
                continue
            metric_row = metrics.compute_additional_metrics(
                tasks=tasks,
                A=A,
                e_jk=e_jk,
                psi=psi,
                phi=phi,
                tau_b=tau_b,
                slot_len=slot_len,
                horizon_sec=horizon_sec,
                assignments=result.get("schedule", []),
                coverage_ratio=float(result.get("coverage_ratio", 0.0)),
            )
            metric_row.update({
                "method": method,
                "horizon_label": label,
                "horizon_sec": horizon_sec,
                "slots": slots,
                "wall_clock_sec": result.get("wall_clock_sec"),
                "solver_status": result.get("solver_status", ""),
            })
            rows.append(metric_row)
            if metric_row.get("battery_invalid_low", 0.0) or metric_row.get("battery_invalid_high", 0.0):
                print(
                    f"  WARNING {method}/{label}: reconstructed battery ratio out of bounds "
                    f"(raw_min={metric_row['battery_raw_min_ratio']:.3e}, "
                    f"raw_max={metric_row['battery_raw_max_ratio']:.3e})"
                )
            print(
                f"  {method}: coverage={metric_row['coverage_ratio']:.4f}, "
                f"job_miss={metric_row['job_deadline_miss_ratio']:.4f}, "
                f"min_battery={metric_row['min_battery_ratio']:.4f}, "
                f"jain={metric_row['jain_fairness_index']:.4f}"
            )

    write_csv(args.out, rows)
    write_trajectory_csv(args.trajectory_out, args.representative_label, rows)
    print_summary(rows)

    plot_metric(
        rows,
        "job_deadline_miss_ratio",
        "Job deadline miss ratio",
        "rebuttal_deadline_miss_vs_horizon",
        labels,
        legend_anchor=(0.68, 0.34),
    )
    plot_metric(rows, "min_battery_ratio", "Minimum battery ratio", "battery_min_ratio_vs_horizon", labels)
    plot_metric(
        rows,
        "low_battery_slot_ratio_5",
        r"Low-battery slot ratio ($b/B \leq 5\%$)",
        "battery_low_slot_ratio_5_vs_horizon",
        labels,
    )
    plot_metric(rows, "jain_fairness_index", "Jain fairness index", "rebuttal_jain_fairness_vs_horizon", labels)
    plot_battery_trajectory(
        rows,
        args.representative_label,
        "avg_battery_ratio_trajectory",
        "Average battery ratio",
        "battery_avg_trajectory",
        slot_len,
    )
    plot_battery_trajectory(
        rows,
        args.representative_label,
        "min_battery_ratio_trajectory",
        "Minimum battery ratio",
        "battery_min_trajectory",
        slot_len,
    )
    print(f"\nSaved metrics CSV to {os.path.abspath(args.out)}")
    print(f"Saved battery trajectory CSV to {os.path.abspath(args.trajectory_out)}")
    print(f"Saved figures to {os.path.abspath(PLOTS_DIR)}")


if __name__ == "__main__":
    main()
