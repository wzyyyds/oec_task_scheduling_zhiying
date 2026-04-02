#!/usr/bin/env python3
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib.pyplot as plt
import pandas as pd


INPUT_CSV = os.path.join("results", "paper", "parameter_sweep_1h.csv")
OUT_DIR = "plots"

PARAMETER_ORDER = [
    "battery_capacity_seconds",
    "cpu_cycles_per_second",
    "energy_per_cpu_cycle",
    "satellite_solar_panel_power_watts",
    "task_average_deadline",
    "task_average_execution",
    "task_average_period",
    "time_slot_length",
]

FILE_STEMS = {
    "battery_capacity_seconds": "paper_sweep_battery_capacity",
    "cpu_cycles_per_second": "paper_sweep_cpu_cycles",
    "energy_per_cpu_cycle": "paper_sweep_energy_per_cycle",
    "satellite_solar_panel_power_watts": "paper_sweep_solar_power",
    "task_average_deadline": "paper_sweep_average_deadline",
    "task_average_execution": "paper_sweep_average_execution",
    "task_average_period": "paper_sweep_average_period",
    "time_slot_length": "paper_sweep_time_slot_length",
}

LABELS = {
    "maxflow_preflow_push": "Max-flow",
    "energy_first": "Energy-first EDF",
    "edf": "Deadline EDF",
    "random": "Random",
}

STYLES = {
    "maxflow_preflow_push": {"color": "#355C7D", "marker": "o"},
    "energy_first": {"color": "#2A9D8F", "marker": "s"},
    "edf": {"color": "#F4A261", "marker": "^"},
    "random": {"color": "#E76F51", "marker": "D"},
}


def save(fig, stem: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, f"{stem}.png")
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def setup_axes(ax):
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)


def format_tick(v: float) -> str:
    if v >= 1e9:
        return f"{v/1e9:.2f}e9"
    if v < 1e-3:
        return f"{v:.1e}"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def format_cpu_tick(v: float) -> str:
    if v <= 0:
        return "0"
    exponent = int(f"{v:.0e}".split("e")[1])
    return rf"$10^{exponent}$"


def plot_parameter(df: pd.DataFrame, parameter_key: str) -> None:
    sub = df[df["parameter_key"] == parameter_key].copy()
    if sub.empty:
        return

    parameter_label = str(sub["parameter_label"].iloc[0])
    values = sorted(sub["parameter_value"].unique())

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.patch.set_facecolor("#FFFFFF")
    setup_axes(ax)
    x_positions = list(range(len(values)))
    x_index = {value: idx for idx, value in enumerate(values)}
    use_log_x = parameter_key == "cpu_cycles_per_second"

    for algorithm in ["maxflow_preflow_push", "energy_first", "edf", "random"]:
        alg = sub[sub["algorithm"] == algorithm].copy()
        if alg.empty:
            continue
        alg = alg.sort_values("parameter_value")
        style = STYLES[algorithm]

        ax.plot(
            alg["parameter_value"] if use_log_x else [x_index[v] for v in alg["parameter_value"]],
            alg["coverage_ratio"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=7,
            label=LABELS[algorithm],
        )

    ax.set_xlabel(parameter_label, fontsize=15)
    ax.set_ylabel("Coverage Ratio", fontsize=15)
    if parameter_key == "cpu_cycles_per_second":
        ax.set_xscale("log")
        tick_labels = [format_cpu_tick(v) for v in values]
        ax.set_xticks(values, tick_labels, rotation=20)
    else:
        tick_labels = [format_tick(v) for v in values]
        ax.set_xticks(x_positions, tick_labels, rotation=20)
    ax.tick_params(axis="both", labelsize=13)

    ax.legend(frameon=False, fontsize=12, loc="best")
    ax.set_title(f"Coverage vs. {parameter_label}", fontsize=17, pad=10)
    fig.tight_layout()
    save(fig, FILE_STEMS[parameter_key])


def main():
    df = pd.read_csv(INPUT_CSV)
    for parameter_key in PARAMETER_ORDER:
        plot_parameter(df, parameter_key)
    print(f"Saved parameter sweep figures to {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
