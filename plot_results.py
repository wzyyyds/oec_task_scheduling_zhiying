#!/usr/bin/env python3
"""
Plot coverage_ratio vs different config parameters from results_all.csv.

Usage:
    python plot_results.py

Assumptions:
    - results_all.csv has columns:
        tag, algorithm,
        coverage_ratio, feasible,
        energy_per_cpu_cycle, cpu_cycles_per_second,
        battery_capacity_seconds, satellite_solar_panel_power_watts,
        time_slot_length, task_average_period,
        task_average_deadline, task_average_execution,
        ... (and possibly others)
    - Each experiment is a single-parameter sweep:
        i.e., for a given parameter, all other config fields stay at baseline.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ===== User config =====

RESULT_CSV = "results_all.csv"
OUT_DIR = "plots"

# baseline config (用于识别单参数 sweep 子集)
BASE_CONFIG = {
    "energy_per_cpu_cycle":      1e-9,
    "cpu_cycles_per_second":     1e10,
    "battery_capacity_seconds":  7200,
    "satellite_solar_panel_power_watts": 40.0,
    "time_slot_length":          60,
    "task_average_period":       300,
    "task_average_deadline":     300,
    "task_average_execution":    50,
}

# 想画的算法顺序（只画在 csv 里实际存在的）
ALGORITHM_ORDER = [
    "maxflow",
    "energy_first",   # heuristic_most_energy_first
    # "edf",
    # "random",
]

# 算法名称映射，用于图例显示
ALGORITHM_LABEL_MAP = {
    "maxflow": "Max-flow",
    "energy_first": "Energy-first EDF",
}

# 想检查的参数 & 横轴标题
PARAMS_TO_PLOT = [
    ("battery_capacity_seconds",         "Battery capacity (seconds)"),
    ("satellite_solar_panel_power_watts","Solar panel power (W)"),
    ("time_slot_length",                 "Time slot length (seconds)"),
    ("task_average_execution",           "Average execution time (seconds)"),
    ("task_average_period",              "Average period (seconds)"),
    ("task_average_deadline",            "Average deadline (seconds)"),
    ("energy_per_cpu_cycle",             "Energy per CPU cycle (J)"),
    ("cpu_cycles_per_second",            "CPU cycles per second"),
]


# ===== Helpers =====

def ensure_outdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def filter_single_param_sweep(df: pd.DataFrame, param: str) -> pd.DataFrame:
    """
    从结果里提取“只改变 param，其它保持 baseline”的子集。
    """
    sub = df.copy()
    for k, base_v in BASE_CONFIG.items():
        if k == param:
            continue
        if k not in sub.columns:
            continue
        # 这些配置在 CSV 里是干净的小数，可以直接用 ==
        sub = sub[sub[k] == base_v]
    return sub


def plot_param(df: pd.DataFrame, param: str, xlabel: str):
    """
    对给定 param，画 coverage_ratio vs param 的多算法折线图。
    """
    sub = filter_single_param_sweep(df, param)

    if param not in sub.columns:
        print(f"[Skip] {param}: column not found.")
        return
    if sub[param].nunique() <= 1:
        print(f"[Skip] {param}: not enough variation after baseline filter.")
        return

    algs_present = [a for a in ALGORITHM_ORDER if a in sub["algorithm"].unique()]
    if not algs_present:
        print(f"[Skip] {param}: no known algorithms present.")
        return

    plt.figure(figsize=(6, 4))  # 控制图大小，让文字适配更美观

    for alg in algs_present:
        df_alg = sub[sub["algorithm"] == alg].copy()
        if df_alg.empty:
            continue

        df_alg = (
            df_alg
            .groupby(param, as_index=False)["coverage_ratio"]
            .mean()
        ).sort_values(by=param)

        x = df_alg[param].values
        y = df_alg["coverage_ratio"].values
        label = ALGORITHM_LABEL_MAP.get(alg, alg)

        plt.plot(
            x, y,
            marker="o",
            linewidth=2,
            markersize=8,
            label=label
        )

    # 字体大小调整
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel("Coverage ratio", fontsize=18)

    # 刻度字体
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(True, linewidth=0.6)
    plt.legend(fontsize=16)

    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f"coverage_vs_{param}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[OK] Saved {out_path}")


# ===== Main =====

def main():
    if not os.path.exists(RESULT_CSV):
        print(f"[Error] {RESULT_CSV} not found in current directory.")
        return

    ensure_outdir(OUT_DIR)

    df = pd.read_csv(RESULT_CSV)

    if "algorithm" not in df.columns or "coverage_ratio" not in df.columns:
        print("[Error] results_all.csv missing required columns: 'algorithm', 'coverage_ratio'.")
        return

    for param, xlabel in PARAMS_TO_PLOT:
        plot_param(df, param, xlabel)


if __name__ == "__main__":
    main()
