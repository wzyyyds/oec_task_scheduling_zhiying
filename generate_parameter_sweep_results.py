#!/usr/bin/env python3
import csv
import os
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np

import alternative_algorithms as alt
from run_experiments import BASE_CONFIG, build_case_from_config


OUT_DIR = os.path.join("results", "paper")
OUT_CSV = os.path.join(OUT_DIR, "parameter_sweep_1h.csv")
HORIZON_SEC = 3600

SWEEPS: List[Tuple[str, str, List[float]]] = [
    ("battery_capacity_seconds", "Battery capacity (s)", [0, 0.5, 1.0, 1.5, 2.0]),
    ("cpu_cycles_per_second", "CPU cycles per second", [1e7, 1e8, 1e9, 1e10, 1e11]),
    ("energy_per_cpu_cycle", "Energy per CPU cycle", [1e-9, 1e-8, 1e-7]),
    ("satellite_solar_panel_power_watts", "Solar panel power (W)", [1, 5, 10, 15, 20, 25, 30]),
    ("task_average_deadline", "Average deadline (s)", [10, 20, 30, 40, 50, 60]),
    ("task_average_execution", "Average execution time (s)", [30, 35, 40, 45, 50, 55, 60]),
    ("task_average_period", "Average period (s)", [100, 120, 140, 160, 180, 200]),
    ("time_slot_length", "Time slot length (s)", [60]),
]

ALGORITHMS = [
    "maxflow_preflow_push",
    "energy_first",
    "edf",
    "random",
]


def truncate_case(tasks, A, e_jk, slot_len: float, horizon_sec: int):
    required_slots = int(horizon_sec / slot_len)
    return tasks, A[:, :, :required_slots], e_jk[:, :required_slots], required_slots


def run_algorithm(
    algorithm: str,
    tasks,
    A: np.ndarray,
    e_jk: np.ndarray,
    slot_len: float,
    tau_b: float,
    psi: float,
    phi: float,
    horizon_sec: int,
) -> Dict[str, object]:
    start_time = time.perf_counter()
    if algorithm == "maxflow_preflow_push":
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
            debug=False,
            flow_algorithm="preflow_push",
        )
    elif algorithm == "energy_first":
        result = alt.heuristic_most_energy_first(
            tasks=tasks,
            A=A,
            e_jk=e_jk,
            psi=psi,
            phi=phi,
            tau_b=tau_b,
            slot_len=slot_len,
            horizon_sec=horizon_sec,
            debug=False,
        )
    elif algorithm == "edf":
        result = alt.heuristic_edf(
            tasks=tasks,
            A=A,
            e_jk=e_jk,
            psi=psi,
            phi=phi,
            tau_b=tau_b,
            slot_len=slot_len,
            horizon_sec=horizon_sec,
            debug=False,
        )
    elif algorithm == "random":
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
            debug=False,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    result["wall_clock_sec"] = time.perf_counter() - start_time
    return result


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows: List[Dict[str, object]] = []

    for parameter_key, parameter_label, values in SWEEPS:
        for value in values:
            config = dict(BASE_CONFIG)
            config[parameter_key] = value

            tasks, A, e_jk, slot_len, tau_b, psi, phi, _ = build_case_from_config(config)
            if HORIZON_SEC > A.shape[2] * slot_len:
                raise ValueError("Requested parameter-sweep horizon exceeds available data range.")

            tasks, A, e_jk, required_slots = truncate_case(tasks, A, e_jk, slot_len, HORIZON_SEC)

            for algorithm in ALGORITHMS:
                result = run_algorithm(algorithm, tasks, A, e_jk, slot_len, tau_b, psi, phi, HORIZON_SEC)
                rows.append({
                    "parameter_key": parameter_key,
                    "parameter_label": parameter_label,
                    "parameter_value": value,
                    "algorithm": algorithm,
                    "horizon_sec": HORIZON_SEC,
                    "slot_len": slot_len,
                    "slots": required_slots,
                    "coverage_ratio": result.get("coverage_ratio"),
                    "completed_job_ratio": result.get("completed_job_ratio"),
                    "wall_clock_sec": result.get("wall_clock_sec"),
                    "battery_capacity_seconds": tau_b,
                    "energy_per_cpu_cycle": psi,
                    "cpu_cycles_per_second": phi,
                    "satellite_solar_panel_power_watts": config["satellite_solar_panel_power_watts"],
                    "task_average_deadline": config["task_average_deadline"],
                    "task_average_execution": config["task_average_execution"],
                    "task_average_period": config["task_average_period"],
                    "time_slot_length": config["time_slot_length"],
                })
            print(f"[{parameter_key}={value}] done")

    write_csv(OUT_CSV, rows)
    print(f"Saved parameter sweep results to {os.path.abspath(OUT_CSV)}")


if __name__ == "__main__":
    main()
