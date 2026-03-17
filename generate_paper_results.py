#!/usr/bin/env python3
import csv
import os
import time
from typing import Dict, List

import numpy as np

import alternative_algorithms as alt
from run_experiments import BASE_CONFIG, build_case_from_config


PAPER_RESULTS_DIR = os.path.join("results", "paper")
SMALL_INSTANCE_CSV = os.path.join(PAPER_RESULTS_DIR, "small_instance_1h_coverage.csv")
BASELINE_SWEEP_CSV = os.path.join(PAPER_RESULTS_DIR, "coverage_horizon_baseline.csv")

HORIZON_OPTIONS = {
    "10min": 10 * 60,
    "30min": 30 * 60,
    "1h": 1 * 3600,
    "6h": 6 * 3600,
    "12h": 12 * 3600,
}

SCENARIOS = {
    "baseline": {
        # Make the paper comparison less forgiving so EDF and Random are more likely to separate.
        "battery_capacity_seconds": 2400,
        "satellite_solar_panel_power_watts": 12.0,
        "task_average_deadline": 120,
    },
}


def apply_satellite_energy_heterogeneity(e_jk: np.ndarray) -> np.ndarray:
    """
    Introduce mild but structured per-satellite heterogeneity without changing the rest of the pipeline.
    The scaling stays close to the original magnitude while making satellite choice matter more.
    """
    Ns = e_jk.shape[0]
    band = np.array([0.45, 0.65, 0.85, 1.15, 1.45], dtype=float)
    scales = np.resize(band, Ns)
    return e_jk * scales[:, None]


def truncate_case(tasks, A, e_jk, slot_len: float, horizon_sec: int):
    required_slots = int(horizon_sec / slot_len)
    return tasks, A[:, :, :required_slots], e_jk[:, :required_slots], required_slots


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    elif algorithm == "milp_small":
        result = alt.milp_small_instance(
            tasks=tasks,
            A=A,
            e_jk=e_jk,
            psi=psi,
            phi=phi,
            tau_b=tau_b,
            slot_len=slot_len,
            horizon_sec=horizon_sec,
            objective_mode="throughput",
            debug=False,
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
            debug=False,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    result["wall_clock_sec"] = time.perf_counter() - start_time
    return result


def build_case(config: Dict[str, float], horizon_label: str):
    tasks, A, e_jk, slot_len, tau_b, psi, phi, expected = build_case_from_config(config)
    e_jk = apply_satellite_energy_heterogeneity(e_jk)
    horizon_sec = HORIZON_OPTIONS[horizon_label]
    if horizon_sec > A.shape[2] * slot_len:
        raise ValueError("Requested horizon exceeds available data range.")
    tasks, A, e_jk, required_slots = truncate_case(tasks, A, e_jk, slot_len, horizon_sec)
    return tasks, A, e_jk, slot_len, tau_b, psi, phi, expected, required_slots


def build_row(
    algorithm: str,
    result: Dict[str, object],
    scenario: str,
    horizon_label: str,
    horizon_sec: int,
    required_slots: int,
    A: np.ndarray,
    slot_len: float,
    tau_b: float,
    psi: float,
    phi: float,
) -> Dict[str, object]:
    return {
        "algorithm": algorithm,
        "scenario": scenario,
        "horizon_label": horizon_label,
        "horizon_sec": horizon_sec,
        "slots": required_slots,
        "Nc": A.shape[0],
        "Ns": A.shape[1],
        "Nt": A.shape[2],
        "slot_len": slot_len,
        "battery_capacity_seconds": tau_b,
        "energy_per_cpu_cycle": psi,
        "cpu_cycles_per_second": phi,
        "coverage_ratio": result.get("coverage_ratio"),
        "completed_jobs": result.get("completed_jobs"),
        "completed_job_ratio": result.get("completed_job_ratio"),
        "wall_clock_sec": result.get("wall_clock_sec"),
        "solver_status": result.get("solver_status"),
        "objective_mode": result.get("objective_mode"),
    }


def generate_small_instance_comparison() -> None:
    scenario = "baseline"
    horizon_label = "1h"
    config = dict(BASE_CONFIG)
    config.update(SCENARIOS[scenario])
    tasks, A, e_jk, slot_len, tau_b, psi, phi, _, required_slots = build_case(config, horizon_label)
    horizon_sec = HORIZON_OPTIONS[horizon_label]
    algorithms = ["maxflow_preflow_push", "milp_small", "energy_first", "edf", "random"]

    rows = []
    for algorithm in algorithms:
        result = run_algorithm(algorithm, tasks, A, e_jk, slot_len, tau_b, psi, phi, horizon_sec)
        rows.append(build_row(
            algorithm, result, scenario, horizon_label, horizon_sec, required_slots, A, slot_len, tau_b, psi, phi
        ))
        print(f"[small-instance:{algorithm}] coverage={result['coverage_ratio']:.4f}")

    write_csv(SMALL_INSTANCE_CSV, rows)


def generate_horizon_sweep(scenario: str, out_csv: str) -> None:
    config = dict(BASE_CONFIG)
    config.update(SCENARIOS[scenario])
    algorithms = ["maxflow_preflow_push", "energy_first", "edf", "random"]

    rows = []
    for horizon_label, horizon_sec in HORIZON_OPTIONS.items():
        tasks, A, e_jk, slot_len, tau_b, psi, phi, _, required_slots = build_case(config, horizon_label)
        for algorithm in algorithms:
            result = run_algorithm(algorithm, tasks, A, e_jk, slot_len, tau_b, psi, phi, horizon_sec)
            rows.append(build_row(
                algorithm, result, scenario, horizon_label, horizon_sec, required_slots, A, slot_len, tau_b, psi, phi
            ))
            print(f"[{scenario}:{horizon_label}:{algorithm}] coverage={result['coverage_ratio']:.4f}")

    write_csv(out_csv, rows)


def main():
    generate_small_instance_comparison()
    generate_horizon_sweep("baseline", BASELINE_SWEEP_CSV)
    print(f"\nSaved paper results under {os.path.abspath(PAPER_RESULTS_DIR)}")


if __name__ == "__main__":
    main()
