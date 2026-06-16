#!/usr/bin/env python3
import argparse
import csv
import os
import time
from typing import Dict, List

import numpy as np

import alternative_algorithms as alt
from run_experiments import BASE_CONFIG, build_case_from_config


OUT_DIR = os.path.join("results", "rebuttal")
OUT_CSV = os.path.join(OUT_DIR, "satellite_count_sweep.csv")
DEFAULT_HORIZON_SEC = 24 * 3600
DEFAULT_SATELLITE_COUNTS = [24, 48, 72, 96, 120, 136, 192, 272]
ALGORITHMS = ["maxflow_preflow_push", "energy_first", "edf", "random"]


def uniform_satellite_indices(total_satellites: int, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("Satellite count must be positive.")
    if count > total_satellites:
        raise ValueError(f"Requested {count} satellites, but only {total_satellites} are available.")
    if count == total_satellites:
        return np.arange(total_satellites, dtype=int)
    return np.linspace(0, total_satellites - 1, count, dtype=int)


def expand_constellation(A: np.ndarray, e_jk: np.ndarray, target_count: int):
    base_count = A.shape[1]
    if target_count <= base_count:
        indices = uniform_satellite_indices(base_count, target_count)
        return A[:, indices, :], e_jk[indices, :]

    copies = int(np.ceil(target_count / base_count))
    A_blocks = []
    e_blocks = []
    nt = A.shape[2]
    for copy_idx in range(copies):
        shift = int(round(copy_idx * nt / copies))
        A_blocks.append(np.roll(A, shift=shift, axis=2))
        e_blocks.append(np.roll(e_jk, shift=shift, axis=1))

    A_expanded = np.concatenate(A_blocks, axis=1)
    e_expanded = np.concatenate(e_blocks, axis=0)
    return A_expanded[:, :target_count, :], e_expanded[:target_count, :]


def truncate_horizon(A: np.ndarray, e_jk: np.ndarray, slot_len: float, horizon_sec: int):
    slots = int(horizon_sec / slot_len)
    return A[:, :, :slots], e_jk[:, :slots], slots


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
    start = time.perf_counter()
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

    result["wall_clock_sec"] = time.perf_counter() - start
    return result


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "algorithm",
        "satellite_count",
        "horizon_sec",
        "slots",
        "Nc",
        "Ns",
        "Nt",
        "num_jobs",
        "num_nodes",
        "num_edges",
        "coverage_ratio",
        "completed_job_ratio",
        "max_flow_value",
        "total_demand",
        "wall_clock_sec",
        "graph_build_time_sec",
        "max_flow_time_sec",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep satellite count for ECoFlow.")
    parser.add_argument(
        "--satellite-counts",
        nargs="*",
        type=int,
        default=DEFAULT_SATELLITE_COUNTS,
        help="Satellite counts to test.",
    )
    parser.add_argument(
        "--horizon-sec",
        type=int,
        default=DEFAULT_HORIZON_SEC,
        help="Scheduling horizon in seconds.",
    )
    parser.add_argument("--out", default=OUT_CSV, help="Output CSV path.")
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=ALGORITHMS,
        help="Algorithms to test. Choices: maxflow_preflow_push energy_first edf random",
    )
    args = parser.parse_args()

    invalid_algorithms = [algorithm for algorithm in args.algorithms if algorithm not in ALGORITHMS]
    if invalid_algorithms:
        raise ValueError(f"Unknown algorithms: {', '.join(invalid_algorithms)}")

    tasks, A_full, e_full, slot_len, tau_b, psi, phi, _ = build_case_from_config(dict(BASE_CONFIG))
    full_horizon_sec = A_full.shape[2] * slot_len
    if args.horizon_sec > full_horizon_sec:
        raise ValueError("Requested horizon exceeds available data range.")

    A_horizon, e_horizon, slots = truncate_horizon(A_full, e_full, slot_len, args.horizon_sec)
    rows: List[Dict[str, object]] = []

    print(
        f"[Dataset] Nc={A_full.shape[0]} Ns={A_full.shape[1]} Nt={A_full.shape[2]} "
        f"slot_len={slot_len} horizon={args.horizon_sec}s",
        flush=True,
    )

    for count in args.satellite_counts:
        A, e_jk = expand_constellation(A_horizon, e_horizon, count)

        for algorithm in args.algorithms:
            result = run_algorithm(algorithm, tasks, A, e_jk, slot_len, tau_b, psi, phi, args.horizon_sec)
            row = {
                "algorithm": algorithm,
                "satellite_count": count,
                "horizon_sec": args.horizon_sec,
                "slots": slots,
                "Nc": A.shape[0],
                "Ns": A.shape[1],
                "Nt": A.shape[2],
                "num_jobs": result.get("num_jobs"),
                "num_nodes": result.get("num_nodes"),
                "num_edges": result.get("num_edges"),
                "coverage_ratio": result.get("coverage_ratio"),
                "completed_job_ratio": result.get("completed_job_ratio"),
                "max_flow_value": result.get("max_flow_value"),
                "total_demand": result.get("total_demand"),
                "wall_clock_sec": result.get("wall_clock_sec"),
                "graph_build_time_sec": result.get("graph_build_time_sec"),
                "max_flow_time_sec": result.get("max_flow_time_sec"),
            }
            rows.append(row)
            write_csv(args.out, rows)
            print(
                f"[Ns={count}][{algorithm}] coverage={row['coverage_ratio']:.4f}",
                flush=True,
            )

    print(f"Saved satellite-count sweep to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
