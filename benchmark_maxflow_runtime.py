#!/usr/bin/env python3
import argparse
import csv
import os
import time
from typing import Dict, List

import numpy as np
import simulation_data.build_testcase as bt
import alternative_algorithms as alt


ACCESS_REPORT_PATH = os.path.join("simulation_data", "parsed_access.json")
ENERGY_REPORT_PATH = os.path.join("simulation_data", "solar_parsed.json")
RUNTIME_RESULTS_DIR = os.path.join("results", "runtime")
OUT_CSV = os.path.join(RUNTIME_RESULTS_DIR, "maxflow_runtime_by_horizon.csv")

BASE_CONFIG = {
    "energy_per_cpu_cycle": 1e-8,
    "cpu_cycles_per_second": 1e10,
    "battery_capacity_seconds": 0.5,
    "satellite_solar_panel_power_watts": 14.0,
    "time_slot_length": 60,
    "task_average_period": 220,
    "task_average_deadline": 30,
    "task_average_execution": 50,
}

HORIZON_OPTIONS = [
    ("10min", 10 * 60),
    ("1h", 1 * 3600),
    ("6h", 6 * 3600),
    ("12h", 12 * 3600),
    ("24h", 24 * 3600),
]
HORIZON_LOOKUP = dict(HORIZON_OPTIONS)
FLOW_ALGORITHMS = ["edmonds_karp", "preflow_push", "ford_fulkerson"]


def build_full_case(config: Dict[str, float]):
    np.random.seed(42)
    bt.np.random.seed(42)
    bt.satellite_solar_panel_power_watts = config["satellite_solar_panel_power_watts"]

    task_info_dict, A_list, e_jk_list, slot_len, tau_b, psi, phi, _ = bt.build_case(
        access_report_path=ACCESS_REPORT_PATH,
        energy_report_path=ENERGY_REPORT_PATH,
        energy_per_cpu_cycle=config["energy_per_cpu_cycle"],
        cpu_cycles_per_second=config["cpu_cycles_per_second"],
        battery_capacity_seconds=config["battery_capacity_seconds"],
        time_slot_length=config["time_slot_length"],
        task_average_period=config["task_average_period"],
        task_average_deadline=config["task_average_deadline"],
        task_average_execution=config["task_average_execution"],
    )

    tasks: List[alt.Task] = []
    for task_id_str, td in task_info_dict.items():
        job_exec_cycles = td.get("job_exec_cycles")
        if job_exec_cycles is None:
            job_exec_cycles = td["job_exec_time"] * config["cpu_cycles_per_second"]
        tasks.append(
            alt.Task(
                task_id=int(task_id_str),
                period=td["period"],
                deadline=td["deadline"],
                job_exec_cycles=job_exec_cycles,
                offset=td.get("offset", 0.0),
            )
        )

    A = np.array(A_list, dtype=int)
    e_jk = np.array(e_jk_list, dtype=float)
    return tasks, A, e_jk, slot_len, tau_b, psi, phi


def benchmark_single_horizon(
    tasks: List[alt.Task],
    A_full: np.ndarray,
    e_full: np.ndarray,
    slot_len: float,
    tau_b: float,
    psi: float,
    phi: float,
    horizon_sec: int,
    flow_algorithm: str,
) -> Dict[str, float]:
    required_slots = int(horizon_sec / slot_len)
    A = A_full[:, :, :required_slots]
    e_jk = e_full[:, :required_slots]

    start_total = time.perf_counter()

    jobs = alt.generate_jobs(tasks, horizon_sec=horizon_sec, phi=phi)
    tau_in = alt.convert_energy_to_time(e_jk, psi=psi, phi=phi)
    eps = alt.compute_numeric_epsilon(
        [job.demand for job in jobs],
        tau_in,
        tau_b,
        slot_len,
    )

    start_graph = time.perf_counter()
    graph, source, sink, total_demand = alt.build_scheduling_graph(
        tasks=tasks,
        jobs=jobs,
        A=A,
        tau_in=tau_in,
        tau_b=tau_b,
        slot_len=slot_len,
        eps=eps,
    )
    end_graph = time.perf_counter()

    start_flow = time.perf_counter()
    max_flow_value, _ = alt.solve_max_flow(graph, source, sink, flow_algorithm)
    end_flow = time.perf_counter()
    end_total = time.perf_counter()

    return {
        "horizon_sec": horizon_sec,
        "slots": required_slots,
        "num_jobs": len(jobs),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "total_demand": total_demand,
        "max_flow_value": max_flow_value,
        "coverage_ratio": (max_flow_value / total_demand) if total_demand > 0 else 1.0,
        "flow_algorithm": flow_algorithm,
        "graph_build_sec": end_graph - start_graph,
        "maxflow_sec": end_flow - start_flow,
        "total_runtime_sec": end_total - start_total,
    }


def write_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark max-flow runtime for different horizons.")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[label for label, _ in HORIZON_OPTIONS],
        help="Subset of horizon labels to run. Choices: 10min 1h 6h 12h 24h",
    )
    parser.add_argument(
        "--out",
        default=OUT_CSV,
        help="CSV path for benchmark results.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=FLOW_ALGORITHMS,
        help="Flow algorithms to benchmark. Choices: edmonds_karp preflow_push ford_fulkerson",
    )
    args = parser.parse_args()

    invalid_labels = [label for label in args.labels if label not in HORIZON_LOOKUP]
    if invalid_labels:
        raise ValueError(f"Unknown labels: {', '.join(invalid_labels)}")
    invalid_algorithms = [name for name in args.algorithms if name not in FLOW_ALGORITHMS]
    if invalid_algorithms:
        raise ValueError(f"Unknown algorithms: {', '.join(invalid_algorithms)}")

    tasks, A_full, e_full, slot_len, tau_b, psi, phi = build_full_case(BASE_CONFIG)
    full_horizon_sec = A_full.shape[2] * slot_len
    print(
        f"[Dataset] Nc={A_full.shape[0]} Ns={A_full.shape[1]} Nt={A_full.shape[2]} "
        f"slot_len={slot_len}",
        flush=True,
    )

    rows = []
    for label in args.labels:
        horizon_sec = HORIZON_LOOKUP[label]
        if horizon_sec > full_horizon_sec:
            print(f"[Skip] {label}: requested horizon exceeds available data.")
            continue

        for flow_algorithm in args.algorithms:
            row = benchmark_single_horizon(
                tasks=tasks,
                A_full=A_full,
                e_full=e_full,
                slot_len=slot_len,
                tau_b=tau_b,
                psi=psi,
                phi=phi,
                horizon_sec=horizon_sec,
                flow_algorithm=flow_algorithm,
            )
            row = {"label": label, **row}
            rows.append(row)

            print(
                f"[{label}][{flow_algorithm}] slots={row['slots']} jobs={row['num_jobs']} "
                f"nodes={row['num_nodes']} edges={row['num_edges']} "
                f"graph={row['graph_build_sec']:.4f}s "
                f"flow={row['maxflow_sec']:.4f}s total={row['total_runtime_sec']:.4f}s"
            )

    write_csv(args.out, rows)
    print(f"\nSaved runtime benchmark to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
