#!/usr/bin/env python3
import argparse
import csv
import os
import time
from typing import Dict, List, Tuple

import numpy as np

import alternative_algorithms as alt
from run_experiments import BASE_CONFIG, build_case_from_config


OUT_DIR = os.path.join("results", "rebuttal")
OUT_CSV = os.path.join(OUT_DIR, "milp_horizon_comparison.csv")

HORIZON_OPTIONS: List[Tuple[str, int]] = [
    ("10min", 10 * 60),
    ("1h", 1 * 3600),
    ("2h", 2 * 3600),
    ("3h", 3 * 3600),
    ("4h", 4 * 3600),
    ("5h", 5 * 3600),
    ("6h", 6 * 3600),
    ("7h", 7 * 3600),
    ("8h", 8 * 3600),
    ("9h", 9 * 3600),
    ("10h", 10 * 3600),
    ("11h", 11 * 3600),
    ("12h", 12 * 3600),
]
HORIZON_LOOKUP = dict(HORIZON_OPTIONS)
HORIZON_RANK = {label: idx for idx, (label, _) in enumerate(HORIZON_OPTIONS)}
ALGORITHMS = ["ecoflow", "milp"]


def truncate_case(tasks, A: np.ndarray, e_jk: np.ndarray, slot_len: float, horizon_sec: int):
    slots = int(horizon_sec / slot_len)
    return tasks, A[:, :, :slots], e_jk[:, :slots], slots


def run_ecoflow(tasks, A: np.ndarray, e_jk: np.ndarray, slot_len: float, tau_b: float, psi: float, phi: float, horizon_sec: int):
    start = time.perf_counter()
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
    result["wall_clock_sec"] = time.perf_counter() - start
    result["solver_status"] = "Optimal"
    result["num_candidate_assignments"] = ""
    result["solve_time_sec"] = result.get("max_flow_time_sec")
    return result


def run_milp(
    tasks,
    A: np.ndarray,
    e_jk: np.ndarray,
    slot_len: float,
    tau_b: float,
    psi: float,
    phi: float,
    horizon_sec: int,
    time_limit_sec: int,
    max_feasible_assignments: int,
):
    start = time.perf_counter()
    result = alt.milp_small_instance(
        tasks=tasks,
        A=A,
        e_jk=e_jk,
        psi=psi,
        phi=phi,
        tau_b=tau_b,
        slot_len=slot_len,
        horizon_sec=horizon_sec,
        time_limit_sec=time_limit_sec,
        max_feasible_assignments=max_feasible_assignments,
        objective_mode="throughput",
        debug=False,
    )
    result["wall_clock_sec"] = time.perf_counter() - start
    if result.get("solver_status") == "skipped_too_large":
        result["coverage_ratio"] = ""
        result["completed_job_ratio"] = ""
        result["max_flow_value"] = ""
        result["wall_clock_sec"] = ""
        result["solve_time_sec"] = ""
    return result


def build_row(algorithm: str, label: str, horizon_sec: int, slots: int, A: np.ndarray, result: Dict[str, object]):
    return {
        "algorithm": algorithm,
        "horizon_label": label,
        "horizon_sec": horizon_sec,
        "slots": slots,
        "Nc": A.shape[0],
        "Ns": A.shape[1],
        "Nt": A.shape[2],
        "num_jobs": result.get("num_jobs"),
        "num_nodes": result.get("num_nodes"),
        "num_edges": result.get("num_edges"),
        "num_candidate_assignments": result.get("num_candidate_assignments"),
        "coverage_ratio": result.get("coverage_ratio"),
        "completed_job_ratio": result.get("completed_job_ratio"),
        "max_flow_value": result.get("max_flow_value"),
        "total_demand": result.get("total_demand"),
        "wall_clock_sec": result.get("wall_clock_sec"),
        "graph_build_time_sec": result.get("graph_build_time_sec"),
        "max_flow_time_sec": result.get("max_flow_time_sec"),
        "solve_time_sec": result.get("solve_time_sec"),
        "solver_status": result.get("solver_status"),
    }


def write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "algorithm",
        "horizon_label",
        "horizon_sec",
        "slots",
        "Nc",
        "Ns",
        "Nt",
        "num_jobs",
        "num_nodes",
        "num_edges",
        "num_candidate_assignments",
        "coverage_ratio",
        "completed_job_ratio",
        "max_flow_value",
        "total_demand",
        "wall_clock_sec",
        "graph_build_time_sec",
        "max_flow_time_sec",
        "solve_time_sec",
        "solver_status",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: str) -> List[Dict[str, object]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def merge_rows(existing_rows: List[Dict[str, object]], new_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    merged = {}
    for row in existing_rows + new_rows:
        key = (row.get("horizon_label"), row.get("algorithm"))
        merged[key] = row
    return sorted(
        merged.values(),
        key=lambda row: (HORIZON_RANK.get(row.get("horizon_label"), 10**9), ALGORITHMS.index(row.get("algorithm", "milp"))),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ECoFlow vs MILP horizon comparison.")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=[label for label, _ in HORIZON_OPTIONS],
        help="Horizon labels to run. Choices: " + " ".join(HORIZON_LOOKUP),
    )
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=ALGORITHMS,
        help="Algorithms to run. Choices: ecoflow milp",
    )
    parser.add_argument("--out", default=OUT_CSV, help="Output CSV path.")
    parser.add_argument("--append", action="store_true", help="Merge new rows into an existing output CSV.")
    parser.add_argument("--milp-time-limit", type=int, default=300, help="MILP/CBC time limit in seconds.")
    parser.add_argument(
        "--max-feasible-assignments",
        type=int,
        default=25000,
        help="Skip MILP cases with more candidate assignment variables than this.",
    )
    args = parser.parse_args()

    invalid_labels = [label for label in args.labels if label not in HORIZON_LOOKUP]
    if invalid_labels:
        raise ValueError(f"Unknown horizon labels: {', '.join(invalid_labels)}")
    invalid_algorithms = [algorithm for algorithm in args.algorithms if algorithm not in ALGORITHMS]
    if invalid_algorithms:
        raise ValueError(f"Unknown algorithms: {', '.join(invalid_algorithms)}")

    tasks, A_full, e_full, slot_len, tau_b, psi, phi, _ = build_case_from_config(dict(BASE_CONFIG))
    full_horizon_sec = A_full.shape[2] * slot_len
    print(
        f"[Dataset] Nc={A_full.shape[0]} Ns={A_full.shape[1]} Nt={A_full.shape[2]} slot_len={slot_len}",
        flush=True,
    )

    rows: List[Dict[str, object]] = []
    for label in args.labels:
        horizon_sec = HORIZON_LOOKUP[label]
        if horizon_sec > full_horizon_sec:
            print(f"[Skip] {label}: requested horizon exceeds available data.")
            continue

        _, A, e_jk, slots = truncate_case(tasks, A_full, e_full, slot_len, horizon_sec)
        print(f"[Case:{label}] slots={slots}", flush=True)

        if "ecoflow" in args.algorithms:
            result = run_ecoflow(tasks, A, e_jk, slot_len, tau_b, psi, phi, horizon_sec)
            rows.append(build_row("ecoflow", label, horizon_sec, slots, A, result))
            print(f"  ecoflow coverage={result['coverage_ratio']:.4f} runtime={result['wall_clock_sec']:.3f}s")

        if "milp" in args.algorithms:
            result = run_milp(
                tasks,
                A,
                e_jk,
                slot_len,
                tau_b,
                psi,
                phi,
                horizon_sec,
                time_limit_sec=args.milp_time_limit,
                max_feasible_assignments=args.max_feasible_assignments,
            )
            rows.append(build_row("milp", label, horizon_sec, slots, A, result))
            if result.get("solver_status") == "skipped_too_large":
                print(
                    "  milp skipped: "
                    f"{result.get('num_candidate_assignments')} candidate assignments exceed "
                    f"limit {args.max_feasible_assignments}"
                )
            else:
                print(
                    f"  milp coverage={result['coverage_ratio']:.4f} "
                    f"runtime={result['wall_clock_sec']:.3f}s status={result.get('solver_status')}"
                )

        output_rows = merge_rows(read_existing_rows(args.out), rows) if args.append else rows
        write_csv(args.out, output_rows)

    print(f"Saved MILP horizon comparison to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
