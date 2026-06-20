#!/usr/bin/env python3
from typing import Dict, List, Tuple

import numpy as np

import alternative_algorithms as alt


JOB_EPS = 1e-6
BATTERY_EPS = 1e-9
BATTERY_WARNING_TOL = 1e-6


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _job_workload(jobs: List[alt.Job]) -> Tuple[np.ndarray, np.ndarray]:
    demand_by_job = np.array([job.demand for job in jobs], dtype=float)
    task_by_job = np.array([job.task_id for job in jobs], dtype=int)
    return demand_by_job, task_by_job


def compute_battery_trajectory(
    assignments: List[Dict[str, float]],
    e_jk: np.ndarray,
    psi: float,
    phi: float,
    tau_b: float,
    slot_len: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Reconstruct battery in compute-time units using the same slot update model
    used by the online baselines: harvest energy, execute assigned workload,
    then cap carryover by the battery capacity.
    """
    tau_in = alt.convert_energy_to_time(e_jk, psi=psi, phi=phi)
    ns, nt = tau_in.shape
    usage = np.zeros((ns, nt), dtype=float)
    for item in assignments:
        sat_idx = int(item["sat_idx"])
        slot_idx = int(item["slot_idx"])
        if 0 <= sat_idx < ns and 0 <= slot_idx < nt:
            usage[sat_idx, slot_idx] += float(item["amount"])

    if np.isscalar(tau_b):
        capacity = np.full(ns, float(tau_b), dtype=float)
    else:
        capacity = np.asarray(tau_b, dtype=float)
        if capacity.shape != (ns,):
            raise ValueError(f"Battery capacity shape {capacity.shape} does not match satellites ({ns},)")

    battery = np.zeros(ns, dtype=float)
    trajectory = np.zeros((ns, nt), dtype=float)
    raw_min = float("inf")
    trajectory_max_ratio = float("-inf")
    for k in range(nt):
        battery += tau_in[:, k]
        battery -= usage[:, k]
        raw_min = min(raw_min, float(np.min(battery)))

        # Preserve warnings for real infeasibility, but clip tiny numerical drift
        # before computing reported battery-ratio metrics.
        battery = np.where(battery < 0.0, np.maximum(battery, 0.0), battery)
        battery = np.where(battery > capacity, np.minimum(battery, capacity), battery)
        trajectory_ratio = np.divide(
            battery,
            capacity,
            out=np.zeros_like(battery),
            where=capacity > 0,
        )
        trajectory_max_ratio = max(trajectory_max_ratio, float(np.max(trajectory_ratio)))
        trajectory[:, k] = battery

    raw_min_ratio = raw_min / float(np.max(capacity)) if capacity.size and np.max(capacity) > 0 else 0.0
    warnings = {
        "battery_invalid_low": float(raw_min < -BATTERY_WARNING_TOL),
        "battery_invalid_high": float(trajectory_max_ratio > 1.0 + BATTERY_WARNING_TOL),
        "battery_raw_min_level": raw_min if np.isfinite(raw_min) else 0.0,
        "battery_raw_min_ratio": raw_min_ratio if np.isfinite(raw_min_ratio) else 0.0,
        "battery_raw_max_ratio": trajectory_max_ratio if np.isfinite(trajectory_max_ratio) else 0.0,
    }

    return trajectory, usage, warnings


def compute_additional_metrics(
    tasks: List[alt.Task],
    A: np.ndarray,
    e_jk: np.ndarray,
    psi: float,
    phi: float,
    tau_b: float,
    slot_len: float,
    horizon_sec: float,
    assignments: List[Dict[str, float]],
    coverage_ratio: float,
) -> Dict[str, float]:
    """
    Compute reviewer-requested metrics from a finished schedule.
    The optimization objective is not changed; all quantities are post-processing
    metrics derived from actual assigned compute time.
    """
    jobs = alt.generate_jobs(tasks, horizon_sec=horizon_sec, phi=phi)
    demand_by_job, task_by_job = _job_workload(jobs)
    num_jobs = len(jobs)
    total_demand = float(np.sum(demand_by_job))

    completed_by_job = np.zeros(num_jobs, dtype=float)
    completed_by_task = np.zeros(len(tasks), dtype=float)
    demand_by_task = np.zeros(len(tasks), dtype=float)
    for job_idx, demand in enumerate(demand_by_job):
        demand_by_task[task_by_job[job_idx]] += demand

    for item in assignments:
        job_idx = int(item["job_idx"])
        if 0 <= job_idx < num_jobs:
            amount = float(item["amount"])
            completed_by_job[job_idx] += amount
            completed_by_task[task_by_job[job_idx]] += amount

    completed_workload = float(np.sum(completed_by_job))
    completed_fraction = np.divide(
        completed_by_job,
        demand_by_job,
        out=np.zeros_like(completed_by_job),
        where=demand_by_job > 0,
    )
    completed_jobs = int(np.sum(completed_fraction >= 1.0 - JOB_EPS))
    missed_jobs = num_jobs - completed_jobs

    total_task_energy = completed_workload * psi * phi
    total_energy_consumption = total_task_energy

    battery_trajectory, usage_by_sat_slot, battery_warnings = compute_battery_trajectory(
        assignments=assignments,
        e_jk=e_jk,
        psi=psi,
        phi=phi,
        tau_b=tau_b,
        slot_len=slot_len,
    )

    if np.isscalar(tau_b):
        capacity = np.full(battery_trajectory.shape[0], float(tau_b), dtype=float)
    else:
        capacity = np.asarray(tau_b, dtype=float)

    if battery_trajectory.size and np.all(capacity > 0):
        battery_ratio = battery_trajectory / capacity[:, None]
    else:
        battery_ratio = np.zeros_like(battery_trajectory)

    task_coverage = np.divide(
        completed_by_task,
        demand_by_task,
        out=np.zeros_like(completed_by_task),
        where=demand_by_task > 0,
    )
    fairness_den = len(tasks) * float(np.sum(task_coverage ** 2))
    jain = float((np.sum(task_coverage) ** 2) / fairness_den) if fairness_den > 0 else 0.0

    min_battery_by_slot = np.min(battery_ratio, axis=0).tolist() if battery_ratio.size else []
    avg_battery_by_slot = np.mean(battery_ratio, axis=0).tolist() if battery_ratio.size else []

    result = {
        "coverage_ratio": coverage_ratio,
        "completed_workload": completed_workload,
        "total_released_workload": total_demand,
        "job_deadline_miss_ratio": _safe_div(missed_jobs, num_jobs),
        "job_completion_ratio": _safe_div(completed_jobs, num_jobs),
        "workload_deadline_miss_ratio": 1.0 - coverage_ratio,
        "completed_jobs_metric": completed_jobs,
        "missed_jobs": missed_jobs,
        "total_task_energy": total_task_energy,
        "total_energy_consumption": total_energy_consumption,
        "energy_per_completed_workload": _safe_div(total_task_energy, completed_workload),
        "energy_per_completed_job": _safe_div(total_task_energy, completed_jobs),
        "min_battery_level": float(np.min(battery_trajectory)) if battery_trajectory.size else 0.0,
        "min_battery_ratio": float(np.min(battery_ratio)) if battery_ratio.size else 0.0,
        "avg_battery_ratio": float(np.mean(battery_ratio)) if battery_ratio.size else 0.0,
        "average_battery_ratio": float(np.mean(battery_ratio)) if battery_ratio.size else 0.0,
        "low_battery_slot_ratio_5": float(np.mean(battery_ratio <= 0.05 + BATTERY_EPS)) if battery_ratio.size else 0.0,
        "low_battery_slot_ratio_10": float(np.mean(battery_ratio <= 0.10 + BATTERY_EPS)) if battery_ratio.size else 0.0,
        "depletion_slot_ratio_0": float(np.mean(battery_ratio <= BATTERY_EPS)) if battery_ratio.size else 0.0,
        "depletion_slot_ratio_5": float(np.mean(battery_ratio <= 0.05 + BATTERY_EPS)) if battery_ratio.size else 0.0,
        "final_avg_battery_ratio": float(np.mean(battery_ratio[:, -1])) if battery_ratio.size else 0.0,
        "mean_per_task_coverage": float(np.mean(task_coverage)) if task_coverage.size else 0.0,
        "min_per_task_coverage": float(np.min(task_coverage)) if task_coverage.size else 0.0,
        "std_per_task_coverage": float(np.std(task_coverage)) if task_coverage.size else 0.0,
        "jain_fairness_index": jain,
        "avg_battery_ratio_trajectory": avg_battery_by_slot,
        "min_battery_ratio_trajectory": min_battery_by_slot,
        "total_scheduled_energy_seconds": float(np.sum(usage_by_sat_slot)),
    }
    result.update(battery_warnings)
    return result
