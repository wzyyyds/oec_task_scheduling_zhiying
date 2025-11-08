#!/usr/bin/env python3
"""
Max-flow–based scheduling feasibility test for periodic EO tasks over a satellite constellation.

Model summary (discrete time):
- Time is slotted: slots k = 0..Nt-1, each of length slot_len (seconds).
- Tasks i = 0..Nc-1 with period T_i (sec), relative deadline D_i (sec), and per-job execution demand p_i (sec).
  Jobs of task i release at T_i, 2T_i, ... within [0, horizon_sec), each due by (release + D_i).
- Coverage matrix A[i, j, k] in {0,1} indicates if satellite j can serve task i during slot k.
- Satellite j has input energy e_jk (Joules) for slot k. Convert to time τ_{j,k} = e_jk / (psi * phi) (seconds).
- Battery carryover: energy node (j,k) -> (j,k-1) with capacity τ_b (seconds) if k >= 1.

Graph:
- Source -> Task_i edge cap = total demand of all Task_i jobs (sum of p_i over all releases).
- Task_i -> each of its Job nodes edge cap = p_i (job's demanded execution time).
- Job -> SatelliteTime(j,k) edges if:
    (a) A[i,j,k] == 1,
    (b) slot k starts >= job release,
    (c) slot k ends   <= job deadline.
  Each such edge cap = slot_len.
- SatelliteTime(j,k) -> Energy(j,k) edge cap = slot_len.
- Energy(j,k) -> Sink edge cap = τ_{j,k}.
- Energy(j,k) -> Energy(j,k-1) edge cap = τ_b (k>=1).

Feasibility:
- Let D_total = sum of all job demands (seconds).
- Run Edmonds–Karp. If max_flow == D_total (within tolerance), the instance is feasible.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import math
import numpy as np
import networkx as nx
import numpy as np
import time
import os
import tqdm
import pandas as pd

# ----------------------------
# Data structures
# ----------------------------

@dataclass
class Task:
    """A periodic task."""
    task_id: int
    period: float          # seconds, T_i
    deadline: float        # seconds, D_i (relative to release)
    job_exec_time: float   # seconds of processing required per job (p_i)
    # Optional: allow a start offset if needed in the future (default releases at T_i, 2T_i, ...)
    offset: float = 0.0    # seconds; by default keep 0.0 so first release is at T_i (per spec)


@dataclass
class Job:
    """A single job (release) of a task."""
    job_id: str           # unique label, e.g., f"J_task{task_id}_r{r_idx}"
    task_id: int
    release: float        # absolute release time (sec)
    deadline_abs: float   # absolute deadline time (sec) = release + D_i
    demand: float         # seconds required = p_i


# ----------------------------
# Helpers
# ----------------------------

def generate_jobs(tasks: List[Task], horizon_sec: float) -> List[Job]:
    """
    Generate all jobs over [0, horizon_sec).
    Per problem statement: releases at T_i, 2T_i, ... (NOT at 0).
    """
    jobs: List[Job] = []
    for t in tasks:
        r_idx = 1
        while True:
            release = t.offset + r_idx * t.period
            if release >= horizon_sec:
                break
            deadline_abs = release + t.deadline
            jobs.append(Job(
                job_id=f"J_task{t.task_id}_r{r_idx}",
                task_id=t.task_id,
                release=release,
                deadline_abs=deadline_abs,
                demand=t.job_exec_time
            ))
            r_idx += 1
    return jobs


def slot_bounds(slot_len: float, k: int) -> Tuple[float, float]:
    """Return (start_time, end_time) for slot k."""
    start = k * slot_len
    end = (k + 1) * slot_len
    return start, end


def convert_energy_to_time(e_jk: np.ndarray, psi: float, phi: float) -> np.ndarray:
    """
    Convert energy (Joules) to seconds of compute time: tau = e / (psi * phi).
    e_jk shape = (Ns, Nt)
    """
    if psi <= 0 or phi <= 0:
        raise ValueError("psi and phi must be positive.")
    return e_jk / (psi * phi)


# ----------------------------
# Graph construction
# ----------------------------

def build_scheduling_graph(
    tasks: List[Task],
    jobs: List[Job],
    A: np.ndarray,             # shape (Nc, Ns, Nt), entries in {0,1}
    tau_in: np.ndarray,        # shape (Ns, Nt), seconds from input energy for slot k
    tau_b: float,              # battery capacity in seconds (converted time)
    slot_len: float            # seconds per time slot
) -> Tuple[nx.DiGraph, str, str, float]:
    """
    Build the flow network graph as described.

    Returns:
        G: the directed graph
        s: source node label
        t: sink node label
        D_total: total demand (sum of all job demands)
    """
    Nc, Ns, Nt = A.shape
    if tau_in.shape != (Ns, Nt):
        raise ValueError(f"tau_in shape {tau_in.shape} must be (Ns, Nt)=({Ns},{Nt}).")
    if any(tasks[i].task_id != i for i in range(Nc)):
        raise ValueError("Expect tasks[i].task_id == i for all i to align with A indexing.")
    if slot_len <= 0:
        raise ValueError("slot_len must be positive.")

    # Index helpers for node names
    def task_node(i: int) -> str:
        return f"TASK_{i}"

    def job_node(job: Job) -> str:
        return f"JOB_{job.job_id}"

    def st_node(j: int, k: int) -> str:
        return f"ST_{j}_{k}"   # satellite-time node

    def en_node(j: int, k: int) -> str:
        return f"EN_{j}_{k}"   # energy node

    s, t = "SRC", "SNK"
    G = nx.DiGraph()

    # Source and sink
    G.add_node(s)
    G.add_node(t)

    # 1) Source -> Task_i with capacity = sum of demands of Task_i's jobs
    # 2) Task_i -> each Job of that task with cap = job demand
    D_total = 0.0
    jobs_by_task: Dict[int, List[Job]] = {}
    for job in jobs:
        jobs_by_task.setdefault(job.task_id, []).append(job)

    for i in range(Nc):
        G.add_node(task_node(i))
        task_jobs = jobs_by_task.get(i, [])
        demand_sum = sum(j.demand for j in task_jobs)
        if demand_sum > 0:
            G.add_edge(s, task_node(i), capacity=demand_sum)
        for j in task_jobs:
            G.add_node(job_node(j))
            if j.demand > 0:
                G.add_edge(task_node(i), job_node(j), capacity=j.demand)
        D_total += demand_sum

    # 3) Job -> SatelliteTime edges according to visibility + window [release, deadline]
    #    capacity = slot_len
    for job in jobs:
        i = job.task_id
        for j in range(Ns):
            for k in range(Nt):
                if A[i, j, k] != 1:
                    continue
                slot_start, slot_end = slot_bounds(slot_len, k)
                if slot_start >= job.release and slot_end <= job.deadline_abs:
                    G.add_node(st_node(j, k))
                    G.add_edge(job_node(job), st_node(j, k), capacity=slot_len)

    # 4) SatelliteTime(j,k) -> Energy(j,k) with capacity = slot_len
    # 5) Energy(j,k) -> Sink with capacity = tau_in[j,k]
    # 6) Energy(j,k) -> Energy(j,k-1) with capacity = tau_b, for k >= 1
    for j in range(Ns):
        for k in range(Nt):
            st = st_node(j, k)
            en = en_node(j, k)
            # It's okay to add nodes/edges even if they aren't used by any job
            G.add_node(st)
            G.add_node(en)
            # ST -> EN
            G.add_edge(st, en, capacity=slot_len)
            # EN -> Sink
            tau = float(tau_in[j, k])
            if tau < 0:
                raise ValueError("tau_in must be non-negative.")
            G.add_edge(en, t, capacity=tau)
            # EN(k) -> EN(k-1) battery carryover
            if k >= 1 and tau_b > 0:
                G.add_edge(en, en_node(j, k - 1), capacity=tau_b)

    return G, s, t, D_total


# ----------------------------
# Feasibility test
# ----------------------------

def feasibility_test(
    tasks: List[Task],
    A: np.ndarray,          # (Nc, Ns, Nt) binary
    e_jk: np.ndarray,       # (Ns, Nt) energy in Joules per slot
    psi: float,             # Joules per CPU cycle
    phi: float,             # cycles per second
    tau_b: float,           # seconds (converted battery capacity)
    slot_len: float,        # seconds per slot
    horizon_sec: float,     # total horizon in seconds (Nt * slot_len)
    flow_tolerance: float = 1e-7,
    return_flow: bool = False
) -> Dict[str, Any]:
    """
    Build the network and run Edmonds–Karp max flow.
    Returns dict with feasibility, max_flow_value, total_demand, and (optionally) flow_dict.
    """
    Nc, Ns, Nt = A.shape
    assert horizon_sec > 0 and math.isclose(horizon_sec, Nt * slot_len, rel_tol=0, abs_tol=1e-6), \
        "horizon_sec must equal Nt * slot_len."

    # Generate jobs from periodic tasks
    print("Generating jobs...")
    jobs = generate_jobs(tasks, horizon_sec=horizon_sec)

    # Convert energy to time per slot
    print("Converting energy to time...")
    tau_in = convert_energy_to_time(e_jk, psi=psi, phi=phi)  # (Ns, Nt)

    # Build graph
    print("Building scheduling graph...")
    start_graph_time = time.perf_counter()
    G, s, t, D_total = build_scheduling_graph(tasks, jobs, A, tau_in, tau_b, slot_len)
    end_graph_time = time.perf_counter()
    print(f"Graph built in {end_graph_time - start_graph_time:.4f} sec: "
          f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Run Edmonds–Karp
    print("Running max-flow (Edmonds–Karp)...")
    start_flow_time = time.perf_counter()
    maxflow_value, flow_dict = nx.maximum_flow(
        G, s, t, flow_func=nx.algorithms.flow.edmonds_karp
    )
    end_flow_time = time.perf_counter()
    print(f"Max-flow computed in {end_flow_time - start_flow_time:.4f} sec.")
    
    print("Calculating feasibility...")
    feasible = (maxflow_value + flow_tolerance >= D_total)

    result = {
        "feasible": feasible,
        "max_flow_value": maxflow_value,
        "total_demand": D_total,
        "num_jobs": len(jobs),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "graph_build_time_sec": end_graph_time - start_graph_time,
        "max_flow_time_sec": end_flow_time - start_flow_time
    }
    if return_flow:
        result["flow"] = flow_dict
        result["graph"] = G
    return result

def run_case(name, tasks, A, e_jk, slot_len, tau_b, psi, phi, expected):
    Nt = A.shape[2]
    horizon_sec = Nt * slot_len
    start_time = time.perf_counter()
    res = feasibility_test(
        tasks=tasks,
        A=A,
        e_jk=e_jk,
        psi=psi,
        phi=phi,
        tau_b=tau_b,
        slot_len=slot_len,
        horizon_sec=horizon_sec,
        return_flow=False
    )
    end_time = time.perf_counter()
    print(f"[{name}] feasible={res['feasible']}  (expected={expected})  "
          f"flow={res['max_flow_value']:.1f} / demand={res['total_demand']:.1f}  "
          f"jobs={res['num_jobs']} nodes={res['num_nodes']} edges={res['num_edges']}  "
          f"time={end_time - start_time:.4f} sec")
    # return the time for running the case
    return end_time - start_time, res

'''
Build a test case with given parameters.
Parameters:
- psi: energy per CPU cycle (J)
- phi: CPU cycles per second
- tau_b: battery capacity in seconds
- slot_len: length of each time slot in seconds
- Nc: number of tasks
- Nt: number of time slots
- Ns: number of satellites
- task_avg_period: average period of tasks in seconds
- task_avg_deadline: average relative deadline of tasks in seconds
- task_avg_exec: average execution time per job in seconds
- visit_prob: probability that a task can see a satellite in a slot
Returns:
- tasks: list of Task objects
- A: visibility matrix (Nc, Ns, Nt)
- e_jk: energy matrix (Ns, Nt)
- slot_len: length of each time slot in seconds
- tau_b: battery capacity in seconds
- psi: energy per CPU cycle (J)
- phi: CPU cycles per second
- expected_feasible: whether the case is expected to be feasible (None as unknown)
'''

def build_case(psi, phi, tau_b, slot_len, Nc, Nt, Ns, task_avg_period, task_avg_deadline, task_avg_exec, visit_prob) -> Tuple[List[Task], np.ndarray, np.ndarray, float, float, float, float, Any]:
    tasks = []
    for i in range(Nc):
        period = np.random.exponential(task_avg_period)
        deadline = min(period, np.random.exponential(task_avg_deadline))
        exec_time = min(deadline, np.random.exponential(task_avg_exec))
        tasks.append(Task(
            task_id=i,
            period=period,
            deadline=deadline,
            job_exec_time=exec_time
        ))

    A = (np.random.rand(Nc, Ns, Nt) < visit_prob).astype(int)

    e_jk = np.random.exponential(10.0, size=(Ns, Nt))  # average 10 Joules per slot

    expected_feasible = None  # unknown

    return tasks, A, e_jk, slot_len, tau_b, psi, phi, expected_feasible


if __name__ == "__main__":
    random_seed_list = [0, 1, 2]
    Nc_list = [50] # varying number of tasks for random cases
    Ns_list = [100] # varying number of satellites for random cases
    Nt_list = [48] # varying number of time slots for random cases
    slot_len_list = [600.0] # varying slot lengths for random cases
    visiting_prob_list = [0.1, 0.3, 0.5, 0.7, 0.9] # varying visiting probabilities for random cases
    PSI = 1.0  # Joules per CPU cycle
    PHI = 1.0  # CPU cycles per second
    TAU_B = 50.0  # battery capacity in seconds
    setup_running_time_list = []
    pbar = tqdm.tqdm(total=len(random_seed_list) * len(Nc_list) * len(Ns_list) * len(Nt_list) * len(slot_len_list) * len(visiting_prob_list), desc="Running efficiency test cases")
    for random_seed in random_seed_list:
        np.random.seed(random_seed)
        for nc in Nc_list:
            for ns in Ns_list:
                for nt in Nt_list:
                    for slot_len in slot_len_list:
                        for visit_prob in visiting_prob_list:
                            tasks, A, e_jk, slot_len, tau_b, psi, phi, expected_feasible = build_case(
                                psi=PSI,
                                phi=PHI,
                                tau_b=TAU_B,
                                slot_len=slot_len,
                                Nc=nc,
                                Nt=nt,
                                Ns=ns,
                                task_avg_period= 5 * slot_len,
                                task_avg_deadline= 3 * slot_len,
                                task_avg_exec= 2 * slot_len,
                                visit_prob=visit_prob
                            )
                            case_name = f"Nc{nc}_Ns{ns}_Nt{nt}_Slot{slot_len}"
                            total_running_time, stats = run_case(case_name, tasks, A, e_jk, slot_len, tau_b, psi, phi, expected_feasible)
                            this_round_stats = {
                                "case_name": case_name,
                                "random_seed": random_seed,
                                "Nc": nc,
                                "Ns": ns,
                                "Nt": nt,
                                "slot_len": slot_len,
                                "visit_prob": visit_prob,
                                "psi": psi,
                                "phi": phi,
                                "tau_b": tau_b,
                                "running_time_sec": total_running_time,
                                "feasible": stats["feasible"],
                                "expected_feasible": expected_feasible,
                                "graph_build_time_sec": stats["graph_build_time_sec"],
                                "max_flow_time_sec": stats["max_flow_time_sec"],
                                "coverage_ratio": stats["max_flow_value"] / stats["total_demand"] if stats["total_demand"] > 0 else 0.0,
                            }
                            setup_running_time_list.append(this_round_stats)
                            # check if the "oec_task_scheduling_efficiency_results.csv" file exists, if not, create it and write the header
                            if not os.path.exists("oec_task_scheduling_efficiency_results.csv"):
                                df = pd.DataFrame(columns=setup_running_time_list[0].keys())
                                df.to_csv("oec_task_scheduling_efficiency_results.csv", index=False)
                                df = pd.DataFrame(setup_running_time_list)
                                df.to_csv("oec_task_scheduling_efficiency_results.csv", mode='a', header=False, index=False)
                            else:
                                df = pd.DataFrame([this_round_stats])
                                df.to_csv("oec_task_scheduling_efficiency_results.csv", mode='a', header=False, index=False)
                            pbar.update(1)
    pbar.close()

    # # save the setup_running_time_list to a csv file
    # df = pd.DataFrame(setup_running_time_list)
    # df.to_csv("oec_task_scheduling_efficiency_results.csv", index=False)
    # print("Efficiency test results saved to oec_task_scheduling_efficiency_results.csv")



