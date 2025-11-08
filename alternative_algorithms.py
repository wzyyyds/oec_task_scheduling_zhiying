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
        "coverage_ratio": maxflow_value / D_total if D_total > 0 else 1.0
    }
    if return_flow:
        result["flow"] = flow_dict
        result["graph"] = G
    return result

# ----------------------------
# Heuristic alternative algorithms
# ----------------------------

'''
Assign task to the satellite with most available energy in its visibility window.
'''
def heuristic_most_energy_first(
    tasks: List[Task],
    A: np.ndarray,          # (Nc, Ns, Nt) binary
    e_jk: np.ndarray,       # (Ns, Nt) energy in Joules
    psi: float,             # Joules per CPU cycle
    phi: float,             # cycles per second
    tau_b: float,           # seconds (converted battery capacity)
    slot_len: float,        # seconds per slot
    horizon_sec: float      # total horizon in seconds (Nt * slot_len)
) -> Dict[str, Any]:
    """
    A simple heuristic that assigns each task to the satellite with the most available energy
    in its visibility window.
    """
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
    # sort jobs by release time
    jobs.sort(key=lambda job: job.release)
    D_total = sum(job.demand for job in jobs)

    # Convert energy to time per slot
    print("Converting energy to time...")
    tau_in = convert_energy_to_time(e_jk, psi=psi, phi=phi)  # (Ns, Nt)

    # Heuristic assignment
    print("Running heuristic assignment...")
    total_assigned = 0.0
    sat_energy_dict = {}  # satellite_id -> remaining energy
    active_job_index_list = []  # list of indices of active jobs
    next_unreleased_job_idx = 0 # index of the next unreleased job in jobs list
    job_index_time_slot_available_satellite_dict = {}  # job_index -> time slot -> list of available satellites
    for time_slot_idx in range(Nt):
        # calculate the remaining energy for each satellite at this time slot
        for sat_idx in range(Ns):
            sat_energy_dict[sat_idx] = sat_energy_dict.get(sat_idx, 0.0) + tau_in[sat_idx, time_slot_idx]
        # check if any new jobs are released at this time slot and add them to the active job list
        slot_start, slot_end = slot_bounds(slot_len, time_slot_idx)
        while (next_unreleased_job_idx < len(jobs) and 
               jobs[next_unreleased_job_idx].release <= slot_start):
            active_job_index_list.append(next_unreleased_job_idx)
            # update the available satellite list for the new job
            for available_time_slot_idx in range(time_slot_idx, Nt):
                slot_start2, slot_end2 = slot_bounds(slot_len, available_time_slot_idx)
                # if the slot's end time exceeds the job's deadline, break
                if slot_end2 > jobs[next_unreleased_job_idx].deadline_abs:
                    break
                # get observed satellite ids
                observed_sat_ids = set()
                for sat_idx in range(Ns):
                    if sat_idx not in observed_sat_ids and A[jobs[next_unreleased_job_idx].task_id, sat_idx, available_time_slot_idx] == 1:
                        observed_sat_ids.add(sat_idx)
                    if sat_idx in observed_sat_ids:
                        if next_unreleased_job_idx not in job_index_time_slot_available_satellite_dict:
                            job_index_time_slot_available_satellite_dict[next_unreleased_job_idx] = {}
                        if available_time_slot_idx not in job_index_time_slot_available_satellite_dict[next_unreleased_job_idx]:
                            job_index_time_slot_available_satellite_dict[next_unreleased_job_idx][available_time_slot_idx] = []
                        job_index_time_slot_available_satellite_dict[next_unreleased_job_idx][available_time_slot_idx].append(sat_idx)
            next_unreleased_job_idx += 1
        # remove jobs that have already missed their deadlines
        active_job_index_list = [
            idx for idx in active_job_index_list
            if jobs[idx].deadline_abs > slot_start
        ]
        # sort the active jobs by their deadlines (earliest deadline first)
        active_job_index_list.sort(key=lambda idx: jobs[idx].deadline_abs)
        # try to assign each active job to the satellite with the most available energy
        for job_idx in active_job_index_list:
            job = jobs[job_idx]
            if job.demand <= 0:
                continue
            available_satellites = job_index_time_slot_available_satellite_dict.get(job_idx, {}).get(time_slot_idx, [])
            if not available_satellites:
                continue
            # find the satellite with the most available energy
            best_satellite = max(available_satellites, key=lambda sat_idx: sat_energy_dict.get(sat_idx, 0.0))
            available_energy = sat_energy_dict.get(best_satellite, 0.0)
            if available_energy <= 0:
                continue
            # assign as much as possible to this job
            assign_amount = min(job.demand, available_energy, slot_len)
            job.demand -= assign_amount
            sat_energy_dict[best_satellite] -= assign_amount
            total_assigned += assign_amount
        # remove completed jobs from the active job list
        active_job_index_list = [idx for idx in active_job_index_list if jobs[idx].demand > 0]
        # update the remaining energy for each satellite at the end of this time slot
        for sat_idx in range(Ns):
            sat_energy_dict[sat_idx] = min(sat_energy_dict[sat_idx], tau_b)  # battery capacity limit
    feasible = (total_assigned >= D_total - 1e-7)
    result = {
        "feasible": feasible,
        "max_flow_value": total_assigned,
        "total_demand": D_total,
        "num_jobs": len(jobs),
        "num_nodes": None,
        "num_edges": None,
        "coverage_ratio": total_assigned / D_total if D_total > 0 else 1.0
    }
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
    # res = heuristic_most_energy_first(
    #     tasks=tasks,
    #     A=A,
    #     e_jk=e_jk,
    #     psi=psi,
    #     phi=phi,
    #     tau_b=tau_b,
    #     slot_len=slot_len,
    #     horizon_sec=horizon_sec
    # )
    end_time = time.perf_counter()
    print(f"[{name}] feasible={res['feasible']}  (expected={expected})  "
          f"flow={res['max_flow_value']:.1f} / demand={res['total_demand']:.1f}  "
          f"coverage={res['coverage_ratio']:.4f}  "
          f"jobs={res['num_jobs']} nodes={res['num_nodes']} edges={res['num_edges']}  "
          f"time={end_time - start_time:.4f} sec")

def build_cases():
    psi, phi = 1.0, 1.0   # so tau = energy (J) = seconds
    tau_b = 0.0

    # ---------- Case 1 ----------
    slot_len = 10.0
    Nt = 3
    Ns = 1
    tasks = [
        Task(task_id=0, period=10.0, deadline=10.0, job_exec_time=10.0)
    ]
    A = np.ones((1, Ns, Nt), dtype=int)  # visible everywhere
    e_jk = np.zeros((Ns, Nt))
    e_jk[0,1] = 10.0
    e_jk[0,2] = 10.0
    yield ("Case 1 (Feasible - exact fit)", tasks, A, e_jk, slot_len, tau_b, psi, phi, True)

    # ---------- Case 2 ----------
    # Same as Case 1 but insufficient energy in eligible slots
    e_jk2 = np.zeros((Ns, Nt))
    e_jk2[0,1] = 5.0
    e_jk2[0,2] = 5.0
    yield ("Case 2 (Infeasible - not enough energy)", tasks, A, e_jk2, slot_len, tau_b, psi, phi, False)

    # ---------- Case 3 ----------
    slot_len = 10.0
    Nt = 4
    Ns = 1
    tasks3 = [
        Task(task_id=0, period=20.0, deadline=20.0, job_exec_time=15.0)
    ]  # releases at 20 (single job), window [20,40]
    A3 = np.zeros((1, Ns, Nt), dtype=int)
    A3[0,0,2] = 1
    A3[0,0,3] = 1
    e_jk3 = np.zeros((Ns, Nt))
    e_jk3[0,2] = 10.0
    e_jk3[0,3] = 10.0
    yield ("Case 3 (Feasible - split over two slots)", tasks3, A3, e_jk3, slot_len, tau_b, psi, phi, True)

    # ---------- Case 4 ----------
    slot_len = 10.0
    Nt = 3
    Ns = 1
    tasks4 = [
        Task(task_id=0, period=10.0, deadline=10.0, job_exec_time=10.0)
    ]
    A4 = np.zeros((1, Ns, Nt), dtype=int)  # only slot 0 visible (before release)
    A4[0,0,0] = 1
    e_jk4 = np.full((Ns, Nt), 10.0)        # plenty of energy
    yield ("Case 4 (Infeasible - no coverage in window)", tasks4, A4, e_jk4, slot_len, tau_b, psi, phi, False)

    # ---------- Case 5 ----------
    # Two tasks, two satellites, both jobs at t=10, each only visible on its own satellite in slot 1.
    slot_len = 10.0
    Nt = 2
    Ns = 2
    tasks5 = [
        Task(task_id=0, period=10.0, deadline=10.0, job_exec_time=10.0),
        Task(task_id=1, period=10.0, deadline=10.0, job_exec_time=10.0)
    ]
    A5 = np.zeros((2, Ns, Nt), dtype=int)
    A5[0,0,1] = 1  # task0 on sat0 at slot1
    A5[1,1,1] = 1  # task1 on sat1 at slot1
    e_jk5 = np.zeros((Ns, Nt))
    e_jk5[0,1] = 10.0
    e_jk5[1,1] = 10.0
    yield ("Case 5 (Feasible - two sats avoid contention)", tasks5, A5, e_jk5, slot_len, tau_b, psi, phi, True)

    # ---------- Case 6 ----------
    # Large feasible case (Nc=100, Ns=100, Nt=1000), each task has one job at t=500,
    Nc, Ns, Nt = 100, 100, 1000
    slot_len = 10.0
    horizon_sec = Nt * slot_len
    tasks = [
        Task(task_id=i, period=5000.0, deadline=10.0, job_exec_time=10.0)
        for i in range(Nc)
    ]
    A = np.zeros((Nc, Ns, Nt), dtype=int)
    for i in range(Nc):
        A[i, i, 500] = 1
    e_jk = np.zeros((Ns, Nt))
    for i in range(Ns):
        e_jk[i, 500] = 10.0
    psi, phi = 1.0, 1.0  # τ = e
    tau_b = 0.0
    expected_feasible = True
    yield ("Case 6 (Large feasible case)", tasks, A, e_jk, slot_len, tau_b, psi, phi, expected_feasible)

    # ---------- Case 7 ----------
    # Large infeasible case (Nc=100, Ns=100, Nt=1000), each task has one job at t=500,
    Nc, Ns, Nt = 100, 100, 1000
    slot_len = 10.0
    horizon_sec = Nt * slot_len
    tasks = [
        Task(task_id=i, period=5000.0, deadline=10.0, job_exec_time=10.0)
        for i in range(Nc)
    ]
    A = np.zeros((Nc, Ns, Nt), dtype=int)
    for i in range(Nc):
        A[i, i, 500] = 1
    e_jk = np.zeros((Ns, Nt))
    for i in range(Ns):
        e_jk[i, 500] = 5.0   # insufficient energy
    psi, phi = 1.0, 1.0  # τ = e
    tau_b = 0.0
    expected_feasible = False
    yield ("Case 7 (Large infeasible case)", tasks, A, e_jk, slot_len, tau_b, psi, phi, expected_feasible)

    # ---------- Super Large Case ----------
    # Nc=100, Ns=100, Nt=2000, each task has multiple jobs depends on the period, which is a randomly generated integer between 10 and 30
    Nc, Ns, Nt = 100, 100, 2000
    slot_len = 10.0
    horizon_sec = Nt * slot_len
    tasks = [
        Task(task_id=i, period=np.random.randint(10, 31) * 10.0, deadline=50.0, job_exec_time=10.0)
        for i in range(Nc)
    ]
    A = np.random.randint(0, 2, size=(Nc, Ns, Nt), dtype=int)
    e_jk = np.random.randint(0, 21, size=(Ns, Nt)).astype(float)  # energy between 0 and 20 Joules
    psi, phi = 1.0, 1.0  # τ = e
    tau_b = 20.0
    expected_feasible = None  # unknown
    yield ("Super Large Case", tasks, A, e_jk, slot_len, tau_b, psi, phi, expected_feasible)

def import_case_from_file(test_case_json_file_path: str) -> Tuple[str, List[Task], np.ndarray, np.ndarray, float, float, float, float, bool]:
    """
    Import a test case from a JSON file.
    The JSON file should contain all necessary parameters to build the test case.
    """
    import json

    with open(test_case_json_file_path, 'r') as f:
        data = json.load(f)

    name = data['name']
    task_info_dict = data['task_info_dict']
    tasks = []
    for task_id_str, td in task_info_dict.items():
        task_id = int(task_id_str)
        tasks.append(Task(
            task_id=task_id,
            period=td['period'],
            deadline=td['deadline'],
            job_exec_time=td['job_exec_time'],
            offset=td.get('offset', 0.0)
        ))
    A = np.array(data['A'], dtype=int)
    e_jk = np.array(data['e_jk'], dtype=float)
    slot_len = data['slot_len']
    tau_b = data['tau_b']
    psi = data['psi']
    phi = data['phi']
    expected = data.get('expected', None)

    return name, tasks, A, e_jk, slot_len, tau_b, psi, phi, expected

if __name__ == "__main__":
    np.random.seed(42)
    name, tasks, A, e_jk, slot_len, tau_b, psi, phi, expected = import_case_from_file("simulation_data/test_case.json")
    print(tasks)
    run_case(name, tasks, A, e_jk, slot_len, tau_b, psi, phi, expected)
    # for case in build_cases():
    #     run_case(*case)