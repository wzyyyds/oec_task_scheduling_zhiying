#!/usr/bin/env python3
"""
Max-flow–based scheduling feasibility test for periodic EO tasks over a satellite constellation.

Model summary (discrete time):
- Time is slotted: slots k = 0..Nt-1, each of length slot_len (seconds).
- Tasks i = 0..Nc-1 with period T_i (sec), relative deadline D_i (sec), and per-job execution demand c_i (cycles).
  Jobs of task i release at T_i, 2T_i, ... within [0, horizon_sec), each due by (release + D_i).
- Coverage matrix A[i, j, k] in {0,1} indicates if satellite j can serve task i during slot k.
- CPU rate is phi (cycles/sec), so each job requires p_i = c_i / phi seconds of processing.
- Satellite j has input energy e_jk (Joules) for slot k. Convert to time τ_{j,k} = e_jk / (psi * phi) (seconds).
- Battery carryover: energy node (j,k) -> (j,k-1) with capacity τ_b (seconds) if k >= 1.

Graph:
- Source -> Task_i edge cap = total demand of all Task_i jobs (sum of p_i over all releases).
- Task_i -> each of its Job nodes edge cap = p_i (job's demanded execution time in seconds).
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
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import importlib
import math
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
    job_exec_cycles: float # CPU cycles required per job (c_i)
    # Optional: allow a start offset if needed in the future (default releases at T_i, 2T_i, ...)
    offset: float = 0.0    # seconds; by default keep 0.0 so first release is at T_i (per spec)


@dataclass
class Job:
    """A single job (release) of a task."""
    job_id: str           # unique label, e.g., f"J_task{task_id}_r{r_idx}"
    task_id: int
    release: float        # absolute release time (sec)
    deadline_abs: float   # absolute deadline time (sec) = release + D_i
    demand: float         # seconds required = c_i / phi


# ----------------------------
# Helpers
# ----------------------------

FLOW_ALGORITHMS = {
    "edmonds_karp": nx.algorithms.flow.edmonds_karp,
    "preflow_push": nx.algorithms.flow.preflow_push,
}

NUMERIC_FLOOR = 1e-30
RELATIVE_EPSILON = 1e-12

def generate_jobs(tasks: List[Task], horizon_sec: float, phi: float) -> List[Job]:
    """
    Generate all jobs over [0, horizon_sec).
    Per problem statement: releases at T_i, 2T_i, ... (NOT at 0).
    """
    if phi <= 0:
        raise ValueError("phi must be positive.")
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
                demand=t.job_exec_cycles / phi
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


def compute_numeric_epsilon(*values: Any) -> float:
    minima = []
    maxima = []
    for value in values:
        arr = np.asarray(value, dtype=float)
        finite = np.abs(arr[np.isfinite(arr)])
        positive = finite[finite > 0.0]
        if positive.size == 0:
            continue
        minima.append(float(np.min(positive)))
        maxima.append(float(np.max(positive)))

    if not maxima:
        return NUMERIC_FLOOR

    scale = max(maxima)
    smallest = min(minima)
    return max(NUMERIC_FLOOR, min(scale * RELATIVE_EPSILON, smallest * 1e-3))


def resolve_flow_func(flow_algorithm: str):
    if flow_algorithm not in FLOW_ALGORITHMS:
        supported = ", ".join(sorted(list(FLOW_ALGORITHMS) + ["ford_fulkerson"]))
        raise ValueError(f"Unsupported flow algorithm '{flow_algorithm}'. Supported: {supported}")
    return FLOW_ALGORITHMS[flow_algorithm]


def _build_capacity_maps(G: nx.DiGraph) -> Tuple[Dict[Any, Dict[Any, float]], Dict[Any, Dict[Any, float]]]:
    capacities: Dict[Any, Dict[Any, float]] = {}
    residual: Dict[Any, Dict[Any, float]] = {}

    for u, v, data in G.edges(data=True):
        cap = float(data.get("capacity", 0.0))
        capacities.setdefault(u, {})[v] = cap
        residual.setdefault(u, {})
        residual.setdefault(v, {})
        residual[u].setdefault(v, 0.0)
        residual[v].setdefault(u, 0.0)
        residual[u][v] += cap

    return capacities, residual


def _dfs_augmenting_path(
    residual: Dict[Any, Dict[Any, float]],
    source: Any,
    sink: Any,
    visited: set,
    path: List[Tuple[Any, Any]],
    eps: float = 1e-12,
) -> bool:
    if source == sink:
        return True

    visited.add(source)
    for nxt, capacity in residual.get(source, {}).items():
        if capacity <= eps or nxt in visited:
            continue
        path.append((source, nxt))
        if _dfs_augmenting_path(residual, nxt, sink, visited, path, eps=eps):
            return True
        path.pop()
    return False


def naive_ford_fulkerson(
    G: nx.DiGraph,
    source: Any,
    sink: Any,
) -> Tuple[float, Dict[Any, Dict[Any, float]]]:
    """
    Naive Ford-Fulkerson using DFS to choose an arbitrary augmenting path.
    Capacities may be floats; we stop when no residual path with capacity > eps exists.
    """
    capacities, residual = _build_capacity_maps(G)
    eps = 1e-12
    max_flow_value = 0.0

    while True:
        path: List[Tuple[Any, Any]] = []
        found = _dfs_augmenting_path(residual, source, sink, visited=set(), path=path, eps=eps)
        if not found:
            break

        bottleneck = min(residual[u][v] for u, v in path)
        max_flow_value += bottleneck

        for u, v in path:
            residual[u][v] -= bottleneck
            residual[v][u] += bottleneck

    flow_dict: Dict[Any, Dict[Any, float]] = {}
    for u, nbrs in capacities.items():
        row: Dict[Any, float] = {}
        for v, cap in nbrs.items():
            flow = cap - residual[u].get(v, 0.0)
            if flow > eps:
                row[v] = flow
        flow_dict[u] = row
    return max_flow_value, flow_dict


def solve_max_flow(
    G: nx.DiGraph,
    source: Any,
    sink: Any,
    flow_algorithm: str,
) -> Tuple[float, Dict[Any, Dict[Any, float]]]:
    if flow_algorithm == "ford_fulkerson":
        return naive_ford_fulkerson(G, source, sink)

    flow_func = resolve_flow_func(flow_algorithm)
    return nx.maximum_flow(G, source, sink, flow_func=flow_func)


def _load_pulp():
    pulp_spec = importlib.util.find_spec("pulp")
    if pulp_spec is None:
        raise ImportError("PuLP is not installed in the current environment.")
    return importlib.import_module("pulp")


# ----------------------------
# Graph construction
# ----------------------------

def build_scheduling_graph(
    tasks: List[Task],
    jobs: List[Job],
    A: np.ndarray,             # shape (Nc, Ns, Nt), entries in {0,1}
    tau_in: np.ndarray,        # shape (Ns, Nt), seconds from input energy for slot k
    tau_b: float,              # battery capacity in seconds (converted time)
    slot_len: float,           # seconds per time slot
    eps: float,
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

    def job_slot_node(job: Job, k: int) -> str:
        return f"JOBSLOT_{job.job_id}_{k}"

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
        if demand_sum > eps:
            G.add_edge(s, task_node(i), capacity=demand_sum)
        for j in task_jobs:
            G.add_node(job_node(j))
            if j.demand > eps:
                G.add_edge(task_node(i), job_node(j), capacity=j.demand)
        D_total += demand_sum

    # 3) Job -> JobSlot -> SatelliteTime edges
    #    Job -> JobSlot capacity = slot_len, which limits each job to at most one slot worth
    #    of execution in a given time slot. This matches the heuristic/MILP convention that a
    #    job cannot consume more than one slot's capacity during the same slot.
    for job in jobs:
        i = job.task_id
        for k in range(Nt):
            slot_start, slot_end = slot_bounds(slot_len, k)
            if slot_start < job.release or slot_end > job.deadline_abs:
                continue

            job_slot = job_slot_node(job, k)
            G.add_node(job_slot)
            G.add_edge(job_node(job), job_slot, capacity=slot_len)

            for j in range(Ns):
                if A[i, j, k] != 1:
                    continue
                G.add_node(st_node(j, k))
                G.add_edge(job_slot, st_node(j, k), capacity=slot_len)

    # 4) SatelliteTime(j,k) -> Energy(j,k) with capacity = slot_len
    # 5) Energy(j,k) -> Sink with capacity = tau_in[j,k]
    # 6) Energy(j,k) -> Energy(j,k-1) with capacity = tau_b, for k >= 1
    for j in range(Ns):
        for k in range(Nt):
            st = st_node(j, k)
            en = en_node(j, k)
            # It's okay to add nodes/edges even if they aren't used by any job
            G.add_node(st) # TODO: only add when tau > 0, otherwise if st not exist, continue
            G.add_node(en)
            # ST -> EN
            G.add_edge(st, en, capacity=slot_len)
            # EN -> Sink
            tau = float(tau_in[j, k])
            if tau < 0:
                raise ValueError("tau_in must be non-negative.")
            if tau > eps:
                G.add_edge(en, t, capacity=tau)
            # EN(k) -> EN(k-1) battery carryover
            if k >= 1 and tau_b > eps:
                G.add_edge(en, en_node(j, k - 1), capacity=tau_b)

    return G, s, t, D_total


# ----------------------------
# Feasibility test
# ----------------------------

def feasibility_test(
    tasks: List[Task],
    A: np.ndarray,
    e_jk: np.ndarray,
    psi: float,
    phi: float,
    tau_b: float,
    slot_len: float,
    horizon_sec: float,
    flow_tolerance: float = None,
    return_flow: bool = False,
    debug: bool = False,
    debug_prefix: str = "",
    flow_algorithm: str = "edmonds_karp"
) -> Dict[str, Any]:
    """
    Build the network and run the selected max-flow algorithm.
    """
    Nc, Ns, Nt = A.shape
    assert horizon_sec > 0 and math.isclose(horizon_sec, Nt * slot_len, rel_tol=0, abs_tol=1e-6)

    if debug:
        print("Generating jobs...")
    jobs = generate_jobs(tasks, horizon_sec=horizon_sec, phi=phi)

    if debug:
        with open(f"{debug_prefix}job_demand_dict.json", "w") as dump_f:
            job_demand_dict = {
                idx: {
                    "demand": job.demand,
                    "release_time": job.release,
                    "deadline": job.deadline_abs
                }
                for idx, job in enumerate(jobs)
            }
            json.dump(job_demand_dict, dump_f, indent=4)

    if debug:
        print("Converting energy to time...")
    tau_in = convert_energy_to_time(e_jk, psi=psi, phi=phi)
    eps = compute_numeric_epsilon(
        [job.demand for job in jobs],
        tau_in,
        tau_b,
        slot_len,
    )
    if flow_tolerance is None:
        flow_tolerance = eps

    if debug:
        with open(f"{debug_prefix}tau_in.json", "w") as dump_f:
            json.dump(tau_in.tolist(), dump_f, indent=4)

    if debug:
        print("Building scheduling graph...")
    start_graph_time = time.perf_counter()
    G, s, t, D_total = build_scheduling_graph(tasks, jobs, A, tau_in, tau_b, slot_len, eps=eps)
    end_graph_time = time.perf_counter()
    if debug:
        print(f"Graph built in {end_graph_time - start_graph_time:.4f} sec: "
              f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
        nx.write_edgelist(G, f"{debug_prefix}edgelist.txt")

    if debug:
        print(f"Running max-flow ({flow_algorithm})...")
    start_flow_time = time.perf_counter()
    maxflow_value, flow_dict = solve_max_flow(G, s, t, flow_algorithm)
    end_flow_time = time.perf_counter()

    if debug:
        print(f"Max-flow computed in {end_flow_time - start_flow_time:.4f} sec.")
        with open(f"{debug_prefix}flow_dict.json", "w") as dump_f:
            json.dump(flow_dict, dump_f, indent=4)

    feasible = (maxflow_value + flow_tolerance >= D_total)
    completed_jobs = 0
    for job in jobs:
        job_node = f"JOB_{job.job_id}"
        assigned = sum(flow_dict.get(job_node, {}).values())
        if assigned + flow_tolerance >= job.demand:
            completed_jobs += 1
    result = {
        "feasible": feasible,
        "max_flow_value": maxflow_value,
        "total_demand": D_total,
        "num_jobs": len(jobs),
        "completed_jobs": completed_jobs,
        "completed_job_ratio": completed_jobs / len(jobs) if jobs else 1.0,
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "coverage_ratio": maxflow_value / D_total if D_total > 0 else 1.0,
        "flow_algorithm": flow_algorithm,
        "graph_build_time_sec": end_graph_time - start_graph_time,
        "max_flow_time_sec": end_flow_time - start_flow_time,
    }
    if return_flow:
        result["flow"] = flow_dict
        result["graph"] = G
    return result


# ----------------------------
# Heuristic alternative algorithms
# ----------------------------

"""
Energy-first EDF:
- jobs are still processed in EDF order within each slot
- for each job, choose the currently visible satellite with the most remaining energy
"""
def heuristic_most_energy_first(
    tasks: List[Task],
    A: np.ndarray,
    e_jk: np.ndarray,
    psi: float,
    phi: float,
    tau_b: float,
    slot_len: float,
    horizon_sec: float,
    debug: bool = False,
) -> Dict[str, Any]:
    Nc, Ns, Nt = A.shape
    assert horizon_sec > 0 and math.isclose(horizon_sec, Nt * slot_len, rel_tol=0, abs_tol=1e-6)

    if debug:
        print("Generating jobs...")
    jobs = generate_jobs(tasks, horizon_sec=horizon_sec, phi=phi)
    jobs.sort(key=lambda job: job.release)
    D_total = sum(job.demand for job in jobs)

    if debug:
        print("Converting energy to time...")
    tau_in = convert_energy_to_time(e_jk, psi=psi, phi=phi)
    eps = compute_numeric_epsilon(
        [job.demand for job in jobs],
        tau_in,
        tau_b,
        slot_len,
    )

    if debug:
        print("Running heuristic assignment...")

    finished_jobs_list = set()
    total_assigned = 0.0
    sat_energy_dict = {}
    active_job_index_list = []
    next_unreleased_job_idx = 0

    if debug:
        with open("heur_job_demand_dict.json", "w") as dump_f:
            json.dump({idx: job.demand for idx, job in enumerate(jobs)}, dump_f, indent=4)

    for time_slot_idx in range(Nt):
        # accumulate harvested energy
        for sat_idx in range(Ns):
            sat_energy_dict[sat_idx] = sat_energy_dict.get(sat_idx, 0.0) + tau_in[sat_idx, time_slot_idx]
        sat_slot_remaining = {sat_idx: slot_len for sat_idx in range(Ns)}

        slot_start, slot_end = slot_bounds(slot_len, time_slot_idx)

        # release new jobs
        while next_unreleased_job_idx < len(jobs) and jobs[next_unreleased_job_idx].release <= slot_start:
            active_job_index_list.append(next_unreleased_job_idx)
            next_unreleased_job_idx += 1

        # drop expired jobs: a slot is usable only if it fully fits before the deadline
        active_job_index_list = [
            idx for idx in active_job_index_list
            if jobs[idx].deadline_abs >= slot_end and jobs[idx].demand > eps
        ]

        # EDF
        active_job_index_list.sort(key=lambda idx: jobs[idx].deadline_abs)

        # assign within this slot
        for jidx in active_job_index_list:
            job = jobs[jidx]
            if job.demand <= eps:
                continue

            available_sats = [
                sat_idx for sat_idx in range(Ns)
                if A[job.task_id, sat_idx, time_slot_idx] == 1
                and sat_energy_dict.get(sat_idx, 0.0) > eps
                and sat_slot_remaining[sat_idx] > eps
            ]
            if not available_sats:
                continue

            # Energy-first: among visible satellites in this slot, pick the most charged one.
            sat = max(available_sats, key=lambda sidx: sat_energy_dict.get(sidx, 0.0))
            avail_E = sat_energy_dict.get(sat, 0.0)
            if avail_E <= eps:
                continue

            assign = min(job.demand, avail_E, sat_slot_remaining[sat])
            if assign <= eps:
                continue

            job.demand -= assign
            sat_energy_dict[sat] = avail_E - assign
            sat_slot_remaining[sat] -= assign
            total_assigned += assign

            if debug:
                print(f"[slot {time_slot_idx}] job {jidx} -> sat {sat}, "
                      f"assign={assign:.4f}, remain_job={job.demand:.4f}, "
                      f"remain_satE={sat_energy_dict[sat]:.4f}")

            if job.demand <= eps:
                finished_jobs_list.add(jidx)

        # 清掉完成的
        active_job_index_list = [idx for idx in active_job_index_list if jobs[idx].demand > eps]

        # 电池上限
        for sat_idx in range(Ns):
            sat_energy_dict[sat_idx] = min(sat_energy_dict[sat_idx], tau_b)

    feasible = (total_assigned >= D_total - eps)

    if debug:
        for jidx, job in enumerate(jobs):
            if jidx not in finished_jobs_list and job.demand > eps:
                print(f"Unfinished job {jidx} (task {job.task_id}), remaining {job.demand}")

    return {
        "feasible": feasible,
        "max_flow_value": total_assigned,
        "total_demand": D_total,
        "num_jobs": len(jobs),
        "completed_jobs": len(finished_jobs_list),
        "completed_job_ratio": len(finished_jobs_list) / len(jobs) if jobs else 1.0,
        "num_nodes": None,
        "num_edges": None,
        "coverage_ratio": total_assigned / D_total if D_total > 0 else 1.0
    }


def heuristic_random_assignment(
    tasks: List[Task],
    A: np.ndarray,
    e_jk: np.ndarray,
    psi: float,
    phi: float,
    tau_b: float,
    slot_len: float,
    horizon_sec: float,
    random_seed: int = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    随机 baseline：
    - 按时隙推进
    - 每个时隙：随机打乱活动 job 顺序
    - 对每个 job：先筛出当前时隙可见的卫星，再随机选一个
    - 若所选卫星没能量 / 没有剩余时隙时间，则本次尝试失败，不重试
    """
    Nc, Ns, Nt = A.shape
    assert horizon_sec > 0 and math.isclose(horizon_sec, Nt * slot_len, rel_tol=0, abs_tol=1e-6)

    # jobs & 总需求
    jobs = generate_jobs(tasks, horizon_sec=horizon_sec, phi=phi)
    jobs.sort(key=lambda job: job.release)
    D_total = sum(j.demand for j in jobs)
    if D_total <= 0:
        return {
            "feasible": True,
            "max_flow_value": 0.0,
            "total_demand": 0.0,
            "coverage_ratio": 1.0,
            "num_jobs": len(jobs),
            "num_nodes": None,
            "num_edges": None,
        }

    # 每时隙输入能量 -> 时间
    tau_in = convert_energy_to_time(e_jk, psi=psi, phi=phi)
    eps = compute_numeric_epsilon(
        remaining := [job.demand for job in jobs],
        tau_in,
        tau_b,
        slot_len,
    )

    # 状态
    sat_energy = np.zeros(Ns, dtype=float)
    active_jobs: List[int] = []
    next_job_idx = 0
    total_assigned = 0.0

    rng = np.random.default_rng(random_seed)

    for k in range(Nt):
        slot_start, slot_end = slot_bounds(slot_len, k)
        sat_slot_remaining = np.full(Ns, slot_len, dtype=float)

        # 累加本时隙 harvested 能量 + 电池上限
        sat_energy += tau_in[:, k]

        # 释放新 job
        while next_job_idx < len(jobs) and jobs[next_job_idx].release <= slot_start:
            active_jobs.append(next_job_idx)
            next_job_idx += 1

        # 去掉过期/已完成 job
        active_jobs = [
            jidx for jidx in active_jobs
            if (jobs[jidx].deadline_abs >= slot_end and remaining[jidx] > eps)
        ]
        if not active_jobs:
            continue

        # 随机打乱 job 顺序
        rng.shuffle(active_jobs)

        # 遍历 job，尝试在当前 slot 执行
        for jidx in active_jobs:
            if remaining[jidx] <= eps:
                continue
            task_id = jobs[jidx].task_id

            visible_sats = [s for s in range(Ns) if A[task_id, s, k] == 1]
            if not visible_sats:
                continue

            sat = int(rng.choice(visible_sats))
            assign = min(remaining[jidx], sat_energy[sat], sat_slot_remaining[sat])
            if assign <= eps:
                continue

            remaining[jidx] -= assign
            sat_energy[sat] -= assign
            sat_slot_remaining[sat] -= assign
            total_assigned += assign

            if debug:
                print(f"[rand][slot {k}] job {jidx} -> sat {sat}, assign={assign:.4f}, "
                      f"remain_job={remaining[jidx]:.4f}, remain_sat={sat_energy[sat]:.4f}")

        sat_energy = np.minimum(sat_energy, tau_b)

    feasible = (total_assigned >= D_total - eps)
    completed_jobs = sum(1 for rem in remaining if rem <= eps)

    return {
        "feasible": feasible,
        "max_flow_value": total_assigned,
        "total_demand": D_total,
        "coverage_ratio": total_assigned / D_total,
        "num_jobs": len(jobs),
        "completed_jobs": completed_jobs,
        "completed_job_ratio": completed_jobs / len(jobs) if jobs else 1.0,
        "num_nodes": None,
        "num_edges": None,
    }



def milp_small_instance(
    tasks: List[Task],
    A: np.ndarray,
    e_jk: np.ndarray,
    psi: float,
    phi: float,
    tau_b: float,
    slot_len: float,
    horizon_sec: float,
    time_limit_sec: int = 300,
    max_feasible_assignments: int = 25000,
    objective_mode: str = "throughput",
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Small-instance MILP baseline solved with PuLP/CBC.
    """
    pulp = _load_pulp()

    Nc, Ns, Nt = A.shape
    assert horizon_sec > 0 and math.isclose(horizon_sec, Nt * slot_len, rel_tol=0, abs_tol=1e-6)

    jobs = generate_jobs(tasks, horizon_sec=horizon_sec, phi=phi)
    D_total = sum(job.demand for job in jobs)
    if D_total <= 0:
        return {
            "feasible": True,
            "max_flow_value": 0.0,
            "total_demand": 0.0,
            "coverage_ratio": 1.0,
            "num_jobs": len(jobs),
            "num_nodes": None,
            "num_edges": None,
            "solver_status": "trivial",
            "objective_mode": objective_mode,
        }

    tau_in = convert_energy_to_time(e_jk, psi=psi, phi=phi)
    eps = compute_numeric_epsilon(
        [job.demand for job in jobs],
        tau_in,
        tau_b,
        slot_len,
    )

    feasible_assignments = []
    feasible_by_job = {}
    feasible_by_job_slot = {}
    feasible_by_sat_slot = {}
    for jidx, job in enumerate(jobs):
        for sat_idx in range(Ns):
            for slot_idx in range(Nt):
                slot_start, slot_end = slot_bounds(slot_len, slot_idx)
                if A[job.task_id, sat_idx, slot_idx] != 1:
                    continue
                if slot_start < job.release or slot_end > job.deadline_abs:
                    continue
                key = (jidx, sat_idx, slot_idx)
                feasible_assignments.append(key)
                feasible_by_job.setdefault(jidx, []).append(key)
                feasible_by_job_slot.setdefault((jidx, slot_idx), []).append(key)
                feasible_by_sat_slot.setdefault((sat_idx, slot_idx), []).append(key)

    if len(feasible_assignments) > max_feasible_assignments:
        return {
            "feasible": False,
            "max_flow_value": 0.0,
            "total_demand": D_total,
            "coverage_ratio": 0.0,
            "num_jobs": len(jobs),
            "num_nodes": None,
            "num_edges": None,
            "solver_status": "skipped_too_large",
            "num_candidate_assignments": len(feasible_assignments),
            "objective_mode": objective_mode,
        }

    start_solve_time = time.perf_counter()
    prob = pulp.LpProblem("satellite_scheduling_small_milp", pulp.LpMaximize)

    y = {
        key: pulp.LpVariable(f"y_{key[0]}_{key[1]}_{key[2]}", lowBound=0.0, upBound=slot_len, cat="Continuous")
        for key in feasible_assignments
    }
    x = {
        key: pulp.LpVariable(f"x_{key[0]}_{key[1]}_{key[2]}", lowBound=0, upBound=1, cat="Binary")
        for key in feasible_assignments
    }
    z = {
        jidx: pulp.LpVariable(f"z_{jidx}", lowBound=0, upBound=1, cat="Binary")
        for jidx in range(len(jobs))
    }
    battery = {
        (sat_idx, slot_idx): pulp.LpVariable(
            f"b_{sat_idx}_{slot_idx}", lowBound=0.0, upBound=tau_b, cat="Continuous"
        )
        for sat_idx in range(Ns) for slot_idx in range(Nt)
    }
    spill = {
        (sat_idx, slot_idx): pulp.LpVariable(f"spill_{sat_idx}_{slot_idx}", lowBound=0.0, cat="Continuous")
        for sat_idx in range(Ns) for slot_idx in range(Nt)
    }

    total_assigned = pulp.lpSum(y.values())
    total_completed = pulp.lpSum(z.values())
    if objective_mode == "throughput":
        prob += total_assigned
    elif objective_mode == "completed_jobs":
        prob += total_completed
    else:
        raise ValueError(f"Unknown MILP objective_mode: {objective_mode}")

    for key in feasible_assignments:
        prob += y[key] <= slot_len * x[key]

    for jidx, job in enumerate(jobs):
        assigned_to_job = pulp.lpSum(y[key] for key in feasible_by_job.get(jidx, []))
        prob += assigned_to_job <= job.demand
        prob += assigned_to_job >= job.demand * z[jidx]

    for sat_idx in range(Ns):
        for slot_idx in range(Nt):
            sat_slot_keys = feasible_by_sat_slot.get((sat_idx, slot_idx), [])
            slot_usage = pulp.lpSum(y[key] for key in sat_slot_keys)
            prob += slot_usage <= slot_len

            if slot_idx == 0:
                prob += battery[(sat_idx, slot_idx)] == (
                    tau_in[sat_idx, slot_idx] - slot_usage - spill[(sat_idx, slot_idx)]
                )
            else:
                prob += battery[(sat_idx, slot_idx)] == (
                    battery[(sat_idx, slot_idx - 1)]
                    + tau_in[sat_idx, slot_idx]
                    - slot_usage
                    - spill[(sat_idx, slot_idx)]
                )

    solver = pulp.PULP_CBC_CMD(msg=debug, timeLimit=time_limit_sec)
    prob.solve(solver)
    end_solve_time = time.perf_counter()

    status = pulp.LpStatus[prob.status]
    assigned = sum(float(var.value() or 0.0) for var in y.values())
    completed_jobs = 0
    for jidx, job in enumerate(jobs):
        assigned_to_job = sum(
            float(y[key].value() or 0.0)
            for key in feasible_by_job.get(jidx, [])
        )
        if assigned_to_job + eps >= job.demand:
            completed_jobs += 1

    return {
        "feasible": assigned >= D_total - eps and status in {"Optimal", "Integer Feasible"},
        "max_flow_value": assigned,
        "total_demand": D_total,
        "coverage_ratio": assigned / D_total if D_total > 0 else 1.0,
        "num_jobs": len(jobs),
        "completed_jobs": completed_jobs,
        "completed_job_ratio": completed_jobs / len(jobs) if jobs else 1.0,
        "num_nodes": None,
        "num_edges": None,
        "solver_status": status,
        "solve_time_sec": end_solve_time - start_solve_time,
        "num_candidate_assignments": len(feasible_assignments),
        "objective_mode": objective_mode,
    }


def heuristic_edf(
    tasks: List[Task],
    A: np.ndarray,
    e_jk: np.ndarray,
    psi: float,
    phi: float,
    tau_b: float,
    slot_len: float,
    horizon_sec: float,
    random_seed: int = 42,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Deadline EDF baseline:
    - 时隙推进
    - 每个时隙：活动 job 按绝对截止时间从早到晚排序（EDF）
    - 对每个 job：在当前时隙可见 && 有能量的卫星中，随机选一个可用卫星
    - 不使用能量优先规则，作为更纯粹的 deadline-driven baseline
    """
    Nc, Ns, Nt = A.shape
    assert horizon_sec > 0 and math.isclose(horizon_sec, Nt * slot_len, rel_tol=0, abs_tol=1e-6)

    jobs = generate_jobs(tasks, horizon_sec=horizon_sec, phi=phi)
    jobs.sort(key=lambda job: job.release)
    D_total = sum(j.demand for j in jobs)
    if D_total <= 0:
        return {
            "feasible": True,
            "max_flow_value": 0.0,
            "total_demand": 0.0,
            "coverage_ratio": 1.0,
            "num_jobs": len(jobs),
            "num_nodes": None,
            "num_edges": None,
        }

    tau_in = convert_energy_to_time(e_jk, psi=psi, phi=phi)
    remaining = [job.demand for job in jobs]
    eps = compute_numeric_epsilon(
        remaining,
        tau_in,
        tau_b,
        slot_len,
    )
    sat_energy = np.zeros(Ns, dtype=float)
    active_jobs: List[int] = []
    next_job_idx = 0
    total_assigned = 0.0
    rng = np.random.default_rng(random_seed)

    for k in range(Nt):
        slot_start, slot_end = slot_bounds(slot_len, k)
        sat_slot_remaining = np.full(Ns, slot_len, dtype=float)

        # 收能量 + 电池上限
        sat_energy += tau_in[:, k]

        # 新 job 到达
        while next_job_idx < len(jobs) and jobs[next_job_idx].release <= slot_start:
            active_jobs.append(next_job_idx)
            next_job_idx += 1

        # 丢掉过期/完成
        active_jobs = [
            jidx for jidx in active_jobs
            if (jobs[jidx].deadline_abs >= slot_end and remaining[jidx] > eps)
        ]
        if not active_jobs:
            continue

        # EDF 排序
        active_jobs.sort(key=lambda jidx: jobs[jidx].deadline_abs)

        # 当前 slot 分配
        for jidx in active_jobs:
            if remaining[jidx] <= eps:
                continue
            task_id = jobs[jidx].task_id

            candidates = [
                s for s in range(Ns)
                if A[task_id, s, k] == 1 and sat_energy[s] > eps and sat_slot_remaining[s] > eps
            ]
            if not candidates:
                continue

            selected_sat = int(rng.choice(candidates))
            assign = min(remaining[jidx], sat_energy[selected_sat], sat_slot_remaining[selected_sat])
            if assign <= eps:
                continue

            remaining[jidx] -= assign
            sat_energy[selected_sat] -= assign
            sat_slot_remaining[selected_sat] -= assign
            total_assigned += assign

            if debug:
                print(f"[edf][slot {k}] job {jidx} -> sat {selected_sat}, assign={assign:.4f}, "
                      f"remain_job={remaining[jidx]:.4f}, remain_sat={sat_energy[selected_sat]:.4f}")

        sat_energy = np.minimum(sat_energy, tau_b)

    feasible = (total_assigned >= D_total - eps)
    completed_jobs = sum(1 for rem in remaining if rem <= eps)

    return {
        "feasible": feasible,
        "max_flow_value": total_assigned,
        "total_demand": D_total,
        "coverage_ratio": total_assigned / D_total,
        "num_jobs": len(jobs),
        "completed_jobs": completed_jobs,
        "completed_job_ratio": completed_jobs / len(jobs) if jobs else 1.0,
        "num_nodes": None,
        "num_edges": None,
    }




def run_case(name, tasks, A, e_jk, slot_len, tau_b, psi, phi, expected):
    Nt = A.shape[2]
    horizon_sec = Nt * slot_len
    start_time = time.perf_counter()
    # res = feasibility_test(
    #     tasks=tasks,
    #     A=A,
    #     e_jk=e_jk,
    #     psi=psi,
    #     phi=phi,
    #     tau_b=tau_b,
    #     slot_len=slot_len,
    #     horizon_sec=horizon_sec,
    #     return_flow=False
    # )
    res = heuristic_most_energy_first(
        tasks=tasks,
        A=A,
        e_jk=e_jk,
        psi=psi,
        phi=phi,
        tau_b=tau_b,
        slot_len=slot_len,
        horizon_sec=horizon_sec
    )
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
        Task(task_id=0, period=10.0, deadline=10.0, job_exec_cycles=10.0)
    ]
    A = np.ones((1, Ns, Nt), dtype=int)  # visible everywhere
    e_jk = np.zeros((Ns, Nt))
    e_jk[0,1] = 10.0
    e_jk[0,2] = 10.0
    yield ("Case 1 (Feasible - exact fit)", tasks, A, e_jk, slot_len, tau_b, psi, phi, True)


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
    phi = data['phi']
    tasks = []
    for task_id_str, td in task_info_dict.items():
        task_id = int(task_id_str)
        job_exec_cycles = td.get('job_exec_cycles')
        if job_exec_cycles is None:
            job_exec_cycles = td['job_exec_time'] * phi
        tasks.append(Task(
            task_id=task_id,
            period=td['period'],
            deadline=td['deadline'],
            job_exec_cycles=job_exec_cycles,
            offset=td.get('offset', 0.0)
        ))
    A = np.array(data['A'], dtype=int)
    e_jk = np.array(data['e_jk'], dtype=float)
    slot_len = data['slot_len']
    tau_b = data['tau_b']
    psi = data['psi']
    expected = data.get('expected', None)

    return name, tasks, A, e_jk, slot_len, tau_b, psi, phi, expected

if __name__ == "__main__":
    np.random.seed(42)
    name, tasks, A, e_jk, slot_len, tau_b, psi, phi, expected = import_case_from_file("simulation_data/test_case.json")
    print(tasks)
    run_case(name, tasks, A, e_jk, slot_len, tau_b, psi, phi, expected)
    # for case in build_cases():
    #     run_case(*case)
