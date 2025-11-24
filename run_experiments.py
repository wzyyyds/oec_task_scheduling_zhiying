#!/usr/bin/env python3
import os
import copy
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import simulation_data.build_testcase as bt
import alternative_algorithms as alt

# ---------- Paths ----------
ACCESS_REPORT_PATH = os.path.join("simulation_data", "parsed_access.json")
ENERGY_REPORT_PATH = os.path.join("simulation_data", "solar_parsed.json")

FEAS_CSV = "results_feasibility.csv"
HEUR_CSV = "results_heuristic.csv"

# ---------- Baseline config ----------
BASE_CONFIG = {
    "energy_per_cpu_cycle": 1e-9,          # psi
    "cpu_cycles_per_second": 1e10,         # phi
    "battery_capacity_seconds": 7200,      # tau_b (seconds of compute)
    "satellite_solar_panel_power_watts": 40.0,
    "time_slot_length": 60,                # slot_len
    "task_average_period": 300,
    "task_average_deadline": 300,
    "task_average_execution": 50,
}


def iter_configs():
    """
    生成所有要测试的配置：
    - baseline
    - 分别对 BASE_CONFIG 中每个关键参数做单参数 sweep
    只改一个字段，其余保持 baseline，方便分析敏感性。
    """

    # ===== 0) baseline =====
    yield "baseline", copy.deepcopy(BASE_CONFIG)

    # ===== 1) energy_per_cpu_cycle (psi) sweep =====
    # 较省电 -> baseline -> 更耗电
    for psi in [1e-9, 1.2e-9, 1.5e-9, 2e-9, 2.5e-9, 3e-9, 3.5e-9]:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["energy_per_cpu_cycle"] = psi
        yield f"psi={psi:g}", cfg

    # ===== 2) cpu_cycles_per_second (phi) sweep =====
    # 算力偏低 -> baseline -> 更高算力
    for phi in [1e10, 1.25e10, 1.5e10, 1.75e10, 2e10]:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["cpu_cycles_per_second"] = phi
        yield f"phi={phi:.2e}", cfg

    # ===== 3) battery_capacity_seconds (tau_b) sweep =====
    # 明显不足 -> 临界 -> 富余
    for cap in [600, 1200, 2400, 3600, 5400, 7200, 9000, 10800]:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["battery_capacity_seconds"] = cap
        yield f"tau_b={cap}", cfg

    # ===== 4) satellite_solar_panel_power_watts (P_solar) sweep =====
    # 低发电 -> baseline -> 高发电
    for p in [5, 10, 15, 20, 25, 30, 35, 40]:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["satellite_solar_panel_power_watts"] = p
        yield f"P_solar={p}", cfg

    # ===== 5) time_slot_length sweep =====
    # 更细时间粒度 -> baseline -> 更粗粒度
    for slot_len in [30, 45, 60, 90, 120]:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["time_slot_length"] = slot_len
        yield f"slot={slot_len}", cfg

    # ===== 6) task_average_execution sweep =====
    # 轻任务 -> baseline -> 重任务
    for exec_t in [50, 60, 70, 80, 90, 100]:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["task_average_execution"] = exec_t
        yield f"exec={exec_t}", cfg

    # ===== 7) task_average_period sweep =====
    # 高频（重载） -> baseline -> 低频（宽松）
    for T in [125, 150, 175, 200, 225]:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["task_average_period"] = T
        yield f"period={T}", cfg

    # ===== 8) task_average_deadline sweep =====
    # 更紧 ddl -> baseline -> 更宽松 ddl
    for D in [150, 225, 300, 375, 450]:
        cfg = copy.deepcopy(BASE_CONFIG)
        cfg["task_average_deadline"] = D
        yield f"deadline={D}", cfg


def build_case_from_config(config: dict):
    """
    用给定 config 调用 build_case，返回：
      tasks, A, e_jk, slot_len, tau_b, psi, phi, expected_feasible
    """
    # 固定随机种子，保证不同进程下同一配置可复现
    np.random.seed(42)
    bt.np.random.seed(42)

    # 同步 solar panel 参数到 build_testcase 内部
    bt.satellite_solar_panel_power_watts = config["satellite_solar_panel_power_watts"]

    task_info_dict, A_list, e_jk_list, slot_len, tau_b, psi, phi, expected_feasible = bt.build_case(
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

    # 转 Task 列表
    tasks = []
    for task_id_str, td in task_info_dict.items():
        task_id = int(task_id_str)
        tasks.append(
            alt.Task(
                task_id=task_id,
                period=td["period"],
                deadline=td["deadline"],
                job_exec_time=td["job_exec_time"],
                offset=td.get("offset", 0.0),
            )
        )

    A = np.array(A_list, dtype=int)
    e_jk = np.array(e_jk_list, dtype=float)
    return tasks, A, e_jk, slot_len, tau_b, psi, phi, expected_feasible


def run_single_config(tag: str, config: dict):
    """
    单配置实验：构造 case，跑多种算法。
    返回一个包含多算法结果的 list，每个元素是一行 CSV 数据。
    """
    tasks, A, e_jk, slot_len, tau_b, psi, phi, expected = build_case_from_config(config)
    Nc, Ns, Nt = A.shape
    horizon_sec = Nt * slot_len

    # ===== 运行所有算法 =====
    results = {}

    # 1) Max-flow (理论最优)
    results["maxflow"] = alt.feasibility_test(
        tasks=tasks, A=A, e_jk=e_jk, psi=psi, phi=phi,
        tau_b=tau_b, slot_len=slot_len, horizon_sec=horizon_sec,
        return_flow=False, debug=True,
    )

    # 2) 能量驱动启发式 (EDF + energy)
    results["energy_first"] = alt.heuristic_most_energy_first(
        tasks=tasks, A=A, e_jk=e_jk, psi=psi, phi=phi,
        tau_b=tau_b, slot_len=slot_len, horizon_sec=horizon_sec,
        debug=False,
    )

    # 3) 随机分配 baseline
    results["random"] = alt.heuristic_random_assignment(
        tasks=tasks, A=A, e_jk=e_jk, psi=psi, phi=phi,
        tau_b=tau_b, slot_len=slot_len, horizon_sec=horizon_sec,
    )

    # 4) EDF baseline
    results["edf"] = alt.heuristic_edf(
        tasks=tasks, A=A, e_jk=e_jk, psi=psi, phi=phi,
        tau_b=tau_b, slot_len=slot_len, horizon_sec=horizon_sec,
    )

    # ===== 通用元信息 =====
    meta = {
        "tag": tag,
        "Nc": Nc, "Ns": Ns, "Nt": Nt,
        "horizon_sec": horizon_sec,
        "energy_per_cpu_cycle": psi,
        "cpu_cycles_per_second": phi,
        "battery_capacity_seconds": tau_b,
        "satellite_solar_panel_power_watts": config["satellite_solar_panel_power_watts"],
        "time_slot_length": slot_len,
        "task_average_period": config["task_average_period"],
        "task_average_deadline": config["task_average_deadline"],
        "task_average_execution": config["task_average_execution"],
        "expected_feasible": expected,
    }

    # ===== 汇总成多行结果 =====
    rows = []
    for alg, res in results.items():
        rows.append({
            **meta,
            "algorithm": alg,
            "feasible": res["feasible"],
            "max_flow_value": res["max_flow_value"],
            "total_demand": res["total_demand"],
            "coverage_ratio": res["coverage_ratio"],
            "num_jobs": res["num_jobs"],
            "num_nodes": res.get("num_nodes"),
            "num_edges": res.get("num_edges"),
        })

    # ===== 打印摘要 =====
    msg = f"[{tag}] "
    msg += " ".join([
        f"{alg}: feas={res['feasible']} cov={res['coverage_ratio']:.3f}"
        for alg, res in results.items()
    ])
    print(msg)

    return rows



def write_csv(path: str, rows: list):
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    import os
    OUT_CSV = os.path.abspath("results_all.csv")  # 用绝对路径避免工作目录混乱
    configs = list(iter_configs())
    total = len(configs)
    print(f"Total configs: {total}", flush=True)

    all_rows = []

    # —— 先同步跑 1 个，保证马上生成文件
    tag0, cfg0 = configs[0]
    print(f"[Warmup] running first config: {tag0}", flush=True)
    rows0 = run_single_config(tag0, cfg0)   # 这里应当返回 list[dict]
    all_rows.extend(rows0)
    write_csv(OUT_CSV, all_rows)
    print(f"[Progress] 1/{total} finished: {tag0}", flush=True)

    # —— 再并行跑剩余
    from concurrent.futures import ProcessPoolExecutor, as_completed
    max_workers = min(os.cpu_count() or 4, max(1, total - 1))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut2tag = {ex.submit(run_single_config, tag, cfg): tag for tag, cfg in configs[1:]}
        for i, fut in enumerate(as_completed(fut2tag), start=2):
            tag = fut2tag[fut]
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[Error] {tag}: {e}", flush=True)
                continue
            all_rows.extend(rows)
            write_csv(OUT_CSV, all_rows)
            print(f"[Progress] {i}/{total} finished: {tag}", flush=True)

    print(f"\n[Done] All algorithm results saved to {OUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
