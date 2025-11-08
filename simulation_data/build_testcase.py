import os
import json
import numpy as np
from typing import List, Tuple, Any
from dataclasses import dataclass
'''
Build a test case with given parameters.
Parameters:
- sat_access_dict: dictionary mapping satellite IDs to their access windows
- energy_dict: dictionary mapping satellite IDs to their energy profiles
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

SEED = 42
np.random.seed(SEED)

energy_per_cpu_cycle = 1e-9  # J
cpu_cycles_per_second = 1e9  # cycles/s
battery_capacity_seconds = 3600  # 1 hour
satellite_solar_panel_power_watts = 20.0  # The power in watts the solar panel can generate at 100% intensity
time_slot_length = 60  # 1 minute
task_average_period = 300  # 5 minutes
task_average_deadline = 300  # 5 minutes
task_average_execution = 50  # 50 seconds

@dataclass
class Task:
    """A periodic task."""
    task_id: int
    period: float          # seconds, T_i
    deadline: float        # seconds, D_i (relative to release)
    job_exec_time: float   # seconds of processing required per job (p_i)
    # Optional: allow a start offset if needed in the future (default releases at T_i, 2T_i, ...)
    offset: float = 0.0    # seconds; by default keep 0.0 so first release is at T_i (per spec)

def convert_access_reports_to_matrix(access_report_path: str, time_slot_length: int):
    """
    Convert satellite access reports to a visibility matrix A.
    Each entry A[i,j,k] indicates whether task i can see satellite j in time slot k.
    Args:
    access_report_path: path to the access report JSON file
    time_slot_length: length of each time slot in seconds
    Returns:
    A: visibility matrix of shape (Nc, Ns, Nt)
    Nc: number of tasks
    Ns: number of satellites
    Nt: number of time slots
    start_time: simulation start time (numpy.datetime64)
    end_time: simulation end time (numpy.datetime64)
    task_id_map: dictionary mapping task names to task IDs
    sat_id_map: dictionary mapping satellite names to satellite IDs
    """
    # read access reports from file (stub implementation)
    if not os.path.exists(access_report_path):
        raise FileNotFoundError(f"Access report file not found: {access_report_path}")
    with open(access_report_path, 'r') as f:
        sat_access_dict = json.load(f)
    # The sat_access_dict is structured as {task_name: {sat_name: [[start_time, end_time, duration], ...]}}
    # The time is in the ISO format as YYYY-MM-DDTHH:MM:SS.ZZZZZZ
    # It needs to be converted to a matrix A of shape (Nc, Ns, Nt)
    ## In the first pass of the data, we get the number of tasks, satellites, and time slots
    ## and make a dictionary of {task_name: task_id}, a dictionary of {sat_name: sat_id}, and get the start and end time of the entire simulation
    task_names = list(sat_access_dict.keys())
    sat_names = set()
    start_time = None
    end_time = None
    for task_name, sat_dict in sat_access_dict.items():
        for sat_name, access_windows in sat_dict.items():
            sat_names.add(sat_name)
            for window in access_windows:
                window_start = np.datetime64(window[0])
                window_end = np.datetime64(window[1])
                if start_time is None or window_start < start_time:
                    start_time = window_start
                if end_time is None or window_end > end_time:
                    end_time = window_end
    sat_names = list(sat_names)
    Nc = len(task_names)
    Ns = len(sat_names)
    total_seconds = (end_time - start_time).astype('timedelta64[s]').astype(int)
    Nt = int(np.ceil(total_seconds / time_slot_length))
    task_id_map = {task_name: i for i, task_name in enumerate(task_names)}
    sat_id_map = {sat_name: j for j, sat_name in enumerate(sat_names)}
    A = np.zeros((Nc, Ns, Nt), dtype=int)
    ## In the second pass, we fill in the matrix A based on the access windows
    for task_name, sat_dict in sat_access_dict.items():
        task_id = task_id_map[task_name]
        for sat_name, access_windows in sat_dict.items():
            sat_id = sat_id_map[sat_name]
            for window in access_windows:
                window_start = np.datetime64(window[0])
                window_end = np.datetime64(window[1])
                start_slot = int((window_start - start_time).astype('timedelta64[s]').astype(int) / time_slot_length)
                end_slot = int((window_end - start_time).astype('timedelta64[s]').astype(int) / time_slot_length)
                for k in range(start_slot, end_slot + 1):
                    if 0 <= k < Nt:
                        A[task_id, sat_id, k] = 1
    return A, Nc, Ns, Nt, start_time, end_time, task_id_map, sat_id_map

def convert_energy_reports_to_matrix(
        energy_report_path: str, 
        solar_panel_power_watts: float,
        start_time: str, 
        end_time: str, 
        time_slot_length: int,
        task_id_map: dict,
        sat_id_map: dict
    ) -> np.ndarray:
    """
    Convert satellite energy reports to an energy matrix e_jk.
    Each entry e_jk[j,k] indicates the energy available on satellite j in time slot k.
    Args:
    energy_report_path: path to the energy report JSON file
    solar_panel_power_watts: power of the solar panel at 100% intensity (W)
    start_time: simulation start time (numpy.datetime64)
    end_time: simulation end time (numpy.datetime64)
    time_slot_length: length of each time slot in seconds
    task_id_map: dictionary mapping task names to task IDs
    sat_id_map: dictionary mapping satellite names to satellite IDs
    Returns:
    e_jk: energy matrix of shape (Ns, Nt)
    """
    # read energy reports from file (stub implementation)
    if not os.path.exists(energy_report_path):
        raise FileNotFoundError(f"Energy report file not found: {energy_report_path}")
    with open(energy_report_path, 'r') as f:
        energy_dict = json.load(f)
    # The energy_dict is structured as {sat_name: {timestamp: intensity_percentage, ...}}
    # The time is in the ISO format as YYYY-MM-DDTHH:MM:SS.ZZZZZZ
    # It needs to be converted to a matrix e_jk of shape (Ns, Nt) where each entry is the energy available in Joules
    ## Initialize the energy matrix
    Ns = len(sat_id_map)
    total_seconds = (np.datetime64(end_time) - np.datetime64(start_time)).astype('timedelta64[s]').astype(int)
    Nt = int(np.ceil(total_seconds / time_slot_length))
    e_jk = np.zeros((Ns, Nt), dtype=float)
    ## Fill in the energy matrix based on the energy reports
    for sat_name, intensity_dict in energy_dict.items():
        if sat_name not in sat_id_map:
            continue
        sat_id = sat_id_map[sat_name]
        for timestamp_str, intensity in intensity_dict.items():
            timestamp = np.datetime64(timestamp_str)
            # if timestamp is outside the simulation time, skip
            if timestamp < np.datetime64(start_time) or timestamp >= np.datetime64(end_time):
                continue
            slot_index = int((timestamp - np.datetime64(start_time)).astype('timedelta64[s]').astype(int) / time_slot_length)
            if 0 <= slot_index < Nt:
                # Energy in Joules = Power (W) * Time (s) * Intensity (%)
                energy = solar_panel_power_watts * time_slot_length * (intensity / 100.0)
                e_jk[sat_id, slot_index] = energy
    return e_jk




def build_case(
    access_report_path: str,
    energy_report_path: str,
    energy_per_cpu_cycle: float,
    cpu_cycles_per_second: float,
    battery_capacity_seconds: float,
    time_slot_length: int,
    task_average_period: float,
    task_average_deadline: float,
    task_average_execution: float
) -> Tuple[List['Task'], np.ndarray, np.ndarray, float, float, float, float, Any]:
    # get visibility matrix A and other parameters
    A, Nc, Ns, Nt, start_time, end_time, task_id_map, sat_id_map = convert_access_reports_to_matrix(access_report_path, time_slot_length)
    # build tasks
    task_info_dict = {}
    for task_name, task_id in task_id_map.items():
        task_info_dict[task_id] = {
            'task_name': task_name,
            'period': np.random.exponential(task_average_period),
            'deadline': np.random.exponential(task_average_deadline),
            'job_exec_time': np.random.exponential(task_average_execution),
            'offset': 0.0
        }
    # convert visibility matrix A from numpy array to list for json serialization
    A = A.tolist()

    e_jk = convert_energy_reports_to_matrix(
        energy_report_path,
        satellite_solar_panel_power_watts,
        start_time,
        end_time,
        time_slot_length,
        task_id_map,
        sat_id_map
    )

    # convert e_jk from numpy array to list for json serialization
    e_jk = e_jk.tolist()

    expected_feasible = None  # unknown

    return task_info_dict, A, e_jk, time_slot_length, battery_capacity_seconds, energy_per_cpu_cycle, cpu_cycles_per_second, expected_feasible

if __name__ == "__main__":
    # Example usage
    access_report_path = "parsed_access.json"
    energy_report_path = "solar_parsed.json"
    task_info_dict, A, e_jk, slot_len, tau_b, psi, phi, expected_feasible = build_case(
        access_report_path=access_report_path,
        energy_report_path=energy_report_path,
        energy_per_cpu_cycle=energy_per_cpu_cycle,
        cpu_cycles_per_second=cpu_cycles_per_second,
        battery_capacity_seconds=battery_capacity_seconds,
        time_slot_length=time_slot_length,
        task_average_period=task_average_period,
        task_average_deadline=task_average_deadline,
        task_average_execution=task_average_execution
    )
    # save the test case info to a json file
    test_case = {
        "name": "satellite_test_case",
        "task_info_dict": task_info_dict,
        "A": A,
        "e_jk": e_jk,
        "slot_len": slot_len,
        "tau_b": tau_b,
        "psi": psi,
        "phi": phi,
        "expected_feasible": expected_feasible
    }
    with open("test_case.json", "w") as f:
        json.dump(test_case, f, indent=4)

    # # print some info
    # print(f"Number of tasks: {len(tasks)}")
    # print(f"Visibility matrix A shape: {A.shape}")
    # print(f"Energy matrix e_jk shape: {e_jk.shape}")
    # # print A and e_jk for verification
    # # save full version of A in a text file
    # with open("visibility_matrix_A.txt", "w") as f:
    #     for i in range(A.shape[0]):
    #         for j in range(A.shape[1]):
    #             f.write(f"Task {i}, Satellite {j}: " + " ".join(map(str, A[i,j,:])) + "\n")
    # print("Energy matrix e_jk:")
    # print(e_jk)