import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import median

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment satellite access and solar data with phase-shifted synthetic satellites."
    )
    parser.add_argument(
        "--access-json",
        default="parsed_access.json",
        help="Path to the parsed access JSON file.",
    )
    parser.add_argument(
        "--solar-json",
        default="solar_parsed.json",
        help="Path to the parsed solar JSON file.",
    )
    parser.add_argument(
        "--new-planes",
        type=int,
        default=1,
        help="How many synthetic orbital planes to append.",
    )
    parser.add_argument(
        "--sats-per-plane",
        type=int,
        default=8,
        help="How many satellites each plane contains.",
    )
    parser.add_argument(
        "--output-access-json",
        default=None,
        help="Optional output path for augmented access JSON. Defaults to in-place overwrite.",
    )
    parser.add_argument(
        "--output-solar-json",
        default=None,
        help="Optional output path for augmented solar JSON. Defaults to in-place overwrite.",
    )
    return parser.parse_args()


def parse_satellite_name(name: str):
    parts = name.split("_")
    return int(parts[1][1:]), int(parts[2][1:])


def format_satellite_name(plane_id: int, sat_id: int):
    return f"Sat_P{plane_id}_S{sat_id}"


def compute_access_bounds(access_dict):
    start_time = None
    end_time = None
    for sat_dict in access_dict.values():
        for windows in sat_dict.values():
            for start_str, end_str, _ in windows:
                start = np.datetime64(start_str)
                end = np.datetime64(end_str)
                if start_time is None or start < start_time:
                    start_time = start
                if end_time is None or end > end_time:
                    end_time = end
    return start_time, end_time


def compute_shift_seconds(access_dict, plane_id: int, sats_per_plane: int, horizon_start):
    reference_place = max(access_dict, key=lambda place: len(access_dict[place]))
    start_offsets = []
    sat_dict = access_dict[reference_place]
    for sat_id in range(1, sats_per_plane + 1):
        sat_name = format_satellite_name(plane_id, sat_id)
        windows = sat_dict.get(sat_name, [])
        if not windows:
            continue
        first_start = np.datetime64(windows[0][0])
        offset = (first_start - horizon_start).astype("timedelta64[s]").astype(int)
        start_offsets.append(offset)
    start_offsets.sort()
    if len(start_offsets) < 2:
        return 6 * 60
    gaps = [b - a for a, b in zip(start_offsets, start_offsets[1:]) if b > a]
    if not gaps:
        return 6 * 60
    return max(60, int(round(median(gaps) / 2.0)))


def wrap_windows(windows, shift_seconds, horizon_start, horizon_end):
    horizon_seconds = (horizon_end - horizon_start).astype("timedelta64[s]").astype(int)
    wrapped = []
    for start_str, end_str, duration in windows:
        start = np.datetime64(start_str)
        end = np.datetime64(end_str)
        window_seconds = (end - start).astype("timedelta64[s]").astype(int)
        start_offset = (start - horizon_start).astype("timedelta64[s]").astype(int)
        shifted_offset = (start_offset + shift_seconds) % horizon_seconds
        shifted_start = horizon_start + np.timedelta64(shifted_offset, "s")
        shifted_end = shifted_start + np.timedelta64(window_seconds, "s")
        if shifted_end <= horizon_end:
            wrapped.append([str(shifted_start), str(shifted_end), duration])
            continue
        overflow_seconds = (shifted_end - horizon_end).astype("timedelta64[s]").astype(int)
        wrapped.append([str(shifted_start), str(horizon_end), float((horizon_end - shifted_start).astype("timedelta64[ms]").astype(int) / 1000.0)])
        wrapped.append([str(horizon_start), str(horizon_start + np.timedelta64(overflow_seconds, "s")), float(overflow_seconds)])
    wrapped.sort(key=lambda item: item[0])
    return wrapped


def shift_solar_series(series_dict, shift_seconds):
    timestamps = sorted(series_dict)
    values = [series_dict[t] for t in timestamps]
    if len(timestamps) < 2:
        return deepcopy(series_dict)
    slot_seconds = int((np.datetime64(timestamps[1]) - np.datetime64(timestamps[0])).astype("timedelta64[s]").astype(int))
    shift_slots = (shift_seconds // slot_seconds) % len(values)
    shifted_values = values[-shift_slots:] + values[:-shift_slots] if shift_slots else values[:]
    return {timestamp: shifted_values[idx] for idx, timestamp in enumerate(timestamps)}


def augment_access(access_dict, new_planes, sats_per_plane):
    augmented = deepcopy(access_dict)
    existing_planes = sorted({parse_satellite_name(name)[0] for sat_dict in access_dict.values() for name in sat_dict})
    max_plane_id = max(existing_planes)
    horizon_start, horizon_end = compute_access_bounds(access_dict)

    plane_shifts = {}
    for new_plane_offset in range(new_planes):
        source_plane = existing_planes[new_plane_offset % len(existing_planes)]
        new_plane = max_plane_id + new_plane_offset + 1
        shift_seconds = compute_shift_seconds(access_dict, source_plane, sats_per_plane, horizon_start)
        plane_shifts[new_plane] = (source_plane, shift_seconds)
        for place, sat_dict in augmented.items():
            for sat_id in range(1, sats_per_plane + 1):
                source_name = format_satellite_name(source_plane, sat_id)
                if source_name not in sat_dict:
                    continue
                new_name = format_satellite_name(new_plane, sat_id)
                sat_dict[new_name] = wrap_windows(
                    sat_dict[source_name],
                    shift_seconds,
                    horizon_start,
                    horizon_end,
                )
    return augmented, plane_shifts


def augment_solar(solar_dict, plane_shifts, sats_per_plane):
    augmented = deepcopy(solar_dict)
    for new_plane, (source_plane, shift_seconds) in plane_shifts.items():
        for sat_id in range(1, sats_per_plane + 1):
            source_name = format_satellite_name(source_plane, sat_id)
            if source_name not in solar_dict:
                continue
            new_name = format_satellite_name(new_plane, sat_id)
            augmented[new_name] = shift_solar_series(solar_dict[source_name], shift_seconds)
    return augmented


def main():
    args = parse_args()
    access_path = Path(args.access_json)
    solar_path = Path(args.solar_json)
    output_access_path = Path(args.output_access_json) if args.output_access_json else access_path
    output_solar_path = Path(args.output_solar_json) if args.output_solar_json else solar_path

    access_dict = json.loads(access_path.read_text())
    solar_dict = json.loads(solar_path.read_text())

    augmented_access, plane_shifts = augment_access(
        access_dict,
        new_planes=args.new_planes,
        sats_per_plane=args.sats_per_plane,
    )
    augmented_solar = augment_solar(
        solar_dict,
        plane_shifts=plane_shifts,
        sats_per_plane=args.sats_per_plane,
    )

    output_access_path.write_text(json.dumps(augmented_access, indent=4))
    output_solar_path.write_text(json.dumps(augmented_solar, indent=4))

    print(f"Added planes: {sorted(plane_shifts)}")
    print(f"Total satellites in solar data: {len(augmented_solar)}")


if __name__ == "__main__":
    main()
