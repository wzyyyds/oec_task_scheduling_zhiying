#!/usr/bin/env python3
"""
Parse satellite access_records reports into:
{ <place_name>: { <satellite_name>: [(start_time, end_time, duration_sec), ...] } }

- Input: a folder containing text files in the shown format.
- Output: prints the nested dict and also saves JSON (optional).
"""

from pathlib import Path
import re
from datetime import datetime
import json
import argparse
from collections import defaultdict

# Regex for a block header like: "Place1-To-Sat_P1_S1"
HEADER_RE = re.compile(r'^\s*([^-]+)-To-(\S+)\s*$')

# Regex for a data line like:
# "1    5 Nov 2025 17:09:07.203    5 Nov 2025 17:20:44.944           697.741"
# Captures: start_str, stop_str, duration_str
ROW_RE = re.compile(
    r'^\s*\d+\s+'                                     # Access number
    r'(\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\s+'  # Start
    r'(\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\s+'  # Stop
    r'([\d.]+)\s*$'                                   # Duration (sec)
)

# Datetime format matching "5 Nov 2025 17:09:07.203"
DT_FMT_WITH_F = "%d %b %Y %H:%M:%S.%f"
DT_FMT_NO_F   = "%d %b %Y %H:%M:%S"


def parse_dt(s: str) -> str:
    """
    Parse the provided UTCG timestamp to ISO 8601 string (UTC-agnostic).
    Keeps high precision if present. Returns ISO string.
    """
    s = s.strip()
    # Try with fractional seconds, fall back to no-fraction
    try:
        dt = datetime.strptime(s, DT_FMT_WITH_F)
    except ValueError:
        dt = datetime.strptime(s, DT_FMT_NO_F)
    # Return ISO-like string without timezone (as file is UTCG text)
    # Example: "2025-11-05T17:09:07.203000"
    return dt.isoformat(timespec="microseconds")


def parse_access_block(lines_iter, first_header_line):
    """
    Given we're positioned at a header line (already read), parse that block.
    Returns: place, satellite, list_of_triples
    """
    m = HEADER_RE.match(first_header_line)
    if not m:
        return None, None, []

    place = m.group(1).strip()
    satellite = m.group(2).strip()

    records = []

    # Skip optional dashed line and any header rows until data lines start
    for line in lines_iter:
        if ROW_RE.match(line):
            # First data line found; process it and continue
            break
        # If we encounter a blank "Statistics" or other section before data, the block is empty
    else:
        # No more lines
        return place, satellite, records

    # We already have one data line in 'line'
    while True:
        rowm = ROW_RE.match(line)
        if rowm:
            start_s, stop_s, dur_s = rowm.groups()
            start_iso = parse_dt(start_s)
            stop_iso  = parse_dt(stop_s)
            duration  = float(dur_s)
            records.append((start_iso, stop_iso, duration))
        else:
            # If the line is not a data row, the block likely ended
            # (e.g., "Statistics", blank line, next header, etc.)
            # Put this line back into the iterator by creating a new iterator that yields it first
            lines_iter = iter([line] + list(lines_iter))
            break

        try:
            line = next(lines_iter)
        except StopIteration:
            break

    return place, satellite, records, lines_iter


def parse_file(path: Path):
    """
    Parse a single file into dict: { place: { satellite: [(start, stop, dur), ...] } }
    """
    result = defaultdict(lambda: defaultdict(list))

    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\n")
        headm = HEADER_RE.match(line)
        if headm:
            # Create an iterator from the remaining lines after this header
            lines_iter = iter(lines[i+1:])
            parsed = parse_access_block(lines_iter, line)
            if parsed is None:
                i += 1
                continue

            # Unpack, handling the optional returned iterator
            if len(parsed) == 4:
                place, satellite, recs, lines_iter = parsed
            else:
                place, satellite, recs = parsed
                lines_iter = iter([])

            if place and satellite and recs:
                result[place][satellite].extend(recs)

            # We consumed some unknown number of lines; recompute i by how many are left in lines_iter
            remaining = list(lines_iter)
            i = n - len(remaining)
            continue

        i += 1

    # Convert defaultdicts to normal dicts
    return {p: dict(sats) for p, sats in result.items()}


def parse_folder(folder: Path, pattern: str = "*.txt"):
    """
    Walk folder for matching files and merge results.
    """
    merged = defaultdict(lambda: defaultdict(list))
    for p in sorted(folder.glob(pattern)):
        file_result = parse_file(p)
        for place, sats in file_result.items():
            for sat, trips in sats.items():
                merged[place][sat].extend(trips)
    return {p: dict(sats) for p, sats in merged.items()}


def main():
    ap = argparse.ArgumentParser(description="Parse satellite access_records reports into a nested dictionary.")
    ap.add_argument("folder", type=Path, help="Folder containing .txt access_records reports")
    ap.add_argument("--glob", default="*.txt", help="Glob pattern (default: *.txt)")
    ap.add_argument("--save-json", type=Path, default=None, help="Optional path to save JSON output")
    args = ap.parse_args()

    data = parse_folder(args.folder, args.glob)
    print(json.dumps(data, indent=2))

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved JSON to: {args.save_json}")


if __name__ == "__main__":
    main()
