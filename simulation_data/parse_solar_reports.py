#!/usr/bin/env python3
"""
Parse satellite solar intensity reports into:
{ <sat_name>: { <time_str>: <solar_percentage_float>, ... }, ... }

- Input: a folder containing text files in the shown format.
- Output: prints the nested dict and can optionally save JSON.

Example header:
  Satellite-Sat_P1_S3
Then repeating pairs (often with blank lines):
  <Time (UTCG) line>
  <Intensity line>
"""

from pathlib import Path
import re
import json
import argparse
from datetime import datetime
from collections import defaultdict

# Match block header like: "Satellite-Sat_P1_S3"
HEADER_RE = re.compile(r'^\s*Satellite-(\S+)\s*$')

# Match a timestamp line like: "5 Nov 2025 17:00:00.000"
TIME_RE = re.compile(
    r'^\s*(\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?)\s*$'
)

# Match an intensity line like: "100.000000" or "0.000000" (may include blanks around)
INT_RE = re.compile(r'^\s*([+-]?\d+(?:\.\d+)?)\s*$')

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

def parse_solar_file(path: Path):
    """
    Parse one file to:
      { sat_name: { time_str: solar_float, ... }, ... }
    """
    out = defaultdict(dict)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    current_sat = None
    pending_time = None

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # Detect a new satellite block
        m_header = HEADER_RE.match(line)
        if m_header:
            current_sat = m_header.group(1).strip()
            pending_time = None
            i += 1
            continue

        # If inside a satellite block, look for time and intensity pairs
        if current_sat:
            m_time = TIME_RE.match(line)
            if m_time:
                # Remember this time, and look ahead for the next numeric intensity line
                pending_time = m_time.group(1).strip()

                # Search forward for the next intensity line
                j = i + 1
                intensity_val = None
                while j < n:
                    m_int = INT_RE.match(lines[j])
                    if m_int:
                        # Avoid picking lines that look like header separators (e.g., "----------")
                        # The INT_RE already filters those out; proceed to record.
                        intensity_val = float(m_int.group(1))
                        break
                    # break if we hit a new Satellite- header (next block starts)
                    if HEADER_RE.match(lines[j]):
                        break
                    j += 1

                if pending_time is not None and intensity_val is not None:
                    # convert pending_time from D MMM YYYY HH:MM:SS.sss to YYYY-MM-DDTHH:MM:SS.sss
                    dt_iso = parse_dt(pending_time)
                    out[current_sat][dt_iso] = intensity_val

                # Move i forward to the point we consumed up to (or just advance 1 if none found)
                i = j if intensity_val is not None else i + 1
                continue

        i += 1

    return dict(out)


def parse_solar_folder(folder: Path, pattern: str = "*.txt"):
    """
    Walk the folder and merge results from all files.
    """
    merged = defaultdict(dict)
    for p in sorted(folder.glob(pattern)):
        file_dict = parse_solar_file(p)
        for sat, time_map in file_dict.items():
            # Merge (later files override identical timestamps if collisions occur)
            merged[sat].update(time_map)
    return {k: v for k, v in merged.items()}


def main():
    ap = argparse.ArgumentParser(description="Parse solar intensity reports into a nested dictionary.")
    ap.add_argument("folder", type=Path, help="Folder containing .txt solar reports")
    ap.add_argument("--glob", default="*.txt", help="Glob pattern (default: *.txt)")
    ap.add_argument("--save-json", type=Path, default=None, help="Optional path to save JSON output")
    args = ap.parse_args()

    data = parse_solar_folder(args.folder, args.glob)
    print(json.dumps(data, indent=2))

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved JSON to: {args.save_json}")


if __name__ == "__main__":
    main()
