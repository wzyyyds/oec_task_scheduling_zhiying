#!/usr/bin/env python3
import argparse
import csv
import os
from typing import Dict, List


DEFAULT_INPUT = os.path.join("results", "rebuttal", "milp_horizon_comparison.csv")
DEFAULT_OUTPUT_MD = os.path.join("results", "rebuttal", "integrality_gap_table.md")
DEFAULT_OUTPUT_CSV = os.path.join("results", "rebuttal", "integrality_gap_table.csv")
HORIZON_ORDER = ["10min", "1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h", "11h", "12h"]


def as_float(value: str):
    if value in (None, ""):
        return None
    return float(value)


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_gap_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    by_key = {(row["horizon_label"], row["algorithm"]): row for row in rows}
    gap_rows: List[Dict[str, object]] = []

    for label in HORIZON_ORDER:
        eco = by_key.get((label, "ecoflow"))
        milp = by_key.get((label, "milp"))
        if not eco or not milp:
            continue

        eco_obj = as_float(eco.get("max_flow_value"))
        milp_obj = as_float(milp.get("max_flow_value"))
        eco_cov = as_float(eco.get("coverage_ratio"))
        milp_cov = as_float(milp.get("coverage_ratio"))
        total_demand = as_float(milp.get("total_demand")) or as_float(eco.get("total_demand"))

        obj_gap_abs = abs(milp_obj - eco_obj)
        obj_gap_rel = obj_gap_abs / abs(milp_obj) if milp_obj else 0.0
        coverage_gap_abs = abs(milp_cov - eco_cov)

        gap_rows.append(
            {
                "horizon": label,
                "jobs": int(float(eco["num_jobs"])),
                "milp_status": milp.get("solver_status", ""),
                "ecoflow_coverage": eco_cov,
                "milp_coverage": milp_cov,
                "coverage_gap_abs": coverage_gap_abs,
                "ecoflow_objective": eco_obj,
                "milp_objective": milp_obj,
                "objective_gap_abs": obj_gap_abs,
                "objective_gap_rel_pct": obj_gap_rel * 100.0,
                "total_demand": total_demand,
            }
        )

    return gap_rows


def write_csv_table(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "horizon",
        "jobs",
        "milp_status",
        "ecoflow_coverage",
        "milp_coverage",
        "coverage_gap_abs",
        "ecoflow_objective",
        "milp_objective",
        "objective_gap_abs",
        "objective_gap_rel_pct",
        "total_demand",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "| Horizon | # Jobs | MILP status | ECoFlow coverage | MILP coverage | Abs. coverage gap | Rel. objective gap |",
        "|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {horizon} | {jobs} | {milp_status} | {ecoflow_coverage:.9f} | "
            "{milp_coverage:.9f} | {coverage_gap_abs:.2e} | {objective_gap_rel_pct:.2e}% |".format(**row)
        )
    lines.append("")
    lines.append(
        "Gap definition: relative objective gap = "
        "`|MILP objective - ECoFlow objective| / |MILP objective| * 100%`."
    )
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an integrality/objective gap table from MILP comparison data.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input MILP comparison CSV.")
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD, help="Output Markdown table.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Output CSV table.")
    args = parser.parse_args()

    rows = build_gap_rows(load_rows(args.input))
    write_csv_table(args.output_csv, rows)
    write_markdown_table(args.output_md, rows)
    print(f"Saved CSV table to {os.path.abspath(args.output_csv)}")
    print(f"Saved Markdown table to {os.path.abspath(args.output_md)}")


if __name__ == "__main__":
    main()
