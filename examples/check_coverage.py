"""Enforce independent line and branch coverage floors from pytest-cov JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FLOORS = {
    "finance": 95.0,
    "compiler": 92.0,
    "representation": 92.0,
    "resources": 90.0,
    "backends": 85.0,
    "_native": 95.0,
}
BRANCH_FLOORS = {
    "finance/curves.py": 85.0,
    "finance/fixed_income.py": 85.0,
    "finance/calibration.py": 80.0,
    "finance/scenarios.py": 85.0,
    "finance/alm.py": 75.0,
    "finance/alm_paths.py": 80.0,
    "finance/life.py": 90.0,
    "finance/life_scenarios.py": 75.0,
    "finance/risk.py": 90.0,
    "finance/optimization.py": 75.0,
    "representation/": 75.0,
    "compiler/": 80.0,
    "algorithms/amplitude_estimation.py": 80.0,
    "_native/": 100.0,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    totals = data["totals"]
    line_percentage = 100 * totals["covered_lines"] / totals["num_statements"]
    branch_percentage = (
        100 * totals.get("covered_branches", 0) / max(1, totals.get("num_branches", 0))
    )
    failed = line_percentage < 93 or branch_percentage < 81
    print(
        f"Global lines {line_percentage:.2f}% (required 93%), "
        f"branches {branch_percentage:.2f}% (required 81%)"
    )
    for section, floor in FLOORS.items():
        summaries = [
            item["summary"]
            for name, item in data["files"].items()
            if f"/qfin/{section}/" in name.replace("\\", "/")
        ]
        total = sum(item["num_statements"] for item in summaries)
        covered = sum(item["covered_lines"] for item in summaries)
        percentage = 0.0 if total == 0 else 100 * covered / total
        passed = total > 0 and percentage >= floor
        print(
            f"{section}: {percentage:.2f}% (required {floor:.0f}%) {'PASS' if passed else 'FAIL'}"
        )
        failed = failed or not passed
    for section, floor in BRANCH_FLOORS.items():
        summaries = [
            item["summary"]
            for name, item in data["files"].items()
            if f"/qfin/{section}" in name.replace("\\", "/")
        ]
        total = sum(item.get("num_branches", 0) for item in summaries)
        covered = sum(item.get("covered_branches", 0) for item in summaries)
        percentage = 0.0 if total == 0 else 100 * covered / total
        passed = total > 0 and percentage >= floor
        print(
            f"{section} branches: {percentage:.2f}% (required {floor:.0f}%) "
            f"{'PASS' if passed else 'FAIL'}"
        )
        failed = failed or not passed
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
