"""Re-run independent corpus assertions and record observed per-metric errors."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

import qfin

ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "rate_conversions": ["discount_factor", "continuous_rate"],
    "daycounts": ["year_fraction"],
    "fixed_income": [
        "dirty_price",
        "macaulay_duration",
        "modified_duration",
        "convexity",
        "dv01",
        "clean_plus_accrued",
        "solved_yield",
    ],
    "curves": ["discount_factor", "node_discount_factors", "zero_rate_discount_identity"],
    "bootstrapping": ["discount_factors"],
    "risk": ["var", "cvar", "mean"],
    "alm": ["asset_pv", "liability_pv", "surplus", "deficit", "funding_ratio"],
    "life": ["premiums", "benefits", "expenses", "net", "pv"],
    "dated_bonds": ["dirty_price", "clean_price", "accrued_interest"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "reference_tests", ROOT / "tests/test_reference_corpus.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_check = module.check
    summaries = {}
    active = {}

    def check(actual, expected, **options):
        original_check(actual, expected, **options)
        group = active["group"]
        metric = METRICS[group][active["call"]]
        active["call"] += 1
        key = f"{group}/{metric}/{active['engine']}"
        error = np.abs(np.asarray(actual) - np.asarray(expected))
        bound = options.get("atol", 1e-12) + options.get("rtol", 2e-12) * np.abs(expected)
        ratio = np.divide(error, bound, out=np.zeros_like(error), where=np.asarray(bound) != 0)
        maximum = float(np.max(error, initial=0))
        item = summaries.setdefault(
            key,
            {
                "comparisons": 0,
                "max_absolute_error": 0.0,
                "max_tolerance_fraction": 0.0,
                "worst_case": None,
            },
        )
        item["comparisons"] += 1
        if item["worst_case"] is None or maximum > item["max_absolute_error"]:
            item["max_absolute_error"] = maximum
            item["worst_case"] = active["id"]
        item["max_tolerance_fraction"] = max(
            item["max_tolerance_fraction"], float(np.max(ratio, initial=0))
        )
        if "materiality" in options:
            item["max_materiality_fraction"] = max(
                item.get("max_materiality_fraction", 0), maximum / options["materiality"]
            )

    module.check = check
    functions = {
        "rate_conversions": "rates",
        "daycounts": "daycounts",
        "schedules": "schedules",
        "fixed_income": "bonds",
        "curves": "curves",
        "bootstrapping": "bootstrap",
        "risk": "risk",
        "alm": "alm",
        "life": "life",
    }
    executed = 0
    independent = 0
    for group, name in functions.items():
        cases = json.loads((ROOT / "tests/reference_data" / f"{group}.json").read_text())
        independent += len(cases)
        for case in cases:
            for engine in (
                ["numpy", "native"]
                if group in {"fixed_income", "risk", "alm", "life"}
                else ["python"]
            ):
                active.update(group=group, id=case["id"], engine=engine, call=0)
                function = getattr(module, f"test_reference_{name}")
                function(case, engine) if engine != "python" else function(case)
                executed += 1
    dated = json.loads((ROOT / "tests/reference_data/dated_bonds.json").read_text())
    independent += len(dated)
    for case in dated:
        bond = qfin.FixedRateBond.from_dates(
            case["issue"],
            case["maturity"],
            case["coupon"],
            face_value=case["face"],
            day_count=case["coupon_day_count"],
            business_day_convention="unadjusted",
            end_of_month=True,
        )
        curve = qfin.YieldCurve([0, 100], [case["rate"]] * 2, valuation_date=case["settlement"])
        for engine in ["numpy", "native"]:
            active.update(group="dated_bonds", id=case["id"], engine=engine, call=0)
            result = qfin.price_bonds(bond, curve, engine=engine)
            for attribute, key in [
                ("dirty_prices", "dirty_price"),
                ("clean_prices", "clean_price"),
                ("accrued_interest", "accrued_interest"),
            ]:
                check(
                    getattr(result, attribute)[0],
                    case[key],
                    rtol=2e-12,
                    atol=1e-10,
                    materiality=max(1e-10, case["face"] * 2e-12),
                )
            executed += 1
    report = {
        "qfin": qfin.__version__,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        ),
        "scope": (
            "All stored corpus, including QuantLib dated bonds and exact date/schedule "
            "assertions. The 26 inline Decimal extensions are separately gated by pytest."
        ),
        "independent_configurations": independent,
        "executed_cases": executed,
        "metrics": summaries,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{executed} corpus cases passed; {len(summaries)} metric/engine summaries recorded")


if __name__ == "__main__":
    main()
