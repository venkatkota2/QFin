"""Small curated numerical mutation campaign, isolated from the working checkout.

This is a bounded regression-sensitivity check, not an exhaustive mutation score.
No runtime dependency or production eval/exec is introduced.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    ("discount sign", "finance/rates.py", "_exp_result(-rate * time)", "_exp_result(rate * time)"),
    ("simple accrual horizon", "finance/rates.py", "base = 1.0 + rate * time", "base = 1.0 + rate"),
    (
        "continuous conversion sign",
        "finance/rates.py",
        "return _finite_result(-log(value) / time)",
        "return _finite_result(log(value) / time)",
    ),
    ("VaR tie boundary", "finance/risk.py", 'side="left"', 'side="right"'),
    ("CVaR tail normalization", "finance/risk.py", "/ (1.0 - confidence)", "/ confidence"),
    ("variance centering", "finance/risk.py", "(losses - mean) ** 2", "losses ** 2"),
    (
        "forward month anchor",
        "finance/dates.py",
        "add_months(anchor, step * months,",
        "add_months(cursor, months,",
    ),
    (
        "backward month anchor",
        "finance/dates.py",
        "add_months(anchor, -step * months,",
        "add_months(cursor, -months,",
    ),
    (
        "bootstrap repricing gate",
        "finance/calibration.py",
        "passed=abs(residual_value) <= tolerance",
        "passed=True",
    ),
]


def campaign():
    from qfin import _native

    site_paths = sorted({sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]})
    tests = [
        str(ROOT / path)
        for path in [
            "tests/finance/test_rates.py",
            "tests/finance/test_risk.py",
            "tests/finance/test_dates.py",
            "tests/finance/test_calibration.py",
            "tests/test_reference_corpus.py",
            "tests/test_reference_extensions.py",
        ]
    ]
    rows = []
    with tempfile.TemporaryDirectory(prefix="qfin-mutation-") as temporary:
        source = Path(temporary) / "src"
        shutil.copytree(ROOT / "src", source, ignore=shutil.ignore_patterns("__pycache__"))
        native_file = Path(_native.require().__file__)
        shutil.copyfile(native_file, source / "qfin" / native_file.name)
        # -S prevents editable-install .pth redirection. Import QFin before pytest
        # and verify its path so a mutation can never accidentally test the original.
        runner = (
            f"import sys; sys.path[:0] = {json.dumps([str(source), *site_paths])}; "
            "import qfin; "
            f"assert qfin.__file__.startswith({str(source)!r}), qfin.__file__; "
            "import pytest; raise SystemExit(pytest.main(sys.argv[1:]))"
        )
        command = [sys.executable, "-S", "-c", runner, "-q", "--tb=short", *tests]
        environment = {
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        baseline = subprocess.run(
            command, capture_output=True, text=True, env=environment, timeout=120
        )
        if baseline.returncode:
            raise RuntimeError(
                "isolated unmodified control failed:\n" + baseline.stdout + baseline.stderr
            )
        for name, relative, before, after in MUTATIONS:
            path = source / "qfin" / relative
            original = path.read_text()
            if before not in original:
                raise RuntimeError(f"mutation site missing: {name}")
            path.write_text(original.replace(before, after))
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, env=environment, timeout=120
                )
            finally:
                path.write_text(original)
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"mutation test infrastructure error: {name}\n{result.stdout}{result.stderr}"
                )
            rows.append(
                {
                    "mutation": name,
                    "module": relative,
                    "killed": result.returncode == 1,
                    "test_summary": result.stdout.strip().splitlines()[-1],
                }
            )
            print(name, "killed" if result.returncode else "SURVIVED", flush=True)
    killed = sum(row["killed"] for row in rows)
    return {
        "scope": "curated numerical regression sensitivity; no exhaustive claim",
        "control": "unmodified isolated source passed",
        "mutations": rows,
        "killed": killed,
        "total": len(rows),
        "score": killed / len(rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = campaign()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["killed"] == result["total"] else 1)
