"""Alternating, same-interpreter comparison of ALM period-boundary reuse."""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def worker(source: Path, repeats: int) -> dict[str, object]:
    # An editable-install meta finder can take precedence even over PYTHONPATH.
    # Prefer the requested checkout for Python modules; the unchanged installed
    # native extension can still be resolved by the existing wheel finder.
    class SourceFinder:
        @staticmethod
        def find_spec(fullname, path=None, target=None):
            if fullname == "qfin" or fullname.startswith("qfin."):
                return importlib.machinery.PathFinder.find_spec(
                    fullname, [str(source / "src")] if path is None else path
                )
            return None

    sys.meta_path.insert(0, SourceFinder())
    import numpy as np

    import qfin
    from qfin.finance import alm_paths

    if Path(alm_paths.__file__).resolve() != (source / "src/qfin/finance/alm_paths.py").resolve():
        raise RuntimeError("benchmark imported the wrong source checkout")

    curve = qfin.YieldCurve([0, 10, 60], [0.02, 0.03, 0.04])
    model = qfin.ALMModel(
        qfin.AssetPortfolio([qfin.FixedRateBond(20, 0.04)] * 20),
        qfin.LiabilityPortfolio([qfin.CashFlow(10, 1800)]),
        curve,
    )
    scenarios = qfin.EconomicScenarioSet(np.zeros((1000, 20, 3)))
    fields = (
        "asset_values",
        "bond_values",
        "cash_values",
        "equity_values",
        "liability_values",
        "liability_payments",
        "surplus",
        "funding_ratio",
        "transaction_costs",
    )
    rows = []
    for chunk in (16, 256):
        result = model.project_paths(scenarios, engine="numpy", scenario_chunk_size=chunk)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = model.project_paths(scenarios, engine="numpy", scenario_chunk_size=chunk)
            durations.append(time.perf_counter() - start)
        rows.append(
            {
                "chunk": chunk,
                "seconds": durations,
                "median_seconds": statistics.median(durations),
                "array_sha256": {
                    name: hashlib.sha256(getattr(result, name).tobytes()).hexdigest()
                    for name in fields
                },
            }
        )
    return {
        "source": qfin.__file__,
        "installed_metadata_version": qfin.__version__,
        "source_alm_sha256": hashlib.sha256(
            (source / "src/qfin/finance/alm_paths.py").read_bytes()
        ).hexdigest(),
        "numpy": np.__version__,
        "rows": rows,
    }


def compare(baseline: Path, candidate: Path, repeats: int) -> dict[str, object]:
    environment = dict(
        os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    blocks = []
    for label, source in [
        ("baseline", baseline),
        ("candidate", candidate),
        ("candidate", candidate),
        ("baseline", baseline),
    ]:
        raw = subprocess.check_output(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(source),
                "--repeats",
                str(repeats),
            ],
            env=environment,
            text=True,
        )
        blocks.append({"label": label, **json.loads(raw)})
        print(f"completed {label}", flush=True)
    return {
        "baseline_commit": subprocess.check_output(
            ["git", "-C", str(baseline), "rev-parse", "HEAD"], text=True
        ).strip(),
        "candidate_parent": subprocess.check_output(
            ["git", "-C", str(candidate), "rev-parse", "HEAD"], text=True
        ).strip(),
        "candidate_dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(candidate), "status", "--porcelain"], text=True
            )
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "warmups_per_block": 1,
        "repetitions_per_block": repeats,
        "size": "1000 scenarios x 20 periods x 20 bonds",
        "blocks": blocks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path, default=Path.cwd())
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.worker:
        print(json.dumps(worker(args.worker, args.repeats)))
    elif args.baseline and args.output:
        result = compare(args.baseline, args.candidate, args.repeats)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    else:
        parser.error("--baseline and --output are required")
