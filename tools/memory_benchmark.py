"""Fresh-process peak RSS and public API timing dispersion (no runtime dependency)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def peak_rss_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in [
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                ]
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def workload(name, engine, chunk):
    import numpy as np

    import qfin
    from qfin.algorithms import CircuitObservation, maximum_likelihood_amplitude_estimate

    curve = qfin.YieldCurve([0, 10, 60], [0.02, 0.03, 0.04])
    bonds = [qfin.FixedRateBond(20, 0.04)] * (10000 if name == "bonds" else 20)
    model = qfin.ALMModel(
        qfin.AssetPortfolio(bonds), qfin.LiabilityPortfolio([qfin.CashFlow(10, 1800)]), curve
    )
    if name == "bonds":
        return lambda: qfin.price_bonds(
            bonds, curve, engine=engine
        ), "10,000 bonds; 400,000 cashflows"
    if name == "rate_scenarios":
        scenarios = qfin.RateScenarioSet.parallel(curve, np.linspace(-0.01, 0.01, 2000))
        return lambda: model.run_scenarios(
            scenarios, engine=engine, chunk_size=chunk
        ), "2,000 scenarios; 20 bonds"
    if name == "alm_paths":
        scenarios = qfin.EconomicScenarioSet(np.zeros((1000, 20, 3)))
        return lambda: model.project_paths(
            scenarios, engine=engine, scenario_chunk_size=chunk
        ), "1,000 scenarios x 20 periods"
    if name == "life_scenarios":
        policies = qfin.PolicyModelPointSet(
            [qfin.LifePolicy(40 + i, 10000, 100, 20) for i in range(10)], [1000] * 10
        )
        assumptions = qfin.ProjectionAssumptions(qfin.MortalityTable([0, 120], [0.01, 0.01]), curve)
        scenarios = qfin.EconomicScenarioSet(np.zeros((1000, 20, 3)))
        return lambda: qfin.project_liability_scenarios(
            policies,
            assumptions,
            scenarios,
            engine=engine,
            scenario_chunk_size=chunk,
            policy_chunk_size=4,
        ), "1,000 scenarios x 10 model points x 20 years"
    if name == "factor_validation":
        from qfin.representation import encode_independent_factors

        factors = encode_independent_factors(
            [qfin.Normal(), qfin.Normal()], qubits_per_factor=6, method="probability"
        )
        factor_model = qfin.FactorizedLossModel(
            factors, qfin.SparseExposureObjective(linear={"factor_0": 1.0})
        )
        problem = qfin.FactorCVaR(factor_model, confidence=0.95)
        return lambda: qfin.evaluate_factor_risk(
            problem, chunk_size=chunk, max_points=4096
        ), "4,096 joint points; streamed"
    observations = tuple(CircuitObservation(k, 500, 1000) for k in [0, 1, 2, 4])
    return lambda: maximum_likelihood_amplitude_estimate(
        observations, grid_size=131073
    ), "131,073-point MLAE likelihood grid"


def measure(name, engine, chunk, repeats):
    import qfin

    function, size = workload(name, engine, chunk)
    before = peak_rss_bytes()
    function()
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        durations.append(time.perf_counter() - started)
    return {
        "workload": name,
        "size": size,
        "engine": engine,
        "chunk": chunk,
        "seconds": durations,
        "median_seconds": statistics.median(durations),
        "min_seconds": min(durations),
        "max_seconds": max(durations),
        "stdev_seconds": statistics.pstdev(durations),
        "warmups": 1,
        "peak_rss_bytes": peak_rss_bytes(),
        "peak_before_workload_bytes": before,
        "rss_scope": "fresh process lifetime, includes imports, inputs, warmup and timed calls",
        "qfin": qfin.__version__,
        "compiler": qfin.system_info()["native_compiler"],
    }


def campaign(repeats):
    rows = []
    for name in [
        "bonds",
        "rate_scenarios",
        "alm_paths",
        "life_scenarios",
        "factor_validation",
        "mlae",
    ]:
        engines = (
            ["numpy", "native"]
            if name in ["bonds", "rate_scenarios", "alm_paths", "life_scenarios"]
            else ["numpy"]
        )
        for engine in engines:
            for chunk in (
                [16, 256]
                if name in ["rate_scenarios", "alm_paths", "life_scenarios", "factor_validation"]
                else [256]
            ):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--worker",
                        name,
                        "--engine",
                        engine,
                        "--chunk",
                        str(chunk),
                        "--repeats",
                        str(repeats),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                rows.append(json.loads(result.stdout))
                print(name, engine, chunk, "recorded", flush=True)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        "measurements": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--engine", default="numpy")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.worker:
        print(json.dumps(measure(args.worker, args.engine, args.chunk, args.repeats)))
    elif args.output:
        args.output.write_text(json.dumps(campaign(args.repeats), indent=2) + "\n")
    else:
        parser.error("output is required")
