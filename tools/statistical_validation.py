"""End-to-end sampled VaR/CVaR study, including adaptive threshold selection.

Runs the public compiler and PennyLane simulation, not production outputs as
expected values. Independent Decimal quantiles/ES define the finite references.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from decimal import Decimal
from pathlib import Path

import numpy as np

import qfin

DISTRIBUTIONS = {
    "uniform": ([0, 1, 2, 3], [1, 1, 1, 1]),
    "two_point": ([0, 10], [9, 1]),
    "repeated_atom": ([-1, 0, 0, 100], [1, 3, 5, 1]),
    "rare_tail": ([0, 1000], [999999, 1]),
    "near_degenerate": ([0, 1], [1, 999999]),
    "symmetric": ([-3, -1, 1, 3], [1, 1, 1, 1]),
}


def exact_risk(losses, weights, confidence):
    total = sum(Decimal(str(w)) for w in weights)
    pairs = sorted(
        (Decimal(str(x)), Decimal(str(w)) / total) for x, w in zip(losses, weights, strict=True)
    )
    alpha = Decimal(str(confidence))
    cumulative = Decimal(0)
    var = None
    integral = Decimal(0)
    for loss, weight in pairs:
        previous = cumulative
        cumulative += weight
        if var is None and cumulative >= alpha:
            var = loss
        integral += loss * max(Decimal(0), cumulative - max(previous, alpha))
    return float(var), float(integral / (1 - alpha))


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    d = 1 + z * z / count
    center = (p + z * z / (2 * count)) / d
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / d
    return [max(0, center - half), min(1, center + half)]


def compile_fixture(name, losses, weights, alpha, kind):
    if name.startswith("factor_"):
        encoding = qfin.DistributionEncoding(
            grid=np.array(losses, dtype=float),
            probabilities=np.array(weights, dtype=float) / sum(weights),
            qubits=int(math.log2(len(losses))),
            lower_bound=min(losses),
            upper_bound=max(losses),
            tail_probability=0.0,
            discretization_error=0.0,
            mean_error=0.0,
            objective="independent finite fixture",
        )
        model = qfin.FactorizedLossModel(
            qfin.FactorizedDistributionEncoding(factors=(encoding,), factor_names=("loss",)),
            qfin.SparseExposureObjective(linear={"loss": 1.0}),
        )
        problem = (qfin.FactorVaR if kind == "var" else qfin.FactorCVaR)(model, confidence=alpha)
        return qfin.compile(problem, target_error=0.01, arithmetic_scale=1.0)
    problem = (qfin.VaR if kind == "var" else qfin.CVaR)(
        qfin.LossDistribution(losses, weights), confidence=alpha
    )
    return qfin.compile(problem, target_error=0.01, min_qubits=2, max_qubits=2)


def study(*, repetitions=100, quick=False, ambiguity_only=False, device="lightning.qubit"):
    rows = []
    configurations = [
        (0.90, 100, (0,)),
        (0.95, 500, (0, 1, 2)),
        (0.99, 2000, (0, 1, 2, 4)),
        (0.995, 100, (0, 1, 2)),
        (0.999, 2000, (0, 1, 2, 4)),
    ]
    if quick:
        configurations = [(0.90, 100, (0,)), (0.995, 500, (0, 1, 2))]
    elif ambiguity_only:
        configurations = [(0.95, 100, (1,))]
    fixtures = list(DISTRIBUTIONS.items())[:2] if quick else list(DISTRIBUTIONS.items())
    fixtures += [("factor_two_point", ([0, 1], [9, 1]))]
    if not quick:
        fixtures += [("factor_uniform", ([0, 1, 2, 3], [1, 1, 1, 1]))]
    for name, (losses, weights) in fixtures:
        for alpha, shots, schedule in configurations:
            exact_var, exact_es = exact_risk(losses, weights, alpha)
            for kind in ["var", "cvar"]:
                compiled = compile_fixture(name, losses, weights, alpha, kind)
                reference = exact_var if kind == "var" else exact_es
                covered = 0
                encoded_covered = 0
                var_correct = 0
                errors = []
                widths = []
                query_counts = []
                started = time.perf_counter()
                for index in range(repetitions):
                    result = compiled.run_quantum(
                        shots=shots,
                        schedule=schedule,
                        seed=71833 + index,
                        likelihood_grid_size=4097,
                        device_name=device,
                    )
                    lower, upper = result.confidence_interval_95
                    covered += lower - 1e-12 <= reference <= upper + 1e-12
                    encoded_covered += lower - 1e-12 <= compiled.encoded_value <= upper + 1e-12
                    var_correct += abs(result.value_at_risk - exact_var) < 1e-10
                    errors.append(result.value - reference)
                    widths.append(upper - lower)
                    query_counts.append(
                        len(result.search.evaluations) + len(result.excess_estimates)
                        if name.startswith("factor_")
                        else len(result.amplitude_estimates)
                    )
                rows.append(
                    {
                        "distribution": name,
                        "kind": kind,
                        "confidence": alpha,
                        "shots": shots,
                        "schedule": list(schedule),
                        "repetitions": repetitions,
                        "seed_start": 71833,
                        "exact_reference": reference,
                        "encoded_reference": compiled.encoded_value,
                        "empirical_interval_coverage": covered / repetitions,
                        "coverage_wilson_95": wilson(covered, repetitions),
                        "encoded_reference_coverage": encoded_covered / repetitions,
                        "encoding_error": compiled.encoded_value - reference,
                        "correct_selected_var_frequency": var_correct / repetitions,
                        "mean_error": sum(errors) / repetitions,
                        "max_absolute_error": max(map(abs, errors)),
                        "mean_interval_width": sum(widths) / repetitions,
                        "query_count_range": [min(query_counts), max(query_counts)],
                        "seconds": time.perf_counter() - started,
                        "provenance": result.provenance,
                    }
                )
                print(f"{name} {kind} alpha={alpha}: coverage={covered}/{repetitions}", flush=True)
    return {
        "schema_version": 1,
        "qfin_version": qfin.__version__,
        "python": platform.python_version(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True)
        ),
        "shot_source": "actual PennyLane circuits",
        "device": device,
        "interpretation": (
            "Empirical marginal coverage; CVaR intervals condition on selected VaR. "
            "No simultaneous guarantee."
        ),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--ambiguity-only", action="store_true")
    parser.add_argument("--device", default="lightning.qubit")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repetitions < 2:
        parser.error("repetitions must be at least two")
    report = study(
        repetitions=args.repetitions,
        quick=args.quick,
        ambiguity_only=args.ambiguity_only,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(report['rows'])} end-to-end statistical cells recorded")
