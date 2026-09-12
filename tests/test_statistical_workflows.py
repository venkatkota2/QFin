"""Small deterministic end-to-end gates; the large coverage study is scheduled."""

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("pennylane")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "statistical_validation", ROOT / "tools/statistical_validation.py"
)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


@pytest.mark.parametrize("family", ["empirical", "factor_two_point"])
@pytest.mark.parametrize("kind", ["var", "cvar"])
def test_seeded_actual_workflow_and_serializable_interval_semantics(family, kind):
    compiled = study.compile_fixture(family, [0, 1], [9, 1], 0.9, kind)
    options = dict(
        shots=200, schedule=(0, 1), seed=92, likelihood_grid_size=4097, device_name="default.qubit"
    )
    first, second = compiled.run_quantum(**options), compiled.run_quantum(**options)
    assert first.value == second.value
    assert first.confidence_interval_95 == second.confidence_interval_95
    report = first.to_dict()
    assert json.loads(json.dumps(report))["value"] == first.value
    assert report["interval_semantics"]["simultaneous_workflow_coverage"] is True
    assert report["interval_semantics"]["includes_deterministic_encoding_error"] is False
    assert report["interval_semantics"]["scope"] == (
        "simultaneous_var_selection_and_excess"
        if kind == "cvar"
        else "simultaneous_adaptive_cdf_bounds"
    )
    assert report["provenance"]["seed"] == 92
    assert report["provenance"]["shots_per_circuit"] == 200
    assert report["provenance"]["mlae_schedule"] == [0, 1]
    report["provenance"]["mlae_schedule"].append(999)
    assert first.provenance["mlae_schedule"] == [0, 1]


@pytest.mark.parametrize("family,maximum", [("empirical", 1000), ("factor_two_point", 1)])
@pytest.mark.parametrize("seed", [71833, 71834, 71835])
def test_ambiguous_rare_tail_propagates_var_selection_uncertainty(family, maximum, seed):
    compiled = study.compile_fixture(family, [0, maximum], [999999, 1], 0.95, "cvar")
    result = compiled.run_quantum(
        shots=100,
        schedule=(1,),
        seed=seed,
        likelihood_grid_size=4097,
        device_name="default.qubit",
    )
    _, exact_es = study.exact_risk([0, maximum], [999999, 1], 0.95)
    lower, upper = result.confidence_interval_95
    assert lower <= exact_es <= upper
    assert upper > lower
    # Empirical endpoint encoding may move the upper support by one ULP.
    assert 0 <= lower <= upper <= maximum + 2 * abs(float(study.np.spacing(maximum)))
    count = (
        len(result.amplitude_estimates)
        if family == "empirical"
        else len(result.search.evaluations) + len(result.excess_estimates)
    )
    assert count <= result.provenance["sampling_objective_budget"]
    if family == "empirical":
        # The observations cannot distinguish alias modes. Do not disguise the
        # unchanged bad point estimate with a zero-width conditional interval.
        assert result.value > 900
        assert result.meets_target_error is False


def test_small_real_circuit_study_records_references_and_monte_carlo_error():
    report = study.study(repetitions=2, quick=True, device="default.qubit")
    assert len(report["rows"]) == 12
    assert report["shot_source"] == "actual PennyLane circuits"
    for row in report["rows"]:
        assert row["encoding_error"] == pytest.approx(0, abs=1e-12)
        coverage = row["empirical_interval_coverage"]
        lo, hi = row["coverage_wilson_95"]
        assert 0 <= lo <= coverage <= hi <= 1
        assert row["max_absolute_error"] >= 0
        assert row["query_count_range"][0] > 0
    assert study.exact_risk([0, 10], [9, 1], 0.9) == (0, 10)
    assert study.exact_risk([0, 1000], [999999, 1], 0.999) == (0, 1)
