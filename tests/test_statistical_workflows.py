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
    assert report["interval_semantics"]["simultaneous_workflow_coverage"] is False
    assert report["interval_semantics"]["includes_deterministic_encoding_error"] is False
    assert report["interval_semantics"]["scope"] == (
        "conditional_on_selected_var" if kind == "cvar" else "adaptive_local_regions"
    )
    assert report["provenance"]["seed"] == 92
    assert report["provenance"]["shots_per_circuit"] == 200
    assert report["provenance"]["mlae_schedule"] == [0, 1]
    report["provenance"]["mlae_schedule"].append(999)
    assert first.provenance["mlae_schedule"] == [0, 1]


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
