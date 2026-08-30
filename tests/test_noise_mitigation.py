import pytest

import qfin

pytest.importorskip("pennylane")


@pytest.fixture
def small_model() -> qfin.CompiledPricingModel:
    return qfin.compile(
        qfin.EuropeanCall(strike=105, maturity=1.0),
        qfin.BlackScholes(spot=100, rate=0.04, volatility=0.20),
        target_error=1.0,
        min_qubits=3,
        max_qubits=3,
    )


def test_noise_model_validation_and_scaling() -> None:
    model = qfin.NoiseModel(0.01, 0.02)
    scaled = model.scaled(2.0)
    assert scaled.depolarizing_probability == 0.02
    assert scaled.readout_bit_flip_probability == 0.04
    with pytest.raises(ValueError, match="depolarizing_probability"):
        qfin.NoiseModel(-0.01)
    with pytest.raises(ValueError, match="readout_bit_flip_probability"):
        qfin.NoiseModel(0.01, 1.01)


def test_zero_noise_preserves_the_ideal_probability(
    small_model: qfin.CompiledPricingModel,
) -> None:
    report = small_model.noise_analysis(qfin.NoiseModel(0.0), power=1)
    assert report.noisy_probability == pytest.approx(report.ideal_probability, abs=1e-10)
    assert report.mitigated_probability == pytest.approx(report.ideal_probability, abs=1e-10)
    assert report.noisy_absolute_error < 1e-10


def test_deterministic_zne_reduces_error_for_reference_circuit(
    small_model: qfin.CompiledPricingModel,
) -> None:
    report = small_model.noise_analysis(
        qfin.NoiseModel(0.001, 0.002),
        power=0,
        shots=None,
        scale_factors=(1.0, 3.0, 5.0),
        extrapolation_order=1,
    )
    assert report.noisy_absolute_error > 0
    assert report.mitigation_improved
    assert report.mitigated_absolute_error < report.noisy_absolute_error
    assert not report.mitigation_clipped
    assert report.to_dict()["simulator"] == "default.mixed"


def test_shot_noise_experiment_is_repeatable_with_a_seed(
    small_model: qfin.CompiledPricingModel,
) -> None:
    kwargs = {
        "power": 0,
        "shots": 500,
        "seed": 23,
        "scale_factors": (1.0, 3.0),
        "extrapolation_order": 1,
    }
    first = small_model.noise_analysis(qfin.NoiseModel(0.002), **kwargs)
    second = small_model.noise_analysis(qfin.NoiseModel(0.002), **kwargs)
    assert first.scaled_probabilities == second.scaled_probabilities


def test_risk_objective_supports_the_same_noise_boundary() -> None:
    compiled = qfin.compile(
        qfin.TailProbability(qfin.LossDistribution([0.0, 1.0, 2.0, 3.0]), 1.5),
        target_error=0.2,
        min_qubits=2,
        max_qubits=2,
    )
    assert isinstance(compiled, qfin.CompiledRiskModel)
    report = compiled.noise_analysis(
        qfin.NoiseModel(0.001),
        scale_factors=(1.0, 3.0),
    )
    assert 0 <= report.mitigated_probability <= 1


def test_noise_analysis_rejects_invalid_folding_schedule(
    small_model: qfin.CompiledPricingModel,
) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        small_model.noise_analysis(
            qfin.NoiseModel(0.001),
            scale_factors=(1.0, 1.0),
        )
