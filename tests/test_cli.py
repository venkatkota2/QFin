"""The intentionally small CLI preserves the existing Python pricing contract."""

import json

import pytest

from qfin.__main__ import main


def _price_arguments(kind: str) -> tuple[str, ...]:
    return ("price", "--kind", kind, "--spot", "100", "--strike", "105",
            "--maturity", "1", "--rate", "0.02", "--volatility", "0.2",
            "--target-error", "10", "--min-qubits", "3", "--max-qubits", "3")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_cli_compile_only_explains_objective_and_units(
    kind: str, capsys: pytest.CaptureFixture[str],
) -> None:
    assert main((*_price_arguments(kind), "--compile-only")) == 0
    explanation = capsys.readouterr().out
    assert f"European {kind}" in explanation
    assert "currency / price units" in explanation
    assert "no quantum-advantage claim" in explanation


@pytest.mark.parametrize("kind", ["call", "put"])
def test_cli_seeded_execution_emits_reproducible_result_json(
    kind: str, capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("pennylane")
    arguments = (*_price_arguments(kind), "--shots", "64", "--schedule", "0", "1",
                 "--seed", "913", "--device", "default.qubit")
    assert main(arguments) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == first
    assert first["target_error_unit"] == "currency / price units"
    lower, upper = first["confidence_interval_95"]
    assert 0 <= lower <= first["value"] <= upper


def test_cli_help_and_invalid_integer_are_actionable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(("--help",))
    assert help_exit.value.code == 0
    assert "limited European-option" in capsys.readouterr().out
    with pytest.raises(SystemExit) as invalid_exit:
        main((*_price_arguments("call"), "--shots", "1.5"))
    assert invalid_exit.value.code == 2
    assert "invalid int value" in capsys.readouterr().err
