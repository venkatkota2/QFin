"""Serializable execution context without importing optional simulation packages."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version


def execution_provenance(
    *,
    device: str,
    seed: int | None,
    shots: int,
    schedule: Sequence[int],
    representation: str,
    qubits: int,
    likelihood_grid_size: int,
    settings: dict[str, object] | None = None,
) -> dict[str, object]:
    dependencies: dict[str, str | None] = {}
    for package in ("qfin-quantum", "numpy", "scipy", "pennylane", "pennylane-lightning"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return {
        "schema_version": 1,
        "dependencies": dependencies,
        "execution_engine": "quantum_simulator",
        "backend": "pennylane",
        "device": device,
        "seed": seed,
        "shots_per_circuit": shots,
        "mlae_schedule": list(schedule),
        "representation": representation,
        "data_qubits": qubits,
        "likelihood_grid_size": likelihood_grid_size,
        "precision_settings": dict(settings or {}),
        "randomness_contract": "seeded backend-dependent sampling; no cross-device bitwise promise",
        "adaptive_seed_policy": "CDF queries add their query index; excess uses a separate offset",
    }


def interval_semantics(kind: str, *, simultaneous: bool = False) -> dict[str, object]:
    return {
        "level": 0.95,
        "scope": (
            "simultaneous_var_selection_and_excess"
            if simultaneous and kind == "conditional_value_at_risk"
            else "simultaneous_adaptive_cdf_bounds"
            if simultaneous and kind == "value_at_risk"
            else "conditional_on_selected_var"
            if kind == "conditional_value_at_risk"
            else "adaptive_local_regions"
            if kind == "value_at_risk"
            else "fixed_objective_guarded_likelihood_region"
        ),
        "simultaneous_workflow_coverage": simultaneous,
        "includes_deterministic_encoding_error": False,
        "sampling_assumptions": (
            "ideal binomial shots conditional on previous adaptive queries; "
            "excludes noise, calibration error, and cross-run simultaneous coverage"
        ),
        "empirical_validation": "docs/statistical-validation.md",
    }
