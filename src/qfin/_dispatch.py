"""Deterministic, benchmark-supported classical engine selection (no calibration)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qfin import _native
from qfin.exceptions import QFinValidationError

Engine = Literal["auto", "numpy", "native"]
SelectedEngine = Literal["numpy", "native"]
KEY_RATE_CASHFLOW_VISITS = 8_000_000
ALM_BASE_CASHFLOWS = 4_096

# Units: counts of scalar cashflow visits, loss observations or model-point years.
# None disables auto native; explicit native remains available when compatible.
POLICIES: dict[str, int | None] = {
    "bond_price": None,
    "yield_solve": 1,
    "key_rate": KEY_RATE_CASHFLOW_VISITS,
    "alm_base": ALM_BASE_CASHFLOWS,
    "rate_scenarios": 1,
    "life": 1,
    "life_scenarios": 1,
    "weighted_risk": 0,
    "alm_paths": None,
    "mortality": None,
}


@dataclass(frozen=True, slots=True)
class EngineDecision:
    engine: SelectedEngine
    reason: str
    work_units: int
    threshold: int | None
    native_compatible: bool


def decide_engine(
    engine: Engine,
    workload: int,
    *,
    native_compatible: bool = True,
    auto_native_threshold: int | None = None,
) -> EngineDecision:
    if engine not in ("auto", "numpy", "native"):
        raise QFinValidationError("engine must be 'auto', 'numpy', or 'native'")
    if engine == "native":
        if not native_compatible:
            raise QFinValidationError(
                "native engine requires linear-zero interpolation with flat-zero extrapolation"
            )
        _native.require()
        selected: SelectedEngine = "native"
        reason = "explicit native request"
    elif engine == "numpy":
        selected, reason = "numpy", "explicit NumPy request"
    elif (
        native_compatible
        and auto_native_threshold is not None
        and workload >= auto_native_threshold
        and _native.available()
    ):
        selected, reason = "native", "workload reached the validated static crossover"
    else:
        selected, reason = (
            "numpy",
            "conservative policy, incompatible semantics or unavailable native",
        )
    return EngineDecision(selected, reason, workload, auto_native_threshold, native_compatible)


def resolve_engine(
    engine: Engine,
    workload: int,
    *,
    native_compatible: bool = True,
    auto_native_threshold: int | None = None,
) -> SelectedEngine:
    return decide_engine(
        engine,
        workload,
        native_compatible=native_compatible,
        auto_native_threshold=auto_native_threshold,
    ).engine
