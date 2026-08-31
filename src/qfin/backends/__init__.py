"""Quantum runtime adapters."""

from qfin.backends.devices import (
    DeviceTarget,
    available_tested_devices,
    resolve_device_target,
    resolve_quantum_device,
)
from qfin.backends.factorized import FactorizedTailPennyLaneBackend
from qfin.backends.pennylane import (
    CompressedPennyLaneBackend,
    DensePennyLaneBackend,
    PennyLaneBackend,
    StructuredPennyLaneBackend,
)
from qfin.backends.risk import RiskPennyLaneBackend

__all__ = [
    "CompressedPennyLaneBackend",
    "DensePennyLaneBackend",
    "DeviceTarget",
    "FactorizedTailPennyLaneBackend",
    "PennyLaneBackend",
    "RiskPennyLaneBackend",
    "StructuredPennyLaneBackend",
    "available_tested_devices",
    "resolve_device_target",
    "resolve_quantum_device",
]
