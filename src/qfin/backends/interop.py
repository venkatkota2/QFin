"""OpenQASM export and non-executing provider capability inspection."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Literal

from qfin.backends.devices import DeviceTarget
from qfin.exceptions import BackendUnavailableError
from qfin.resources.device import (
    CircuitRuntime,
    TranspiledCircuitResources,
    to_openqasm_tape,
)


@dataclass(frozen=True, slots=True)
class QasmExport:
    """One target-routed OpenQASM 2 circuit and its provenance."""

    program: str
    power: int
    target: DeviceTarget
    resources: TranspiledCircuitResources
    sha256: str
    format: str = "OpenQASM 2.0"

    def to_dict(self, *, include_program: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "format": self.format,
            "power": self.power,
            "target": self.target.to_dict(),
            "resources": self.resources.to_dict(),
            "sha256": self.sha256,
        }
        if include_program:
            result["program"] = self.program
        return result


def export_openqasm(
    runtime: CircuitRuntime,
    *,
    power: int = 0,
    target: DeviceTarget | Literal["all_to_all", "linear"] = "all_to_all",
) -> QasmExport:
    """Export a decomposed QFin circuit without requiring Qiskit."""

    program, resources, resolved_target = to_openqasm_tape(
        runtime,
        power=power,
        target=target,
    )
    return QasmExport(
        program=program,
        power=power,
        target=resolved_target,
        resources=resources,
        sha256=sha256(program.encode("utf-8")).hexdigest(),
    )


def export_qiskit(
    runtime: CircuitRuntime,
    *,
    power: int = 0,
    target: DeviceTarget | Literal["all_to_all", "linear"] = "all_to_all",
) -> Any:
    """Return a Qiskit ``QuantumCircuit`` parsed from QFin's OpenQASM export."""

    if find_spec("qiskit") is None:
        raise BackendUnavailableError(
            "Qiskit export requires the optional dependency. Install QFin with "
            "`python -m pip install -e '.[qiskit]'`."
        )
    exported = export_openqasm(runtime, power=power, target=target)
    qasm2 = import_module("qiskit.qasm2")
    return qasm2.loads(exported.program)


@dataclass(frozen=True, slots=True)
class ProviderCapabilityReport:
    """Read-only capability snapshot for a Qiskit-style backend object."""

    backend_name: str
    provider: str
    num_qubits: int
    operation_names: tuple[str, ...]
    coupling_map: tuple[tuple[int, int], ...]
    coupling_connected: bool
    has_measurement: bool
    has_reset: bool
    has_entangling_gate: bool
    supports_dynamic_circuits: bool
    required_wires: int | None
    sufficient_wires: bool
    qfin_export_compatible: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "backend_name": self.backend_name,
            "provider": self.provider,
            "num_qubits": self.num_qubits,
            "operation_names": list(self.operation_names),
            "coupling_map": [list(edge) for edge in self.coupling_map],
            "coupling_connected": self.coupling_connected,
            "has_measurement": self.has_measurement,
            "has_reset": self.has_reset,
            "has_entangling_gate": self.has_entangling_gate,
            "supports_dynamic_circuits": self.supports_dynamic_circuits,
            "required_wires": self.required_wires,
            "sufficient_wires": self.sufficient_wires,
            "qfin_export_compatible": self.qfin_export_compatible,
            "caveat": (
                "Static capability inspection only. QFin does not authenticate, submit jobs, "
                "read calibration data, or claim execution support for this provider backend."
            ),
        }


def _backend_value(backend: Any, name: str, default: Any = None) -> Any:
    value = getattr(backend, name, default)
    return value() if callable(value) else value


def _connected(num_qubits: int, edges: tuple[tuple[int, int], ...]) -> bool:
    if num_qubits <= 1:
        return True
    if not edges:
        return False
    adjacency = {wire: set[int]() for wire in range(num_qubits)}
    for left, right in edges:
        if 0 <= left < num_qubits and 0 <= right < num_qubits:
            adjacency[left].add(right)
            adjacency[right].add(left)
    visited = {0}
    pending = [0]
    while pending:
        wire = pending.pop()
        for neighbour in adjacency[wire] - visited:
            visited.add(neighbour)
            pending.append(neighbour)
    return len(visited) == num_qubits


def inspect_qiskit_backend(
    backend: Any,
    *,
    required_wires: int | None = None,
) -> ProviderCapabilityReport:
    """Inspect a BackendV2-like object without credentials or job submission."""

    if required_wires is not None and required_wires < 1:
        raise ValueError("required_wires must be positive")
    backend_name = str(_backend_value(backend, "name", type(backend).__name__))
    provider_object = _backend_value(backend, "provider")
    provider = type(provider_object).__name__ if provider_object is not None else "unknown"
    num_qubits = int(_backend_value(backend, "num_qubits", 0))
    if num_qubits < 1:
        raise ValueError("backend must expose a positive num_qubits value")

    names_value = _backend_value(backend, "operation_names", ())
    operation_names = tuple(sorted({str(name).lower() for name in names_value}))
    coupling_value = _backend_value(backend, "coupling_map")
    if coupling_value is None:
        raw_edges: tuple[tuple[int, int], ...] = ()
    else:
        get_edges = getattr(coupling_value, "get_edges", None)
        values = get_edges() if callable(get_edges) else coupling_value
        raw_edges = tuple((int(left), int(right)) for left, right in values)
    coupling_map = tuple(
        sorted({(min(left, right), max(left, right)) for left, right in raw_edges})
    )

    entangling = {"cx", "cz", "ecr", "iswap", "rxx", "rzz"}
    dynamic = {"if_else", "while_loop", "for_loop", "switch_case"}
    has_measurement = "measure" in operation_names
    has_entangling = bool(entangling.intersection(operation_names))
    sufficient = required_wires is None or num_qubits >= required_wires
    connected = _connected(num_qubits, coupling_map) if coupling_map else num_qubits == 1
    compatible = (
        has_measurement
        and has_entangling
        and sufficient
        and (connected or required_wires in (None, 1))
    )
    return ProviderCapabilityReport(
        backend_name=backend_name,
        provider=provider,
        num_qubits=num_qubits,
        operation_names=operation_names,
        coupling_map=coupling_map,
        coupling_connected=connected,
        has_measurement=has_measurement,
        has_reset="reset" in operation_names,
        has_entangling_gate=has_entangling,
        supports_dynamic_circuits=bool(dynamic.intersection(operation_names)),
        required_wires=required_wires,
        sufficient_wires=sufficient,
        qfin_export_compatible=compatible,
    )


__all__ = [
    "ProviderCapabilityReport",
    "QasmExport",
    "export_openqasm",
    "export_qiskit",
    "inspect_qiskit_backend",
]
