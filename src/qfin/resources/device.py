"""Gate-set decomposition and topology-aware resource accounting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from qfin.backends.devices import DeviceTarget, resolve_device_target
from qfin.exceptions import BackendUnavailableError, ResourceLimitError


class CircuitRuntime(Protocol):
    """Minimal circuit boundary required by device-realism tools."""

    total_wires: int
    objective_wire: int

    def circuit_tape(self, power: int = 0) -> Any: ...

    def queue_circuit(self, power: int = 0) -> None: ...

    def probability(
        self,
        power: int = 0,
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class TranspiledCircuitResources:
    """One MLAE circuit before and after portable decomposition and routing."""

    power: int
    high_level_gates: int
    high_level_depth: int
    decomposed_gates: int
    decomposed_depth: int
    routing_swaps: int
    routed_gates: int
    routed_depth: int
    one_qubit_gates: int
    two_qubit_gates: int
    gate_types: tuple[tuple[str, int], ...]
    routed_two_qubit_edges: tuple[tuple[int, int], ...]
    logical_to_physical: tuple[int, ...]
    objective_physical_wire: int

    @property
    def routing_gate_overhead(self) -> float:
        return self.routed_gates / self.decomposed_gates

    def to_dict(self) -> dict[str, object]:
        return {
            "power": self.power,
            "high_level_gates": self.high_level_gates,
            "high_level_depth": self.high_level_depth,
            "decomposed_gates": self.decomposed_gates,
            "decomposed_depth": self.decomposed_depth,
            "routing_swaps": self.routing_swaps,
            "routed_gates": self.routed_gates,
            "routed_depth": self.routed_depth,
            "one_qubit_gates": self.one_qubit_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "gate_types": dict(self.gate_types),
            "routed_two_qubit_edges": [list(edge) for edge in self.routed_two_qubit_edges],
            "logical_to_physical": list(self.logical_to_physical),
            "objective_physical_wire": self.objective_physical_wire,
            "routing_gate_overhead": self.routing_gate_overhead,
        }


@dataclass(frozen=True, slots=True)
class DeviceResourceReport:
    """Target-aware resources for a full non-adaptive MLAE schedule."""

    target: DeviceTarget
    schedule: tuple[int, ...]
    circuits: tuple[TranspiledCircuitResources, ...]
    shots_per_circuit: int
    objective_evaluations: int
    total_circuit_executions: int
    total_shots: int
    total_routed_gates_per_objective: int
    total_executed_gates: int
    maximum_routed_depth: int
    maximum_two_qubit_gates: int
    estimate_kind: str = "portable_gate_set_and_topology_estimate"

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target.to_dict(),
            "schedule": list(self.schedule),
            "circuits": [circuit.to_dict() for circuit in self.circuits],
            "shots_per_circuit": self.shots_per_circuit,
            "objective_evaluations": self.objective_evaluations,
            "total_circuit_executions": self.total_circuit_executions,
            "total_shots": self.total_shots,
            "total_routed_gates_per_objective": self.total_routed_gates_per_objective,
            "total_executed_gates": self.total_executed_gates,
            "maximum_routed_depth": self.maximum_routed_depth,
            "maximum_two_qubit_gates": self.maximum_two_qubit_gates,
            "estimate_kind": self.estimate_kind,
            "caveat": (
                "Portable PennyLane decomposition and SWAP routing estimate. It is not "
                "a pulse-level, calibrated-device, fault-tolerant, or runtime prediction."
            ),
        }


def _qml() -> Any:
    try:
        import pennylane as qml
    except ImportError as exc:
        raise BackendUnavailableError(
            "PennyLane is required for device resource analysis. Install QFin with "
            "`python -m pip install -e '.[quantum]'`."
        ) from exc
    return qml


def _tape_resources(tape: Any) -> tuple[int, int, dict[str, int], dict[int, int]]:
    qml = _qml()
    resources = qml.resource.resources_from_tape(tape)
    depth = 0 if resources.depth is None else int(resources.depth)
    return (
        int(sum(resources.gate_types.values())),
        depth,
        {str(name): int(count) for name, count in resources.gate_types.items()},
        {int(size): int(count) for size, count in resources.gate_sizes.items()},
    )


def transpile_circuit(
    runtime: CircuitRuntime,
    *,
    power: int = 0,
    target: DeviceTarget | Literal["all_to_all", "linear"] = "all_to_all",
) -> tuple[Any, TranspiledCircuitResources, DeviceTarget]:
    """Return a basis-decomposed, routed tape plus transparent counts."""

    if power < 0:
        raise ValueError("power must be non-negative")
    qml = _qml()
    resolved_target = resolve_device_target(target, wires=runtime.total_wires)
    original = runtime.circuit_tape(power)
    high_gates, high_depth, _, _ = _tape_resources(original)
    gate_set = set(resolved_target.basis_gates)
    try:
        decomposed = qml.transforms.decompose(
            original,
            gate_set=gate_set,
            max_expansion=50,
            num_work_wires=0,
            strict=True,
        )[0][0]
    except Exception as exc:
        raise ResourceLimitError(f"circuit cannot be decomposed into {sorted(gate_set)}") from exc
    invalid = sorted({operation.name for operation in decomposed.operations} - gate_set)
    if invalid:
        raise ResourceLimitError(
            "portable decomposition left unsupported operations: " + ", ".join(invalid)
        )
    decomposed_gates, decomposed_depth, _, _ = _tape_resources(decomposed)

    if resolved_target.is_all_to_all:
        routed_with_swaps = decomposed
    else:
        try:
            routed_with_swaps = qml.transforms.transpile(
                decomposed,
                coupling_map=list(resolved_target.coupling_map),
            )[0][0]
        except Exception as exc:
            raise ResourceLimitError(
                f"circuit cannot be routed onto {resolved_target.name!r}"
            ) from exc
    routing_swaps = sum(operation.name == "SWAP" for operation in routed_with_swaps.operations)
    physical_to_logical = list(range(resolved_target.wires))
    for operation in routed_with_swaps.operations:
        if operation.name == "SWAP":
            left, right = (int(wire) for wire in operation.wires)
            physical_to_logical[left], physical_to_logical[right] = (
                physical_to_logical[right],
                physical_to_logical[left],
            )
    logical_to_physical = [0] * resolved_target.wires
    for physical, logical in enumerate(physical_to_logical):
        logical_to_physical[logical] = physical
    routed = qml.transforms.decompose(
        routed_with_swaps,
        gate_set=gate_set,
        max_expansion=50,
        num_work_wires=0,
        strict=True,
    )[0][0]
    invalid = sorted({operation.name for operation in routed.operations} - gate_set)
    if invalid:
        raise ResourceLimitError("routing left unsupported operations: " + ", ".join(invalid))

    target_edges = set(resolved_target.coupling_map)
    used_edges: set[tuple[int, int]] = set()
    for operation in routed.operations:
        if len(operation.wires) > 2:
            raise ResourceLimitError("portable decomposition left a gate wider than two wires")
        if len(operation.wires) == 2:
            left, right = (int(wire) for wire in operation.wires)
            edge = (min(left, right), max(left, right))
            if edge not in target_edges:
                raise ResourceLimitError(
                    f"routing produced non-adjacent two-qubit gate on edge {edge}"
                )
            used_edges.add(edge)

    routed_gates, routed_depth, gate_types, gate_sizes = _tape_resources(routed)
    gate_counter = Counter(gate_types)
    report = TranspiledCircuitResources(
        power=power,
        high_level_gates=high_gates,
        high_level_depth=high_depth,
        decomposed_gates=decomposed_gates,
        decomposed_depth=decomposed_depth,
        routing_swaps=routing_swaps,
        routed_gates=routed_gates,
        routed_depth=routed_depth,
        one_qubit_gates=gate_sizes.get(1, 0),
        two_qubit_gates=gate_sizes.get(2, 0),
        gate_types=tuple(sorted(gate_counter.items())),
        routed_two_qubit_edges=tuple(sorted(used_edges)),
        logical_to_physical=tuple(logical_to_physical),
        objective_physical_wire=logical_to_physical[runtime.objective_wire],
    )
    return routed, report, resolved_target


def estimate_device_resources(
    runtime: CircuitRuntime,
    *,
    schedule: Sequence[int] = (0, 1, 2, 4),
    shots: int = 1_000,
    target: DeviceTarget | Literal["all_to_all", "linear"] = "all_to_all",
    objective_evaluations: int = 1,
) -> DeviceResourceReport:
    """Profile every unique circuit in a non-adaptive MLAE workflow."""

    powers = tuple(int(power) for power in schedule)
    if not powers or any(power < 0 for power in powers):
        raise ValueError("schedule must contain non-negative Grover powers")
    if len(set(powers)) != len(powers):
        raise ValueError("schedule powers must be unique")
    if shots <= 0:
        raise ValueError("shots must be positive")
    if objective_evaluations < 1:
        raise ValueError("objective_evaluations must be positive")

    reports: list[TranspiledCircuitResources] = []
    resolved_target: DeviceTarget | None = None
    for power in powers:
        _, report, circuit_target = transpile_circuit(
            runtime,
            power=power,
            target=target,
        )
        reports.append(report)
        resolved_target = circuit_target
    assert resolved_target is not None
    circuits = tuple(reports)
    gates_per_objective = sum(circuit.routed_gates for circuit in circuits)
    circuit_executions = len(circuits) * objective_evaluations
    return DeviceResourceReport(
        target=resolved_target,
        schedule=powers,
        circuits=circuits,
        shots_per_circuit=shots,
        objective_evaluations=objective_evaluations,
        total_circuit_executions=circuit_executions,
        total_shots=shots * circuit_executions,
        total_routed_gates_per_objective=gates_per_objective,
        total_executed_gates=gates_per_objective * shots * objective_evaluations,
        maximum_routed_depth=max(circuit.routed_depth for circuit in circuits),
        maximum_two_qubit_gates=max(circuit.two_qubit_gates for circuit in circuits),
    )


def to_openqasm_tape(
    runtime: CircuitRuntime,
    *,
    power: int = 0,
    target: DeviceTarget | Literal["all_to_all", "linear"] = "all_to_all",
) -> tuple[str, TranspiledCircuitResources, DeviceTarget]:
    """Serialize one routed QFin circuit as interoperable OpenQASM 2."""

    qml = _qml()
    tape, report, resolved_target = transpile_circuit(
        runtime,
        power=power,
        target=target,
    )
    program = cast(
        str,
        qml.to_openqasm(
            tape,
            wires=qml.wires.Wires(range(resolved_target.wires)),
            rotations=False,
            measure_all=True,
            precision=12,
        ),
    )
    return program, report, resolved_target


__all__ = [
    "CircuitRuntime",
    "DeviceResourceReport",
    "TranspiledCircuitResources",
    "estimate_device_resources",
    "to_openqasm_tape",
    "transpile_circuit",
]
