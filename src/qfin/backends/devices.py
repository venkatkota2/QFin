"""Tested PennyLane devices and explicit research resource targets."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

from qfin.exceptions import BackendUnavailableError

Topology = Literal["all_to_all", "linear", "custom"]

_TESTED_DEVICES = ("lightning.qubit", "default.qubit", "default.mixed")
_SUPPORTED_BASIS_GATES = frozenset({"RX", "RY", "RZ", "CNOT"})


def resolve_quantum_device(device_name: str, *, require_available: bool = True) -> str:
    """Resolve ``auto`` and reject device claims QFin has not tested."""

    if device_name == "auto":
        if find_spec("pennylane_lightning") is not None:
            return "lightning.qubit"
        if find_spec("pennylane") is not None:
            return "default.qubit"
        if not require_available:
            return "default.qubit"
        raise BackendUnavailableError(
            "No tested PennyLane device is installed. Install QFin with "
            "`python -m pip install -e '.[quantum]'`."
        )
    if device_name not in _TESTED_DEVICES:
        raise BackendUnavailableError(
            f"PennyLane device {device_name!r} is not registered as tested by QFin 0.7; "
            f"choose one of {', '.join(_TESTED_DEVICES)} or use OpenQASM/Qiskit export"
        )
    if require_available and find_spec("pennylane") is None:
        raise BackendUnavailableError(
            "PennyLane is required for quantum execution. Install QFin with "
            "`python -m pip install -e '.[quantum]'`."
        )
    if (
        require_available
        and device_name == "lightning.qubit"
        and find_spec("pennylane_lightning") is None
    ):
        raise BackendUnavailableError(
            "lightning.qubit was requested but PennyLane-Lightning is not installed"
        )
    return device_name


def available_tested_devices() -> tuple[str, ...]:
    """Return only devices installed and exercised by QFin's test contract."""

    if find_spec("pennylane") is None:
        return ()
    devices = ["default.qubit", "default.mixed"]
    if find_spec("pennylane_lightning") is not None:
        devices.insert(0, "lightning.qubit")
    return tuple(devices)


@dataclass(frozen=True, slots=True)
class DeviceTarget:
    """Portable gate and coupling target used for resource analysis.

    Built-in targets are explicitly synthetic research targets. They estimate
    decomposition and routing overhead without pretending to represent a
    particular vendor device or calibration snapshot.
    """

    name: str
    wires: int
    coupling_map: tuple[tuple[int, int], ...]
    topology: Topology
    basis_gates: tuple[str, ...] = ("RX", "RY", "RZ", "CNOT")
    provider: str = "qfin-research"
    hardware_calibrated: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("target name must not be empty")
        if self.wires < 2:
            raise ValueError("target requires at least two wires")
        if self.topology not in ("all_to_all", "linear", "custom"):
            raise ValueError("topology must be 'all_to_all', 'linear', or 'custom'")
        if not self.basis_gates or len(set(self.basis_gates)) != len(self.basis_gates):
            raise ValueError("basis_gates must be non-empty and unique")
        unsupported = set(self.basis_gates) - _SUPPORTED_BASIS_GATES
        if unsupported:
            raise ValueError(
                "QFin's portable decomposer does not support basis gates: "
                + ", ".join(sorted(unsupported))
            )
        if set(self.basis_gates) != _SUPPORTED_BASIS_GATES:
            raise ValueError("portable targets currently require RX, RY, RZ, and CNOT")

        normalized: list[tuple[int, int]] = []
        for left, right in self.coupling_map:
            if left == right:
                raise ValueError("coupling edges cannot be self-loops")
            if not 0 <= left < self.wires or not 0 <= right < self.wires:
                raise ValueError("coupling edge references an unavailable wire")
            normalized.append((min(left, right), max(left, right)))
        if len(set(normalized)) != len(normalized):
            raise ValueError("coupling_map edges must be unique")
        normalized_tuple = tuple(sorted(normalized))
        if not normalized_tuple:
            raise ValueError("coupling_map must contain at least one edge")
        object.__setattr__(self, "coupling_map", normalized_tuple)
        if not self._is_connected():
            raise ValueError("coupling_map must connect every target wire")

        complete_edges = self.wires * (self.wires - 1) // 2
        if self.topology == "all_to_all" and len(normalized_tuple) != complete_edges:
            raise ValueError("all_to_all target must contain every undirected edge")
        if self.topology == "linear":
            expected = tuple((wire, wire + 1) for wire in range(self.wires - 1))
            if normalized_tuple != expected:
                raise ValueError("linear target must contain nearest-neighbour edges only")

    def _is_connected(self) -> bool:
        adjacency = {wire: set[int]() for wire in range(self.wires)}
        for left, right in self.coupling_map:
            adjacency[left].add(right)
            adjacency[right].add(left)
        visited = {0}
        pending = [0]
        while pending:
            wire = pending.pop()
            for neighbour in adjacency[wire] - visited:
                visited.add(neighbour)
                pending.append(neighbour)
        return len(visited) == self.wires

    @property
    def is_all_to_all(self) -> bool:
        return len(self.coupling_map) == self.wires * (self.wires - 1) // 2

    @classmethod
    def all_to_all(cls, wires: int) -> DeviceTarget:
        return cls(
            name=f"research-all-to-all-{wires}q",
            wires=wires,
            coupling_map=tuple(
                (left, right) for left in range(wires) for right in range(left + 1, wires)
            ),
            topology="all_to_all",
        )

    @classmethod
    def linear(cls, wires: int) -> DeviceTarget:
        return cls(
            name=f"research-linear-{wires}q",
            wires=wires,
            coupling_map=tuple((wire, wire + 1) for wire in range(wires - 1)),
            topology="linear",
        )

    @classmethod
    def custom(
        cls,
        name: str,
        wires: int,
        coupling_map: tuple[tuple[int, int], ...],
    ) -> DeviceTarget:
        return cls(
            name=name,
            wires=wires,
            coupling_map=coupling_map,
            topology="custom",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "wires": self.wires,
            "basis_gates": list(self.basis_gates),
            "topology": self.topology,
            "coupling_map": [list(edge) for edge in self.coupling_map],
            "provider": self.provider,
            "hardware_calibrated": self.hardware_calibrated,
            "caveat": (
                "Synthetic portable target; counts do not include calibration, "
                "scheduling, pulse compilation, queue time, or error correction."
            ),
        }


def resolve_device_target(
    target: DeviceTarget | Literal["all_to_all", "linear"],
    *,
    wires: int,
) -> DeviceTarget:
    """Resolve a built-in topology and enforce circuit-width compatibility."""

    if target == "all_to_all":
        return DeviceTarget.all_to_all(wires)
    if target == "linear":
        return DeviceTarget.linear(wires)
    if not isinstance(target, DeviceTarget):
        raise TypeError("target must be a DeviceTarget, 'all_to_all', or 'linear'")
    if target.wires != wires:
        raise ValueError(
            f"target has {target.wires} wires but the compiled circuit requires {wires}"
        )
    return target


__all__ = [
    "DeviceTarget",
    "Topology",
    "available_tested_devices",
    "resolve_device_target",
    "resolve_quantum_device",
]
