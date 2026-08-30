"""Comparable state-preparation costs and target-aware selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qfin.exceptions import ResourceLimitError
from qfin.representation.encoding import DistributionEncoding
from qfin.representation.factorized import FactorizedDistributionEncoding

Representation = DistributionEncoding | FactorizedDistributionEncoding


class RepresentationTarget(Protocol):
    """Structural device fields used without importing a backend package."""

    @property
    def name(self) -> str: ...

    @property
    def wires(self) -> int: ...

    @property
    def topology(self) -> str: ...


@dataclass(frozen=True, slots=True)
class StatePreparationCost:
    """Construction and circuit cost for one state-preparation strategy."""

    strategy: str
    implemented: bool
    exact_for_representation: bool
    portable_gate_decomposition: bool
    data_qubits: int
    ancilla_qubits: int
    total_wires: int
    classical_parameters: int
    stored_values: int
    estimated_classical_memory_bytes: int
    high_level_gates: int
    high_level_depth_upper_bound: int
    requires_joint_materialization: bool
    target_compatible: bool
    within_parameter_limit: bool
    within_memory_limit: bool
    available: bool
    scaling: str
    note: str

    @property
    def selectable(self) -> bool:
        return (
            self.implemented
            and self.portable_gate_decomposition
            and self.target_compatible
            and self.within_parameter_limit
            and self.within_memory_limit
            and self.available
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "implemented": self.implemented,
            "exact_for_representation": self.exact_for_representation,
            "portable_gate_decomposition": self.portable_gate_decomposition,
            "data_qubits": self.data_qubits,
            "ancilla_qubits": self.ancilla_qubits,
            "total_wires": self.total_wires,
            "classical_parameters": self.classical_parameters,
            "stored_values": self.stored_values,
            "estimated_classical_memory_bytes": self.estimated_classical_memory_bytes,
            "high_level_gates": self.high_level_gates,
            "high_level_depth_upper_bound": self.high_level_depth_upper_bound,
            "requires_joint_materialization": self.requires_joint_materialization,
            "target_compatible": self.target_compatible,
            "within_parameter_limit": self.within_parameter_limit,
            "within_memory_limit": self.within_memory_limit,
            "available": self.available,
            "selectable": self.selectable,
            "scaling": self.scaling,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class StatePreparationStrategyReport:
    """Candidate comparison and the strategy selected under explicit limits."""

    candidates: tuple[StatePreparationCost, ...]
    selected_strategy: str | None
    target_name: str | None
    target_topology: str | None
    max_parameters: int
    max_memory_bytes: int
    selection_reason: str

    @property
    def selected(self) -> StatePreparationCost | None:
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.strategy == self.selected_strategy
            ),
            None,
        )

    def require_selected(self) -> StatePreparationCost:
        selected = self.selected
        if selected is None:
            raise ResourceLimitError(self.selection_reason)
        return selected

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_strategy": self.selected_strategy,
            "target_name": self.target_name,
            "target_topology": self.target_topology,
            "max_parameters": self.max_parameters,
            "max_memory_bytes": self.max_memory_bytes,
            "selection_reason": self.selection_reason,
            "caveat": (
                "Construction estimates compare implemented QFin loaders. They do not "
                "include payoff-oracle synthesis, pulse scheduling, or error correction."
            ),
        }


def _cost(
    *,
    strategy: str,
    implemented: bool,
    portable: bool,
    data_qubits: int,
    ancilla_qubits: int,
    parameters: int,
    stored_values: int,
    memory_bytes: int,
    gates: int,
    depth: int,
    joint: bool,
    available: bool,
    target: RepresentationTarget | None,
    max_parameters: int,
    max_memory_bytes: int,
    scaling: str,
    note: str,
) -> StatePreparationCost:
    total_wires = data_qubits + ancilla_qubits
    return StatePreparationCost(
        strategy=strategy,
        implemented=implemented,
        exact_for_representation=True,
        portable_gate_decomposition=portable,
        data_qubits=data_qubits,
        ancilla_qubits=ancilla_qubits,
        total_wires=total_wires,
        classical_parameters=parameters,
        stored_values=stored_values,
        estimated_classical_memory_bytes=memory_bytes,
        high_level_gates=gates,
        high_level_depth_upper_bound=depth,
        requires_joint_materialization=joint,
        target_compatible=target is None or total_wires <= target.wires,
        within_parameter_limit=parameters <= max_parameters,
        within_memory_limit=memory_bytes <= max_memory_bytes,
        available=available,
        scaling=scaling,
        note=note,
    )


def compare_state_preparation_strategies(
    representation: Representation,
    *,
    target: RepresentationTarget | None = None,
    ancilla_qubits: int = 0,
    max_parameters: int = 32_767,
    max_memory_bytes: int = 256 * 1024 * 1024,
    max_flattened_points: int = 65_536,
) -> StatePreparationStrategyReport:
    """Compare implemented loaders without allocating a joint factor table."""

    if ancilla_qubits < 0:
        raise ValueError("ancilla_qubits must be non-negative")
    if max_parameters < 0 or max_memory_bytes < 1 or max_flattened_points < 1:
        raise ValueError("strategy limits must be non-negative and non-zero")

    candidates: list[StatePreparationCost] = []
    preferred: str
    if isinstance(representation, DistributionEncoding):
        points = representation.grid_points
        uniform = representation.state_preparation_method == "uniform_quantile_hadamard" and bool(
            np.allclose(representation.probabilities, 1.0 / points)
        )
        if uniform:
            candidates.append(
                _cost(
                    strategy="uniform_quantile_hadamard",
                    implemented=True,
                    portable=True,
                    data_qubits=representation.qubits,
                    ancilla_qubits=ancilla_qubits,
                    parameters=0,
                    stored_values=points,
                    memory_bytes=representation.grid.nbytes,
                    gates=representation.qubits,
                    depth=1,
                    joint=False,
                    available=True,
                    target=target,
                    max_parameters=max_parameters,
                    max_memory_bytes=max_memory_bytes,
                    scaling="O(qubits) quantum gates; O(2**qubits) classical grid values",
                    note="Uniform amplitudes are prepared with one Hadamard per data wire.",
                )
            )
            preferred = "uniform_quantile_hadamard"
        else:
            preferred = "probability_tree_multiplexed_ry"
        candidates.append(
            _cost(
                strategy="probability_tree_multiplexed_ry",
                implemented=True,
                portable=True,
                data_qubits=representation.qubits,
                ancilla_qubits=ancilla_qubits,
                parameters=points - 1,
                stored_values=2 * points - 1,
                memory_bytes=(2 * points - 1) * 8,
                gates=points - 1,
                depth=points - 1,
                joint=False,
                available=True,
                target=target,
                max_parameters=max_parameters,
                max_memory_bytes=max_memory_bytes,
                scaling="O(2**qubits) angles and high-level multiplexed rotations",
                note="Generic exact probability-tree loader for the finite representation.",
            )
        )
        dense_entries = points * points
        candidates.append(
            _cost(
                strategy="dense_unitary_reference",
                implemented=True,
                portable=False,
                data_qubits=representation.qubits,
                ancilla_qubits=ancilla_qubits,
                parameters=points,
                stored_values=dense_entries,
                memory_bytes=dense_entries * 16,
                gates=max(0, 2 * points - 2),
                depth=max(0, 2 * points - 2),
                joint=False,
                available=True,
                target=target,
                max_parameters=max_parameters,
                max_memory_bytes=max_memory_bytes,
                scaling="O(4**qubits) dense complex matrix storage",
                note=(
                    "Numerical simulator reference; deliberately excluded from portable selection."
                ),
            )
        )
    else:
        data_qubits = representation.total_qubits
        marginal_parameters = sum(
            0
            if factor.state_preparation_method == "uniform_quantile_hadamard"
            else factor.grid_points - 1
            for factor in representation.factors
        )
        marginal_gates = sum(
            factor.qubits
            if factor.state_preparation_method == "uniform_quantile_hadamard"
            else factor.grid_points - 1
            for factor in representation.factors
        )
        marginal_depth = max(
            1
            if factor.state_preparation_method == "uniform_quantile_hadamard"
            else factor.grid_points - 1
            for factor in representation.factors
        )
        transform_values = 0
        if representation.transform is not None:
            transform_values = int(
                representation.transform.matrix.size + representation.transform.offset.size
            )
        stored_values = 2 * representation.stored_marginal_points + transform_values
        candidates.append(
            _cost(
                strategy="factorized_marginal_loader",
                implemented=True,
                portable=True,
                data_qubits=data_qubits,
                ancilla_qubits=ancilla_qubits,
                parameters=marginal_parameters,
                stored_values=stored_values,
                memory_bytes=stored_values * 8,
                gates=marginal_gates,
                depth=marginal_depth,
                joint=False,
                available=True,
                target=target,
                max_parameters=max_parameters,
                max_memory_bytes=max_memory_bytes,
                scaling="O(sum(2**factor_qubits)); no joint probability table",
                note=(
                    "Implemented independent-register loader. Affine factor transforms remain "
                    "classical interpretation metadata."
                ),
            )
        )
        preferred = "factorized_marginal_loader"
        joint_points = representation.joint_grid_points
        candidates.append(
            _cost(
                strategy="flattened_probability_tree",
                implemented=True,
                portable=True,
                data_qubits=data_qubits,
                ancilla_qubits=ancilla_qubits,
                parameters=joint_points - 1,
                stored_values=2 * joint_points - 1,
                memory_bytes=(2 * joint_points - 1) * 8,
                gates=joint_points - 1,
                depth=joint_points - 1,
                joint=True,
                available=joint_points <= max_flattened_points,
                target=target,
                max_parameters=max_parameters,
                max_memory_bytes=max_memory_bytes,
                scaling="O(product(2**factor_qubits)) joint angles",
                note="Generic fallback requiring explicit Cartesian-product materialization.",
            )
        )
        dense_entries = joint_points * joint_points
        candidates.append(
            _cost(
                strategy="dense_joint_unitary_reference",
                implemented=False,
                portable=False,
                data_qubits=data_qubits,
                ancilla_qubits=ancilla_qubits,
                parameters=joint_points,
                stored_values=dense_entries,
                memory_bytes=dense_entries * 16,
                gates=max(0, 2 * joint_points - 2),
                depth=max(0, 2 * joint_points - 2),
                joint=True,
                available=False,
                target=target,
                max_parameters=max_parameters,
                max_memory_bytes=max_memory_bytes,
                scaling="O(joint_grid_points**2) dense complex storage",
                note="Reported only to expose the rejected dense-memory cost.",
            )
        )

    selected = next(
        (
            candidate
            for candidate in candidates
            if candidate.strategy == preferred and candidate.selectable
        ),
        None,
    )
    if selected is None:
        target_text = "the requested target" if target is not None else "the configured limits"
        reason = f"no implemented portable state-preparation strategy satisfies {target_text}"
    else:
        reason = (
            f"selected {selected.strategy}: it preserves the representation and satisfies "
            "the explicit wire, parameter, and memory limits"
        )
    return StatePreparationStrategyReport(
        candidates=tuple(candidates),
        selected_strategy=None if selected is None else selected.strategy,
        target_name=None if target is None else target.name,
        target_topology=None if target is None else target.topology,
        max_parameters=max_parameters,
        max_memory_bytes=max_memory_bytes,
        selection_reason=reason,
    )


__all__ = [
    "Representation",
    "RepresentationTarget",
    "StatePreparationCost",
    "StatePreparationStrategyReport",
    "compare_state_preparation_strategies",
]
