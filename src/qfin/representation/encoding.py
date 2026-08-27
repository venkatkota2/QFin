"""Automatic financial-distribution representations."""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from qfin.finance.distributions import Distribution, EmpiricalDistribution

Objective = Literal["expectation"] | Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class DistributionEncoding:
    """A finite probability representation with accuracy metadata."""

    grid: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    qubits: int
    lower_bound: float
    upper_bound: float
    tail_probability: float
    discretization_error: float
    mean_error: float
    objective: str
    encoding_method: str = "probability_amplitude"
    state_preparation_method: str = "probability_tree_multiplexed_ry"

    def __post_init__(self) -> None:
        grid = np.asarray(self.grid, dtype=np.float64).reshape(-1)
        probabilities = np.asarray(self.probabilities, dtype=np.float64).reshape(-1)
        if grid.shape != probabilities.shape or grid.size != 2**self.qubits:
            raise ValueError("grid and probabilities must each contain 2**qubits values")
        if np.any(probabilities < 0) or not np.isclose(np.sum(probabilities), 1.0):
            raise ValueError("probabilities must be non-negative and sum to one")
        grid.setflags(write=False)
        probabilities.setflags(write=False)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "probabilities", probabilities)

    @property
    def grid_points(self) -> int:
        return int(self.grid.size)

    @property
    def conditional_mean(self) -> float:
        return float(np.dot(self.grid, self.probabilities))

    def state_vector(self) -> NDArray[np.float64]:
        """Return amplitudes whose squared magnitudes equal the probabilities."""
        return np.sqrt(self.probabilities)

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "encoding_method": self.encoding_method,
            "state_preparation_method": self.state_preparation_method,
            "qubits": self.qubits,
            "grid_points": self.grid_points,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "tail_probability": self.tail_probability,
            "discretization_error": self.discretization_error,
            "mean_error": self.mean_error,
            "objective": self.objective,
        }


def _objective_values(
    objective: Objective, grid: NDArray[np.float64]
) -> NDArray[np.float64]:
    if objective == "expectation":
        return grid
    if callable(objective):
        values = np.asarray(objective(grid), dtype=np.float64)
        if values.shape != grid.shape or not np.all(np.isfinite(values)):
            raise ValueError("objective callable must return one finite value per grid point")
        return values
    raise ValueError("objective must be 'expectation' or a callable")


def _domain_bounds(
    distribution: Distribution,
    tail_probability: float,
    bounds: tuple[float, float] | None,
) -> tuple[float, float]:
    if bounds is None:
        quantiles = np.asarray(
            distribution.ppf([tail_probability / 2, 1 - tail_probability / 2]),
            dtype=np.float64,
        )
        lower, upper = float(quantiles[0]), float(quantiles[1])
    else:
        lower, upper = float(bounds[0]), float(bounds[1])
    if not (isfinite(lower) and isfinite(upper)):
        raise ValueError("distribution bounds must be finite")
    if lower == upper:
        padding = max(1.0, abs(lower)) * 1e-6
        lower, upper = lower - padding, upper + padding
    if lower > upper:
        raise ValueError("lower bound must be less than upper bound")
    return lower, upper


def _empirical_grid(
    distribution: EmpiricalDistribution,
    edges: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    assert distribution.probabilities is not None
    included = (distribution.values >= edges[0]) & (distribution.values <= edges[-1])
    included_values = distribution.values[included]
    included_probabilities = distribution.probabilities[included]
    omitted_mass = float(1.0 - np.sum(included_probabilities))
    bin_index = np.searchsorted(edges, included_values, side="right") - 1
    bin_index = np.clip(bin_index, 0, edges.size - 2)
    probabilities = np.zeros(edges.size - 1, dtype=np.float64)
    weighted_values = np.zeros(edges.size - 1, dtype=np.float64)
    np.add.at(probabilities, bin_index, included_probabilities)
    np.add.at(weighted_values, bin_index, included_probabilities * included_values)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    grid = midpoints.copy()
    occupied = probabilities > 0
    grid[occupied] = weighted_values[occupied] / probabilities[occupied]
    return grid, probabilities, omitted_mass


def _fixed_encoding(
    distribution: Distribution,
    *,
    qubits: int,
    tail_probability: float,
    bounds: tuple[float, float] | None,
    objective: Objective,
    convergence_error: float,
) -> DistributionEncoding:
    lower, upper = _domain_bounds(distribution, tail_probability, bounds)
    edges = np.linspace(lower, upper, 2**qubits + 1, dtype=np.float64)

    if isinstance(distribution, EmpiricalDistribution):
        grid, probabilities, omitted_mass = _empirical_grid(distribution, edges)
    else:
        cdf_values = np.asarray(distribution.cdf(edges), dtype=np.float64)
        probabilities = np.maximum(np.diff(cdf_values), 0.0)
        omitted_mass = float(np.clip(cdf_values[0] + 1.0 - cdf_values[-1], 0.0, 1.0))
        grid = 0.5 * (edges[:-1] + edges[1:])

    included_mass = float(np.sum(probabilities))
    if included_mass <= 0:
        raise ValueError("selected domain contains no probability mass")
    probabilities = probabilities / included_mass
    mean_error = abs(float(np.dot(grid, probabilities)) - distribution.mean)
    label = "expectation" if objective == "expectation" else "callable"
    return DistributionEncoding(
        grid=grid,
        probabilities=probabilities,
        qubits=qubits,
        lower_bound=lower,
        upper_bound=upper,
        tail_probability=omitted_mass,
        discretization_error=convergence_error,
        mean_error=mean_error,
        objective=label,
    )


def encode(
    distribution: Distribution,
    *,
    target_error: float = 1e-3,
    objective: Objective = "expectation",
    qubits: int | None = None,
    min_qubits: int = 3,
    max_qubits: int = 10,
    tail_probability: float = 1e-6,
    bounds: tuple[float, float] | None = None,
) -> DistributionEncoding:
    """Compile a distribution into a finite probability-amplitude encoding.

    Automatic qubit selection compares the objective expectation on successive
    grids. ``target_error`` is expressed in the objective's output units.
    """

    if not isfinite(target_error) or target_error <= 0:
        raise ValueError("target_error must be finite and greater than zero")
    if not 0 <= tail_probability < 1:
        raise ValueError("tail_probability must lie in [0, 1)")
    if min_qubits < 1 or max_qubits < min_qubits:
        raise ValueError("require 1 <= min_qubits <= max_qubits")
    if qubits is not None and not min_qubits <= qubits <= max_qubits:
        raise ValueError("qubits must lie between min_qubits and max_qubits")

    if qubits is not None:
        return _fixed_encoding(
            distribution,
            qubits=qubits,
            tail_probability=tail_probability,
            bounds=bounds,
            objective=objective,
            convergence_error=0.0,
        )

    previous_value: float | None = None
    selected: DistributionEncoding | None = None
    for candidate in range(min_qubits, max_qubits + 1):
        encoding = _fixed_encoding(
            distribution,
            qubits=candidate,
            tail_probability=tail_probability,
            bounds=bounds,
            objective=objective,
            convergence_error=float("inf"),
        )
        values = _objective_values(objective, encoding.grid)
        current_value = float(np.dot(encoding.probabilities, values))
        if previous_value is not None:
            convergence = abs(current_value - previous_value)
            encoding = DistributionEncoding(
                grid=encoding.grid,
                probabilities=encoding.probabilities,
                qubits=encoding.qubits,
                lower_bound=encoding.lower_bound,
                upper_bound=encoding.upper_bound,
                tail_probability=encoding.tail_probability,
                discretization_error=convergence,
                mean_error=encoding.mean_error,
                objective=encoding.objective,
            )
            if convergence <= target_error:
                return encoding
        previous_value = current_value
        selected = encoding

    assert selected is not None
    return selected


def _fixed_quantile_encoding(
    distribution: Distribution,
    *,
    qubits: int,
    tail_probability: float,
    objective: Objective,
    convergence_error: float,
) -> DistributionEncoding:
    """Build midpoint quadrature in probability space.

    Every point has equal weight, so the data register is prepared with one
    Hadamard per qubit. The nonlinear inverse CDF is represented by the
    classical interpretation of each basis state instead of by state-loading
    angles.
    """

    lower_quantile = tail_probability / 2.0
    upper_quantile = 1.0 - lower_quantile
    points = 2**qubits
    quantiles = lower_quantile + (
        (np.arange(points, dtype=np.float64) + 0.5)
        * (upper_quantile - lower_quantile)
        / points
    )
    grid = np.asarray(distribution.ppf(quantiles), dtype=np.float64)
    if grid.shape != (points,) or not np.all(np.isfinite(grid)):
        raise ValueError("distribution ppf must return one finite value per quantile")
    probabilities = np.full(points, 1.0 / points, dtype=np.float64)
    domain = np.asarray(
        distribution.ppf([lower_quantile, upper_quantile]), dtype=np.float64
    )
    if not np.all(np.isfinite(domain)):
        raise ValueError("quantile domain bounds must be finite")
    label = "expectation" if objective == "expectation" else "callable"
    return DistributionEncoding(
        grid=grid,
        probabilities=probabilities,
        qubits=qubits,
        lower_bound=float(domain[0]),
        upper_bound=float(domain[1]),
        tail_probability=tail_probability,
        discretization_error=convergence_error,
        mean_error=abs(float(np.mean(grid)) - distribution.mean),
        objective=label,
        encoding_method="inverse_cdf_quantile",
        state_preparation_method="uniform_quantile_hadamard",
    )


def encode_quantiles(
    distribution: Distribution,
    *,
    target_error: float = 1e-3,
    objective: Objective = "expectation",
    qubits: int | None = None,
    min_qubits: int = 3,
    max_qubits: int = 12,
    tail_probability: float = 1e-6,
) -> DistributionEncoding:
    """Encode a distribution through equal-probability inverse-CDF points.

    The computational basis is interpreted as midpoint quantiles of the
    selected distribution. Consequently the quantum probabilities are
    uniform and require only ``qubits`` Hadamard gates, rather than one
    probability-loading parameter per grid point. Automatic qubit selection
    compares the requested financial objective on successive quadratures.
    """

    if not isfinite(target_error) or target_error <= 0:
        raise ValueError("target_error must be finite and greater than zero")
    if not 0 < tail_probability < 1:
        raise ValueError("tail_probability must lie strictly between zero and one")
    if min_qubits < 1 or max_qubits < min_qubits:
        raise ValueError("require 1 <= min_qubits <= max_qubits")
    if qubits is not None and not min_qubits <= qubits <= max_qubits:
        raise ValueError("qubits must lie between min_qubits and max_qubits")

    if qubits is not None:
        return _fixed_quantile_encoding(
            distribution,
            qubits=qubits,
            tail_probability=tail_probability,
            objective=objective,
            convergence_error=0.0,
        )

    previous_value: float | None = None
    selected: DistributionEncoding | None = None
    for candidate in range(min_qubits, max_qubits + 1):
        encoding = _fixed_quantile_encoding(
            distribution,
            qubits=candidate,
            tail_probability=tail_probability,
            objective=objective,
            convergence_error=float("inf"),
        )
        values = _objective_values(objective, encoding.grid)
        current_value = float(np.mean(values))
        if previous_value is not None:
            convergence = abs(current_value - previous_value)
            encoding = DistributionEncoding(
                grid=encoding.grid,
                probabilities=encoding.probabilities,
                qubits=encoding.qubits,
                lower_bound=encoding.lower_bound,
                upper_bound=encoding.upper_bound,
                tail_probability=encoding.tail_probability,
                discretization_error=convergence,
                mean_error=encoding.mean_error,
                objective=encoding.objective,
                encoding_method=encoding.encoding_method,
                state_preparation_method=encoding.state_preparation_method,
            )
            if convergence <= target_error:
                return encoding
        previous_value = current_value
        selected = encoding

    assert selected is not None
    return selected
