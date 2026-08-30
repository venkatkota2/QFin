"""Mean-variance portfolio problems with validated classical baselines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from qfin.exceptions import OptimizationError

FloatArray = NDArray[np.float64]
OptimizationMethod = Literal["auto", "slsqp", "closed_form"]


@dataclass(frozen=True, slots=True)
class PortfolioOptimizationResult:
    """Validated weights and financial metrics from a classical solver."""

    weights: FloatArray
    asset_names: Sequence[str]
    expected_return: float
    variance: float
    volatility: float
    utility: float
    baseline_weights: FloatArray
    baseline_utility: float
    utility_improvement: float
    budget_residual: float
    target_return_residual: float | None
    solver: str
    success: bool
    iterations: int
    message: str

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        baseline = np.asarray(self.baseline_weights, dtype=np.float64).reshape(-1)
        names = tuple(self.asset_names)
        if weights.shape != baseline.shape or weights.size != len(names):
            raise ValueError("weights, baseline_weights, and asset_names must align")
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(baseline)):
            raise ValueError("optimization weights must be finite")
        weights.setflags(write=False)
        baseline.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "baseline_weights", baseline)
        object.__setattr__(self, "asset_names", names)

    def to_dict(self) -> dict[str, object]:
        return {
            "weights": self.weights.tolist(),
            "asset_names": list(self.asset_names),
            "expected_return": self.expected_return,
            "variance": self.variance,
            "volatility": self.volatility,
            "utility": self.utility,
            "baseline_weights": self.baseline_weights.tolist(),
            "baseline_utility": self.baseline_utility,
            "utility_improvement": self.utility_improvement,
            "budget_residual": self.budget_residual,
            "target_return_residual": self.target_return_residual,
            "solver": self.solver,
            "success": self.success,
            "iterations": self.iterations,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class MeanVarianceProblem:
    """Continuous mean-variance allocation with explicit constraints."""

    expected_returns: FloatArray
    covariance: FloatArray
    risk_aversion: float = 1.0
    asset_names: Sequence[str] | None = None
    budget: float = 1.0
    long_only: bool = True
    lower_bounds: ArrayLike | None = None
    upper_bounds: ArrayLike | None = None
    target_return: float | None = None

    def __post_init__(self) -> None:
        expected_returns = np.asarray(self.expected_returns, dtype=np.float64).reshape(-1)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        asset_count = expected_returns.size
        if asset_count < 2 or not np.all(np.isfinite(expected_returns)):
            raise ValueError("expected_returns must contain at least two finite values")
        if covariance.shape != (asset_count, asset_count):
            raise ValueError("covariance must be square with one row per asset")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("covariance must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
            raise ValueError("covariance must be symmetric")
        eigenvalues = np.linalg.eigvalsh(covariance)
        if float(np.min(eigenvalues)) < -1e-10:
            raise ValueError("covariance must be positive semidefinite")
        if not isfinite(self.risk_aversion) or self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be finite and positive")
        if not isfinite(self.budget) or self.budget <= 0:
            raise ValueError("budget must be finite and positive")
        if self.target_return is not None and not isfinite(self.target_return):
            raise ValueError("target_return must be finite")

        names = (
            tuple(f"asset_{index}" for index in range(asset_count))
            if self.asset_names is None
            else tuple(self.asset_names)
        )
        if len(names) != asset_count or not all(names):
            raise ValueError("asset_names must contain one non-empty name per asset")
        if len(set(names)) != len(names):
            raise ValueError("asset_names must be unique")

        default_lower = 0.0 if self.long_only else -np.inf
        default_upper = self.budget if self.long_only else np.inf
        lower = self._bounds_array(self.lower_bounds, default_lower, asset_count, "lower_bounds")
        upper = self._bounds_array(self.upper_bounds, default_upper, asset_count, "upper_bounds")
        if np.any(lower > upper):
            raise ValueError("lower_bounds cannot exceed upper_bounds")
        if np.all(np.isfinite(lower)) and float(np.sum(lower)) > self.budget + 1e-12:
            raise ValueError("lower_bounds exceed the available budget")
        if np.all(np.isfinite(upper)) and float(np.sum(upper)) < self.budget - 1e-12:
            raise ValueError("upper_bounds cannot satisfy the budget")
        if self.long_only and (np.any(lower < 0) or np.any(upper < 0)):
            raise ValueError("long_only bounds must be non-negative")

        expected_returns = np.ascontiguousarray(expected_returns)
        covariance = np.ascontiguousarray(covariance)
        lower = np.ascontiguousarray(lower)
        upper = np.ascontiguousarray(upper)
        for values in (expected_returns, covariance, lower, upper):
            values.setflags(write=False)
        object.__setattr__(self, "expected_returns", expected_returns)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "asset_names", names)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)

    @staticmethod
    def _bounds_array(
        values: ArrayLike | None,
        default: float,
        size: int,
        name: str,
    ) -> FloatArray:
        if values is None:
            return np.full(size, default, dtype=np.float64)
        array = np.asarray(values, dtype=np.float64)
        if array.ndim == 0:
            array = np.full(size, float(array), dtype=np.float64)
        else:
            array = array.reshape(-1)
        if array.shape != (size,) or np.any(np.isnan(array)):
            raise ValueError(f"{name} must contain one non-NaN value per asset")
        return array

    @property
    def asset_count(self) -> int:
        return int(self.expected_returns.size)

    def expected_portfolio_return(self, weights: ArrayLike) -> float:
        values = self._validated_weights(weights)
        return float(np.dot(values, self.expected_returns))

    def portfolio_variance(self, weights: ArrayLike) -> float:
        values = self._validated_weights(weights)
        return float(values @ self.covariance @ values)

    def utility(self, weights: ArrayLike) -> float:
        values = self._validated_weights(weights)
        expected_return = float(np.dot(values, self.expected_returns))
        variance = float(values @ self.covariance @ values)
        return expected_return - 0.5 * self.risk_aversion * variance

    def _validated_weights(self, weights: ArrayLike) -> FloatArray:
        values = np.asarray(weights, dtype=np.float64).reshape(-1)
        if values.shape != (self.asset_count,) or not np.all(np.isfinite(values)):
            raise ValueError("weights must contain one finite value per asset")
        return values

    def _initial_weights(self) -> FloatArray:
        assert isinstance(self.lower_bounds, np.ndarray)
        assert isinstance(self.upper_bounds, np.ndarray)
        weights = np.full(self.asset_count, self.budget / self.asset_count, dtype=np.float64)
        weights = np.maximum(weights, self.lower_bounds)
        weights = np.minimum(weights, self.upper_bounds)
        for _ in range(self.asset_count + 1):
            residual = self.budget - float(np.sum(weights))
            if abs(residual) <= 1e-12:
                return weights
            room = self.upper_bounds - weights if residual > 0 else weights - self.lower_bounds
            eligible = room > 1e-14
            if not np.any(eligible):
                break
            finite_room = np.where(np.isfinite(room), room, 1.0)
            allocation = finite_room * eligible
            allocation /= float(np.sum(allocation))
            change = min(abs(residual), float(np.sum(finite_room[eligible])))
            weights += np.sign(residual) * change * allocation
        if abs(float(np.sum(weights)) - self.budget) > 1e-9:
            raise OptimizationError("could not construct a feasible initial budget allocation")
        return weights

    def _result(
        self,
        weights: FloatArray,
        baseline: FloatArray,
        *,
        solver: str,
        iterations: int,
        message: str,
    ) -> PortfolioOptimizationResult:
        assert isinstance(self.lower_bounds, np.ndarray)
        assert isinstance(self.upper_bounds, np.ndarray)
        tolerance = 1e-7
        if np.any(weights < self.lower_bounds - tolerance) or np.any(
            weights > self.upper_bounds + tolerance
        ):
            raise OptimizationError("solver returned weights outside the configured bounds")
        budget_residual = float(np.sum(weights) - self.budget)
        expected_return = self.expected_portfolio_return(weights)
        target_residual = (
            None if self.target_return is None else expected_return - self.target_return
        )
        if abs(budget_residual) > tolerance:
            raise OptimizationError("solver returned a portfolio outside the budget constraint")
        if target_residual is not None and target_residual < -tolerance:
            raise OptimizationError("solver returned a portfolio below the target return")
        variance = max(self.portfolio_variance(weights), 0.0)
        utility = self.utility(weights)
        baseline_utility = self.utility(baseline)
        return PortfolioOptimizationResult(
            weights=weights,
            asset_names=self.asset_names or (),
            expected_return=expected_return,
            variance=variance,
            volatility=sqrt(variance),
            utility=utility,
            baseline_weights=baseline,
            baseline_utility=baseline_utility,
            utility_improvement=utility - baseline_utility,
            budget_residual=budget_residual,
            target_return_residual=target_residual,
            solver=solver,
            success=True,
            iterations=iterations,
            message=message,
        )

    def _solve_closed_form(self) -> PortfolioOptimizationResult:
        assert isinstance(self.lower_bounds, np.ndarray)
        assert isinstance(self.upper_bounds, np.ndarray)
        if np.any(np.isfinite(self.lower_bounds)) or np.any(np.isfinite(self.upper_bounds)):
            raise ValueError("closed_form requires unbounded weights")
        if self.target_return is not None:
            raise ValueError("closed_form does not support target_return")
        inverse = np.linalg.pinv(self.covariance, hermitian=True)
        ones = np.ones(self.asset_count, dtype=np.float64)
        denominator = float(ones @ inverse @ ones)
        if denominator <= 0:
            raise OptimizationError("covariance does not support the budget constraint")
        multiplier = (
            float(ones @ inverse @ self.expected_returns) - self.risk_aversion * self.budget
        ) / denominator
        weights = inverse @ (self.expected_returns - multiplier * ones) / self.risk_aversion
        baseline = self._initial_weights()
        return self._result(
            np.asarray(weights, dtype=np.float64),
            baseline,
            solver="closed_form_equality_constrained_mean_variance",
            iterations=1,
            message="analytical equality-constrained solution",
        )

    def _solve_slsqp(self) -> PortfolioOptimizationResult:
        assert isinstance(self.lower_bounds, np.ndarray)
        assert isinstance(self.upper_bounds, np.ndarray)
        initial = self._initial_weights()

        def objective(weights: FloatArray) -> float:
            return -self.utility(weights)

        def gradient(weights: FloatArray) -> FloatArray:
            return np.asarray(
                self.risk_aversion * (self.covariance @ weights) - self.expected_returns,
                dtype=np.float64,
            )

        constraints: list[dict[str, Any]] = [
            {
                "type": "eq",
                "fun": lambda weights: float(np.sum(weights) - self.budget),
                "jac": lambda weights: np.ones_like(weights),
            }
        ]
        if self.target_return is not None:
            target = self.target_return
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights: float(weights @ self.expected_returns - target),
                    "jac": lambda weights: self.expected_returns,
                }
            )
        bounds = list(zip(self.lower_bounds, self.upper_bounds, strict=True))
        result: Any = minimize(
            objective,
            initial,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
        )
        if not bool(result.success):
            raise OptimizationError(f"SLSQP failed: {result.message}")
        weights = np.asarray(result.x, dtype=np.float64)
        return self._result(
            weights,
            initial,
            solver="scipy_slsqp_continuous_mean_variance",
            iterations=int(result.nit),
            message=str(result.message),
        )

    def solve(self, *, method: OptimizationMethod = "auto") -> PortfolioOptimizationResult:
        """Run a deterministic continuous classical baseline."""

        if method not in ("auto", "slsqp", "closed_form"):
            raise ValueError("method must be 'auto', 'slsqp', or 'closed_form'")
        assert isinstance(self.lower_bounds, np.ndarray)
        assert isinstance(self.upper_bounds, np.ndarray)
        unbounded = not np.any(np.isfinite(self.lower_bounds)) and not np.any(
            np.isfinite(self.upper_bounds)
        )
        if method == "closed_form" or (
            method == "auto" and unbounded and self.target_return is None
        ):
            return self._solve_closed_form()
        return self._solve_slsqp()


__all__ = [
    "MeanVarianceProblem",
    "OptimizationMethod",
    "PortfolioOptimizationResult",
]
