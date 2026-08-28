"""Compare QFin's vectorized Walsh transform with the v0.3 loop structure."""

from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from qfin.circuits.walsh_payoff import _fast_walsh_hadamard


def reference_transform(values: NDArray[np.float64]) -> NDArray[np.float64]:
    transformed = values.copy()
    width = 1
    while width < transformed.size:
        for start in range(0, transformed.size, 2 * width):
            left = transformed[start : start + width].copy()
            right = transformed[start + width : start + 2 * width].copy()
            transformed[start : start + width] = left + right
            transformed[start + width : start + 2 * width] = left - right
        width *= 2
    return transformed


values = np.random.default_rng(7).normal(size=2**16)
started = perf_counter()
reference = reference_transform(values)
reference_seconds = perf_counter() - started
started = perf_counter()
optimized = _fast_walsh_hadamard(values)
optimized_seconds = perf_counter() - started

print(f"points={values.size:,}")
print(f"reference_seconds={reference_seconds:.6f}")
print(f"optimized_seconds={optimized_seconds:.6f}")
print(f"speedup={reference_seconds / optimized_seconds:.2f}x")
print(f"max_error={np.max(np.abs(reference - optimized)):.3e}")
