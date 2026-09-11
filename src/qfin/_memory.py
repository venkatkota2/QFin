"""Cheap deterministic allocation preflight; no machine-specific calibration."""

from __future__ import annotations

from collections.abc import Sequence

from qfin.exceptions import ResourceLimitError

MAX_OPERATION_BYTES = 512 * 1024 * 1024


def check_allocation(shape: Sequence[int], *, arrays: int = 1, itemsize: int = 8) -> int:
    """Bound persistent buffers before allocation; inputs and runtime RSS are separate."""
    count = 1
    for dimension in shape:
        if dimension < 0:
            raise ResourceLimitError("allocation dimensions must be non-negative")
        count *= int(dimension)
    if count * arrays * itemsize > MAX_OPERATION_BYTES:
        raise ResourceLimitError("buffers exceed the 512 MiB operation allocation limit")
    return count
