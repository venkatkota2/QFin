"""Shared low-level validation and immutable-array ownership helpers."""

from __future__ import annotations

from collections.abc import Iterable
from operator import index as integer_index
from typing import SupportsIndex, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from qfin.exceptions import QFinValidationError


def require_integer(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an exact integer without accepting booleans or truncating floats."""

    if isinstance(value, bool):
        raise QFinValidationError(f"{name} must be an integer")
    try:
        result = integer_index(cast(SupportsIndex, value))
    except TypeError as exc:
        raise QFinValidationError(f"{name} must be an integer") from exc
    if minimum is not None and result < minimum:
        if maximum is not None:
            qualifier = f"in [{minimum}, {maximum}]"
        elif minimum == 0:
            qualifier = "non-negative"
        elif minimum == 1:
            qualifier = "positive"
        else:
            qualifier = f"at least {minimum}"
        raise QFinValidationError(f"{name} must be {qualifier}")
    if maximum is not None and result > maximum:
        qualifier = f"at most {maximum}" if minimum is None else f"in [{minimum}, {maximum}]"
        raise QFinValidationError(f"{name} must be {qualifier}")
    return result


def require_integer_sequence(
    values: Iterable[object],
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> tuple[int, ...]:
    """Return exact integers from an iterable without silently truncating values."""

    try:
        items = tuple(values)
    except TypeError as exc:
        raise QFinValidationError(f"{name} must be an iterable of integers") from exc
    return tuple(
        require_integer(
            value,
            f"{name}[{position}]",
            minimum=minimum,
            maximum=maximum,
        )
        for position, value in enumerate(items)
    )


def readonly_float64(value: ArrayLike) -> NDArray[np.float64]:
    """Return an owned, C-contiguous, immutable float64 array."""

    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result


def readonly_int64(value: ArrayLike) -> NDArray[np.int64]:
    """Return an owned, C-contiguous, immutable int64 array."""

    result = np.array(value, dtype=np.int64, order="C", copy=True)
    result.setflags(write=False)
    return result


def readonly_int32(value: ArrayLike) -> NDArray[np.int32]:
    """Return an owned, C-contiguous, immutable int32 array."""

    result = np.array(value, dtype=np.int32, order="C", copy=True)
    result.setflags(write=False)
    return result


def readonly_bool(value: ArrayLike) -> NDArray[np.bool_]:
    """Return an owned, C-contiguous, immutable boolean array."""

    result = np.array(value, dtype=np.bool_, order="C", copy=True)
    result.setflags(write=False)
    return result


__all__ = [
    "readonly_bool",
    "readonly_float64",
    "readonly_int32",
    "readonly_int64",
    "require_integer",
    "require_integer_sequence",
]
