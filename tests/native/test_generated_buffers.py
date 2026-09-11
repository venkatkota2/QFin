"""Generated native buffers, with scalar exact sums and invalid offset rejection."""

from math import fsum

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import qfin
from qfin import _native

pytestmark = pytest.mark.skipif(not _native.available(), reason="native extension unavailable")


@settings(max_examples=100, deadline=None, derandomize=True)
@given(st.lists(st.integers(-1_000_000, 1_000_000), min_size=0, max_size=100), st.integers(0, 100))
def test_generated_native_segments_and_empty_streams(amounts, split):
    cut = min(split, len(amounts))
    offsets = np.array([0, cut, cut, len(amounts)], dtype=np.int64)
    values = np.array(amounts, dtype=np.float32)
    values.setflags(write=False)
    result = _native.require().price_cashflow_batches(
        np.ones(len(values)), values, offsets, [0, 2], [0, 0]
    )
    np.testing.assert_array_equal(result["prices"], [fsum(amounts[:cut]), 0, fsum(amounts[cut:])])
    np.testing.assert_array_equal(values, amounts)
    assert not values.flags.writeable


@settings(max_examples=100, deadline=None, derandomize=True)
@given(st.lists(st.integers(-(2**63), 2**63 - 1), min_size=0, max_size=20))
def test_generated_arbitrary_offsets_cannot_escape_bounds(offsets):
    # Empty cashflow buffers admit only nondecreasing all-zero offsets.
    valid = len(offsets) >= 1 and all(value == 0 for value in offsets)
    if valid:
        result = _native.require().price_cashflow_batches([], [], offsets, [0], [0])
        assert len(result["prices"]) == len(offsets) - 1
    else:
        with pytest.raises(qfin.QFinValidationError):
            _native.require().price_cashflow_batches(
                [], [], np.array(offsets, dtype=np.int64), [0], [0]
            )
