"""Measure existing implementation alternatives, including public API overhead.

Private-kernel microbenchmarks are labelled separately from public API timings.
Run with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""

from __future__ import annotations

import argparse
import json
import math
import tracemalloc
from collections.abc import Callable
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from native_benchmark import Measurement, _markdown, _seconds, _structured_report

import qfin
from qfin._numerics import stable_sum
from qfin.finance import fixed_income


def _bond(coupon: float, dated: bool) -> qfin.FixedRateBond:
    if dated:
        return qfin.FixedRateBond.from_dates(
            '2024-01-31', '2044-01-31', coupon, frequency=2, end_of_month=True,
        )
    return qfin.FixedRateBond(20, coupon, frequency=2)


def _old_par(curve: qfin.YieldCurve, dated: bool) -> float:
    zero = qfin.price_bonds(_bond(0, dated), curve, engine='numpy').clean_prices[0]
    unit = qfin.price_bonds(_bond(1, dated), curve, engine='numpy').clean_prices[0]
    return float((100-zero)/(unit-zero))


def _repeat_bincount(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    counts = np.diff(offsets)
    indices = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    return np.bincount(indices, weights=values, minlength=counts.size)


def _reduceat(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    output = np.zeros(offsets.size-1)
    valid = np.diff(offsets) > 0
    output[valid] = np.add.reduceat(values, offsets[:-1][valid])
    return output


def _prefix(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    prefix = np.concatenate(([0.], np.cumsum(values)))
    return prefix[offsets[1:]] - prefix[offsets[:-1]]


def _peak(function: Callable[[], object]) -> int:
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _kahan(values: np.ndarray) -> float:
    total = correction = 0.
    for item in values:
        adjusted = float(item)-correction
        updated = total+adjusted
        correction = (updated-total)-adjusted
        total = updated
    return total


def _neumaier(values: np.ndarray) -> float:
    total = correction = 0.
    for value in values:
        item = float(value)
        updated = total+item
        correction += (total-updated)+item if abs(total) >= abs(item) else (item-updated)+total
        total = updated
    return total+correction


def _naive(values: np.ndarray) -> float:
    total = 0.
    for value in values:
        total += float(value)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--json-output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    rows = []
    memory = []
    for dated in (False, True):
        curve = qfin.YieldCurve([0, 5, 30], [-.01, .03, .05],
                               valuation_date='2024-04-30' if dated else None)
        bond = _bond(0, dated)
        reference = partial(_old_par, curve, dated)
        current = partial(qfin.par_yield, bond, curve)
        rows.append(Measurement('Par yield', 'dated' if dated else 'floating',
            'Two derived bonds and full repricing', 'One schedule',
            _seconds(reference, args.repeats), _seconds(current, args.repeats),
            abs(reference()-current())))

    curve = qfin.YieldCurve([0, 30], [.01, .05])
    for size in (1, 100, 1_000, 10_000):
        bonds = [qfin.FixedRateBond(1+i%30, .04, frequency=(1, 2, 4)[i%3]) for i in range(size)]
        invoke = partial(qfin.price_bonds, bonds, curve, engine='numpy')
        with patch.object(fixed_income, '_segment_sum', _repeat_bincount):
            reference_value = invoke().dirty_prices
            reference_seconds = _seconds(invoke, args.repeats)
            reference_peak = _peak(invoke)
        for name, method in [
            ('reduceat', _reduceat), ('current dispatch', fixed_income._segment_sum),
            ('prefix differences', _prefix),
        ]:
            with patch.object(fixed_income, '_segment_sum', method):
                actual = invoke().dirty_prices
                seconds = _seconds(invoke, args.repeats)
                peak = _peak(invoke)
            rows.append(Measurement('Public bond reduction alternatives', f'{size:,} bonds',
                'repeat + bincount', name, reference_seconds, seconds,
                float(np.max(np.abs(actual-reference_value)))))
            memory.append(dict(problem=f'{size:,} bonds', method=name,
                               reference_peak_bytes=reference_peak, current_peak_bytes=peak,
                               scope='tracemalloc peak including tracked NumPy buffers; not RSS'))

    stress = np.tile([1e12/3, 1/3, -1e12/3], 10_000)
    expected = math.fsum(float(x) for x in stress)
    summations = []
    methods = [('naive Python', lambda: _naive(stress)),
               ('NumPy pairwise', lambda: float(np.sum(stress))),
               ('Kahan Python', lambda: _kahan(stress)),
               ('Neumaier Python', lambda: _neumaier(stress)),
               ('QFin selective fsum', lambda: stable_sum(stress))]
    for name, function in methods:
        summations.append(dict(method=name, seconds=_seconds(function, args.repeats),
                               value=function(), reference=expected,
                               absolute_error=abs(function()-expected),
                               scope='reduction microbenchmark; 30,000 mixed-sign values'))
    # Prefix differencing loses a one-unit segment following a trillion-scale prefix.
    cancellation = np.concatenate((np.full(10_000, 1e12), [1.]))
    offsets = np.array([0, 10_000, 10_001], dtype=np.int64)
    segmentation_stress = {name: method(cancellation, offsets).tolist() for name, method in
                           [('prefix', _prefix), ('reduceat', _reduceat),
                            ('repeat_bincount', _repeat_bincount)]}
    report = _structured_report(rows, args.repeats)
    report['allocation_measurements'] = memory
    report['summation_microbenchmarks'] = summations
    report['segmentation_cancellation'] = segmentation_stress
    report['notes'] = [
        'Public timings include schedule generation, input conversion and analytics.',
        'Candidate patching is outside the timed calls.',
        'Python Kahan/Neumaier costs are not claims about equivalent compiled loops.',
        'Prefix differences are rejected because they lose small later segments.',
        'Native outputs keep NumPy ownership; sanitizer CI tests their lifetime.',
    ]
    args.json_output.write_text(json.dumps(report, indent=2)+'\n')
    markdown = _markdown(rows, args.repeats)
    markdown += '\n## Cancellation microbenchmark\n\n'
    markdown += '| Method | Seconds | Absolute error |\n| --- | ---: | ---: |\n'
    for item in summations:
        markdown += f"| {item['method']} | {item['seconds']:.6f} | {item['absolute_error']:.6g} |\n"
    markdown += (
        '\nThe fsum reference is 3,333.333333333333 for the supplied float products. '
        'Private reduction timings are not public-API speedups. '
        'Allocation peaks and the prefix-cancellation counterexample are recorded in JSON.\n'
    )
    args.output.write_text(markdown)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
