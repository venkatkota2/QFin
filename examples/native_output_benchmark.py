"""Compare private native output construction with an installed baseline extension.

Supply the extension from a wheel built at the audited main commit. This measures
one batched Python/native call including output allocation, not a public valuation
API speedup. Full public API comparisons live in native_benchmark.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from functools import partial
from pathlib import Path

import numpy as np
from native_benchmark import Measurement, _markdown, _seconds, _structured_report

import qfin


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-extension', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--json-output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    spec = importlib.util.spec_from_file_location('_qfin_native', args.baseline_extension)
    if spec is None or spec.loader is None:
        parser.error('baseline extension could not be loaded')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    current = qfin._native.require()
    rows = []
    for count in (100_000, 1_000_000):
        inputs = (np.array([0.]), np.array([100.]), np.array([0, 1], dtype=np.int64),
                  np.array([1.]), np.array([0.]), np.array([0.]), np.zeros((count, 1)))
        old = partial(baseline.scenario_portfolio_present_values, *inputs)
        new = partial(current.scenario_portfolio_present_values, *inputs)
        reference = old()
        actual = new()
        assert actual.flags.owndata and actual.flags.c_contiguous
        np.testing.assert_array_equal(actual, np.full(count, 100.))
        rows.append(Measurement('Private native output investigation',
            f'{count:,} scenarios x one zero-time cash flow',
            'Audited-main vector then array', 'Current NumPy-owned output',
            _seconds(old, args.repeats), _seconds(new, args.repeats),
            float(np.max(np.abs(actual-reference)))))
    report = _structured_report(rows, args.repeats)
    report['baseline_commit'] = '3dba3b3803f54891d12ad2c138b4dcc276de9d42'
    report['interpretation'] = (
        'Private batched-call investigation, not a public API speedup. Current code also '
        'validates stressed-node discounts and output finiteness. The difference cannot '
        'be attributed entirely to output-copy removal. Retained change saves one '
        'scenario-length vector and copy; no capsule ownership transfer is used.'
    )
    args.json_output.write_text(json.dumps(report, indent=2)+'\n')
    args.output.write_text(_markdown(rows, args.repeats)+'\n'+report['interpretation']+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
