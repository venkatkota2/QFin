"""Enforce targeted numerical line-coverage floors from pytest-cov JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FLOORS = {
    'finance': 95., 'compiler': 92., 'representation': 92.,
    'resources': 90., 'backends': 85., '_native': 95.,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    failed = False
    for section, floor in FLOORS.items():
        summaries = [item['summary'] for name, item in data['files'].items()
                     if f'/qfin/{section}/' in name.replace('\\', '/')]
        total = sum(item['num_statements'] for item in summaries)
        covered = sum(item['covered_lines'] for item in summaries)
        percentage = 0. if total == 0 else 100*covered/total
        passed = total > 0 and percentage >= floor
        print(f'{section}: {percentage:.2f}% (required {floor:.0f}%) '
              f"{'PASS' if passed else 'FAIL'}")
        failed = failed or not passed
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
