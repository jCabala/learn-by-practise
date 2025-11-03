"""
Compute energy from gem5 result file placed in `part_3/best/result` and print to terminal.

Energy (J) = (Runtime Dynamic + Subthreshold Leakage + Gate Leakage) * Simulated seconds

This script looks for a `result` or `results` file in the same directory as the script
and parses the required fields. If any field is missing it will report which ones.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Optional, Tuple


ROOT = os.path.dirname(__file__)
RESULT_PATHS = [
	os.path.join(ROOT, 'result'),
	os.path.join(ROOT, 'results'),
	os.path.join(ROOT, 'gem5.out', 'stats.txt'),
]


def find_result_file() -> Optional[str]:
	for p in RESULT_PATHS:
		if os.path.isfile(p):
			return p
	return None


def parse_metrics(text: str) -> Optional[Tuple[float, float, float, float]]:
	def f(pattern: str):
		m = re.search(pattern, text)
		return float(m.group(1)) if m else None

	sim = f(r"Simulated seconds\s*=\s*([0-9.eE+\-]+)")
	runtime = f(r"Runtime Dynamic\s*=\s*([0-9.eE+\-]+)\s*W")
	sub = f(r"Subthreshold Leakage\s*=\s*([0-9.eE+\-]+)\s*W")
	gate = f(r"Gate Leakage\s*=\s*([0-9.eE+\-]+)\s*W")

	if None in (sim, runtime, sub, gate):
		return None
	return sim, runtime, sub, gate


def main() -> int:
	path = find_result_file()
	if path is None:
		print('No result file found. Looked for:', ', '.join(RESULT_PATHS))
		return 2

	with open(path, 'r', encoding='utf-8', errors='ignore') as f:
		txt = f.read()

	parsed = parse_metrics(txt)
	if parsed is None:
		print('Could not parse all required metrics from', path)
		print('Found contents (first 200 chars):')
		print(txt[:200])
		return 3

	sim, runtime, sub, gate = parsed
	total_power = runtime + sub + gate
	energy = total_power * sim

	print('Parsed metrics from', path)
	print(f"Simulated seconds: {sim:.6g}")
	print(f"Runtime Dynamic (W): {runtime:.6g}")
	print(f"Subthreshold Leakage (W): {sub:.6g}")
	print(f"Gate Leakage (W): {gate:.6g}")
	print(f"Total power estimate (W): {total_power:.6g}")
	print(f"Estimated energy (J): {energy:.6g}")

	return 0


if __name__ == '__main__':
	raise SystemExit(main())
