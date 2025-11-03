"""
Parse gem5 output under `results/out` and plot energy (J) vs number of vecing-point physical registers.

How it computes energy:
 - Finds the fields `Simulated seconds`, `Runtime Dynamic`, `Subthreshold Leakage`, and `Gate Leakage` (all in logs/results files).
 - Estimates total power = Runtime Dynamic + Subthreshold Leakage + Gate Leakage (W)
 - Energy (J) = total_power * simulated_seconds

Usage: run this script from the `part_3/vec_regs` directory with the same Python you use to run other scripts.
It will write `results/plots/vec_regs_energy.png` and a CSV of values.
"""

from __future__ import annotations

import os
import re
import csv
from typing import Dict, Tuple, Optional

import matplotlib.pyplot as plt


ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "results", "out")
PLOTS_DIR = os.path.join(ROOT, "results", "plots")


def parse_metrics_from_text(text: str) -> Optional[Tuple[float, float, float, float]]:
	"""Return (sim_seconds, runtime_dynamic_W, subthreshold_W, gate_W) or None if missing."""
	def find_float(pattern: str):
		m = re.search(pattern, text)
		return float(m.group(1)) if m else None

	sim = find_float(r"Simulated seconds\s*=\s*([0-9.eE+\-]+)")
	runtime = find_float(r"Runtime Dynamic\s*=\s*([0-9.eE+\-]+)\s*W")
	sub = find_float(r"Subthreshold Leakage\s*=\s*([0-9.eE+\-]+)\s*W")
	gate = find_float(r"Gate Leakage\s*=\s*([0-9.eE+\-]+)\s*W")

	if None in (sim, runtime, sub, gate):
		return None
	return sim, runtime, sub, gate


def extract_reg_count(name: str) -> Optional[int]:
	"""Extract the numeric register count from names like VECREGS_128 or 5_VECREGS_128.log"""
	m = re.search(r"VECREGS_([0-9]+)", name)
	if m:
		return int(m.group(1))
	return None


def collect_energy_values(out_dir: str) -> Dict[int, float]:
	"""Scan files and directories under out_dir and return mapping reg_count -> energy_joules."""
	values: Dict[int, float] = {}

	# iterate entries
	for entry in sorted(os.listdir(out_dir)):
		path = os.path.join(out_dir, entry)

		# try top-level .log files
		if os.path.isfile(path) and entry.endswith('.log'):
			with open(path, 'r', encoding='utf-8', errors='ignore') as f:
				txt = f.read()
			regs = extract_reg_count(entry)
			parsed = parse_metrics_from_text(txt)
			if regs is not None and parsed is not None:
				sim, runtime, sub, gate = parsed
				total_power = runtime + sub + gate
				energy = total_power * sim
				values[regs] = energy
			continue

		# if directory, try to find a `results` file or gem5.out-like content
		if os.path.isdir(path):
			# common files: results, gem5.out (folder), gem5.out/stats.txt
			# try <dir>/results first
			candidates = [
				os.path.join(path, 'results'),
				os.path.join(path, 'gem5.out'),
				os.path.join(path, 'gem5.out', 'results'),
				os.path.join(path, 'gem5.out', 'stats.txt'),
				os.path.join(path, 'results.txt'),
			]
			txt = None
			for c in candidates:
				if os.path.isfile(c):
					with open(c, 'r', encoding='utf-8', errors='ignore') as f:
						txt = f.read()
					break

			if txt is None:
				# try files in the dir for anything that looks like gem5 output
				for subf in os.listdir(path):
					subp = os.path.join(path, subf)
					if os.path.isfile(subp):
						with open(subp, 'r', encoding='utf-8', errors='ignore') as f:
							data = f.read()
						if 'Simulated seconds' in data or 'Runtime Dynamic' in data:
							txt = data
							break

			if txt is None:
				continue

			regs = extract_reg_count(entry)
			if regs is None:
				# maybe directory named VECREGS_... (already handled) else skip
				regs = extract_reg_count(entry)
			parsed = parse_metrics_from_text(txt)
			if regs is not None and parsed is not None:
				sim, runtime, sub, gate = parsed
				total_power = runtime + sub + gate
				energy = total_power * sim
				values[regs] = energy

	return values


def plot_values(values: Dict[int, float], out_dir: str) -> None:
	if not values:
		print('No valid data found in', OUT_DIR)
		return

	os.makedirs(out_dir, exist_ok=True)

	items = sorted(values.items())
	regs = [k for k, _ in items]
	energy = [v for _, v in items]

	# save CSV
	csv_path = os.path.join(out_dir, 'vec_regs_energy.csv')
	with open(csv_path, 'w', newline='') as cf:
		w = csv.writer(cf)
		w.writerow(['num_vec_regs', 'energy_joules'])
		for r, e in zip(regs, energy):
			w.writerow([r, e])

	plt.figure(figsize=(7, 4.5))
	plt.plot(regs, energy, marker='o', linestyle='-')
	plt.xlabel('Number of physical vector registers')
	plt.ylabel('Energy (Joules)')
	plt.title('Energy vs number of vector physical registers')
	plt.grid(alpha=0.4)

	png = os.path.join(out_dir, 'vec_regs_energy.png')
	pdf = os.path.join(out_dir, 'vec_regs_energy.pdf')
	plt.tight_layout()
	plt.savefig(png, dpi=200)
	plt.savefig(pdf)
	print('Wrote', png, 'and', pdf, 'and', csv_path)


def main() -> None:
	values = collect_energy_values(OUT_DIR)
	if not values:
		print('No values parsed. Looked in', OUT_DIR)
		return
	plot_values(values, PLOTS_DIR)


if __name__ == '__main__':
	main()

