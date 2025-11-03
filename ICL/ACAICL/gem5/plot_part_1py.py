import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# --- Plot style constants ---
# A compact, high-contrast palette and a set of markers/linestyles to
# make each line visually distinct. Adjust these lists to change the
# look of the plots (order matters; they will be cycled).
DEFAULT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]
# Use only solid lines per user request
LINESTYLES = ["-"]

FOLDER_PATTERN = re.compile(r"^rob_(\d+)_lsq_(\d+)$")


def parse_results_file(results_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (total_energy (J), simulated_seconds).
    total_energy = (Subthreshold Leakage + Gate Leakage + Runtime Dynamic) * Simulated seconds.
    If a component is missing, energy may be None while simulated_seconds could still be available.
    """
    if not os.path.isfile(results_path):
        return None, None

    with open(results_path, "r", encoding="utf-8") as f:
        text = f.read()

    def grab(name: str) -> Optional[float]:
        m = re.search(rf"^\s*{re.escape(name)}\s*=\s*([0-9.+\-eE]+)", text, flags=re.MULTILINE)
        return float(m.group(1)) if m else None

    sub = grab("Subthreshold Leakage")
    gate = grab("Gate Leakage")
    runtime = grab("Runtime Dynamic")
    sim_seconds = grab("Simulated seconds")

    total_energy = None
    if sub is not None and gate is not None and runtime is not None and sim_seconds is not None:
        total_energy = (sub + gate + runtime) * sim_seconds

    return total_energy, sim_seconds


def collect_data(base_dir: str) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]], List[int], List[int]]:
    """
    Scans base_dir for rob_*_lsq_* folders, parses total energy and simulated seconds, and returns:
      - energy_data[rob][lsq] = total_energy (J)
      - seconds_data[rob][lsq] = simulated seconds
      - sorted unique rob sizes
      - sorted unique lsq sizes
    """
    energy_data: Dict[int, Dict[int, float]] = defaultdict(dict)
    seconds_data: Dict[int, Dict[int, float]] = defaultdict(dict)
    rob_set, lsq_set = set(), set()

    for name in os.listdir(base_dir):
        m = FOLDER_PATTERN.match(name)
        if not m:
            continue
        rob, lsq = int(m.group(1)), int(m.group(2))
        results_path = os.path.join(base_dir, name, "results")
        energy, sim_seconds = parse_results_file(results_path)

        any_valid = False
        if energy is not None:
            energy_data[rob][lsq] = energy
            any_valid = True
        if sim_seconds is not None:
            seconds_data[rob][lsq] = sim_seconds
            any_valid = True
        if any_valid:
            rob_set.add(rob)
            lsq_set.add(lsq)

    rob_sizes = sorted(rob_set)
    lsq_sizes = sorted(lsq_set)
    return energy_data, seconds_data, rob_sizes, lsq_sizes


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_power_vs_rob_by_lsq(
    data: Dict[int, Dict[int, float]],
    rob_sizes: List[int],
    lsq_sizes: List[int],
    out_dir: str,
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    color_cycle = cycle(DEFAULT_COLORS)
    marker_cycle = cycle(MARKERS)
    ls_cycle = cycle(LINESTYLES)
    for lsq in lsq_sizes:
        xs, ys = [], []
        for rob in rob_sizes:
            val = data.get(rob, {}).get(lsq)
            if val is not None:
                xs.append(rob)
                ys.append(val)
        if xs:
            color = next(color_cycle)
            marker = next(marker_cycle)
            linestyle = next(ls_cycle)
            plt.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=6,
                label=f"LSQ={lsq}",
            )

    plt.title("Total Energy vs ROB Size", fontsize=14, fontweight="bold")
    plt.xlabel("ROB size", fontsize=11)
    plt.ylabel("Total Energy (J)", fontsize=11)
    plt.xticks(rob_sizes)
    plt.grid(True, alpha=0.25)
    plt.legend(title="LSQ", fontsize=10, title_fontsize=11, ncol=2, frameon=True)
    out_png = os.path.join(out_dir, "energy_vs_rob_by_lsq.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    if save_svg:
        plt.savefig(os.path.join(out_dir, "energy_vs_rob_by_lsq.svg"))
    plt.close()


def plot_power_vs_lsq_by_rob(
    data: Dict[int, Dict[int, float]],
    rob_sizes: List[int],
    lsq_sizes: List[int],
    out_dir: str,
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    color_cycle = cycle(DEFAULT_COLORS)
    marker_cycle = cycle(MARKERS)
    ls_cycle = cycle(LINESTYLES)
    for rob in rob_sizes:
        xs, ys = [], []
        for lsq in lsq_sizes:
            val = data.get(rob, {}).get(lsq)
            if val is not None:
                xs.append(lsq)
                ys.append(val)
        if xs:
            color = next(color_cycle)
            marker = next(marker_cycle)
            linestyle = next(ls_cycle)
            plt.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=6,
                label=f"ROB={rob}",
            )

    plt.title("Total Energy vs LSQ Size vs ROB Size", fontsize=14, fontweight="bold")
    plt.xlabel("LSQ size", fontsize=11)
    plt.ylabel("Total Energy (J)", fontsize=11)
    plt.xticks(lsq_sizes)
    plt.grid(True, alpha=0.25)
    plt.legend(title="ROB", fontsize=10, title_fontsize=11, ncol=2, frameon=True)
    out_png = os.path.join(out_dir, "energy_vs_lsq_by_rob.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    if save_svg:
        plt.savefig(os.path.join(out_dir, "energy_vs_lsq_by_rob.svg"))
    plt.close()


def plot_power_heatmap(
    data: Dict[int, Dict[int, float]],
    rob_sizes: List[int],
    lsq_sizes: List[int],
    out_dir: str,
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    # Build matrix with rows=LSQ, cols=ROB
    mat = np.full((len(lsq_sizes), len(rob_sizes)), np.nan, dtype=float)
    for i, lsq in enumerate(lsq_sizes):
        for j, rob in enumerate(rob_sizes):
            v = data.get(rob, {}).get(lsq)
            if v is not None:
                mat[i, j] = v

    plt.figure(figsize=(8, 5))
    im = plt.imshow(mat, aspect="auto", origin="upper", cmap="viridis")
    cbar = plt.colorbar(im, label="Total Energy (J)")
    cbar.ax.tick_params(labelsize=10)
    plt.xticks(range(len(rob_sizes)), rob_sizes, rotation=0)
    plt.yticks(range(len(lsq_sizes)), lsq_sizes)
    plt.xlabel("ROB size")
    plt.ylabel("LSQ size")
    plt.title("Total Energy across ROB and LSQ", fontsize=14, fontweight="bold")
    out_png = os.path.join(out_dir, "energy_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    if save_svg:
        plt.savefig(os.path.join(out_dir, "energy_heatmap.svg"))
    plt.close()


def plot_seconds_vs_rob_by_lsq(
    data: Dict[int, Dict[int, float]],
    rob_sizes: List[int],
    lsq_sizes: List[int],
    out_dir: str,
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    color_cycle = cycle(DEFAULT_COLORS)
    marker_cycle = cycle(MARKERS)
    ls_cycle = cycle(LINESTYLES)
    for lsq in lsq_sizes:
        xs, ys = [], []
        for rob in rob_sizes:
            val = data.get(rob, {}).get(lsq)
            if val is not None:
                xs.append(rob)
                ys.append(val)
        if xs:
            color = next(color_cycle)
            marker = next(marker_cycle)
            linestyle = next(ls_cycle)
            plt.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=6,
                label=f"LSQ={lsq}",
            )

    plt.title("Simulated Seconds vs ROB Size", fontsize=14, fontweight="bold")
    plt.xlabel("ROB size", fontsize=11)
    plt.ylabel("Simulated seconds (s)", fontsize=11)
    plt.xticks(rob_sizes)
    plt.grid(True, alpha=0.25)
    plt.legend(title="LSQ", fontsize=10, title_fontsize=11, ncol=2, frameon=True)
    out_png = os.path.join(out_dir, "seconds_vs_rob_by_lsq.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    if save_svg:
        plt.savefig(os.path.join(out_dir, "seconds_vs_rob_by_lsq.svg"))
    plt.close()


def plot_seconds_vs_lsq_by_rob(
    data: Dict[int, Dict[int, float]],
    rob_sizes: List[int],
    lsq_sizes: List[int],
    out_dir: str,
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    plt.figure(figsize=(8, 5))
    color_cycle = cycle(DEFAULT_COLORS)
    marker_cycle = cycle(MARKERS)
    ls_cycle = cycle(LINESTYLES)
    for rob in rob_sizes:
        xs, ys = [], []
        for lsq in lsq_sizes:
            val = data.get(rob, {}).get(lsq)
            if val is not None:
                xs.append(lsq)
                ys.append(val)
        if xs:
            color = next(color_cycle)
            marker = next(marker_cycle)
            linestyle = next(ls_cycle)
            plt.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=6,
                label=f"ROB={rob}",
            )

    plt.title("Simulated Seconds vs LSQ Size vs ROB Size", fontsize=14, fontweight="bold")
    plt.xlabel("LSQ size", fontsize=11)
    plt.ylabel("Simulated seconds (s)", fontsize=11)
    plt.xticks(lsq_sizes)
    plt.grid(True, alpha=0.25)
    plt.legend(title="ROB", fontsize=10, title_fontsize=11, ncol=2, frameon=True)
    out_png = os.path.join(out_dir, "seconds_vs_lsq_by_rob.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    if save_svg:
        plt.savefig(os.path.join(out_dir, "seconds_vs_lsq_by_rob.svg"))
    plt.close()


def plot_seconds_heatmap(
    data: Dict[int, Dict[int, float]],
    rob_sizes: List[int],
    lsq_sizes: List[int],
    out_dir: str,
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    mat = np.full((len(lsq_sizes), len(rob_sizes)), np.nan, dtype=float)
    for i, lsq in enumerate(lsq_sizes):
        for j, rob in enumerate(rob_sizes):
            v = data.get(rob, {}).get(lsq)
            if v is not None:
                mat[i, j] = v

    plt.figure(figsize=(8, 5))
    im = plt.imshow(mat, aspect="auto", origin="upper", cmap="magma")
    cbar = plt.colorbar(im, label="Simulated seconds (s)")
    cbar.ax.tick_params(labelsize=10)
    plt.xticks(range(len(rob_sizes)), rob_sizes, rotation=0)
    plt.yticks(range(len(lsq_sizes)), lsq_sizes)
    plt.xlabel("ROB size")
    plt.ylabel("LSQ size")
    plt.title("Simulated Seconds across ROB and LSQ", fontsize=14, fontweight="bold")
    out_png = os.path.join(out_dir, "seconds_heatmap.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    if save_svg:
        plt.savefig(os.path.join(out_dir, "seconds_heatmap.svg"))
    plt.close()


def plot_seconds_vs_lsq_by_selected_robs(
    data: Dict[int, Dict[int, float]],
    rob_sizes: List[int],
    lsq_sizes: List[int],
    out_dir: str,
    selected_robs: List[int] = (128, 256, 512),
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    """
    Plot 'Simulated seconds vs LSQ size' but only for a selected subset of ROB sizes.
    Saves file to out_dir with a filename that indicates the selected ROBs.
    """
    # Filter the available rob sizes by the requested selection but keep ordering
    robs_to_plot = [r for r in selected_robs if r in rob_sizes]
    if not robs_to_plot:
        # Nothing to plot
        return

    plt.figure(figsize=(8, 5))
    color_cycle = cycle(DEFAULT_COLORS)
    marker_cycle = cycle(MARKERS)
    ls_cycle = cycle(LINESTYLES)

    for rob in robs_to_plot:
        xs, ys = [], []
        for lsq in lsq_sizes:
            val = data.get(rob, {}).get(lsq)
            if val is not None:
                xs.append(lsq)
                ys.append(val)
        if xs:
            color = next(color_cycle)
            marker = next(marker_cycle)
            linestyle = next(ls_cycle)
            plt.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=6,
                label=f"ROB={rob}",
            )

    plt.title(f"Simulated Seconds vs LSQ Size (ROB: {', '.join(map(str, robs_to_plot))})", fontsize=14, fontweight="bold")
    plt.xlabel("LSQ size", fontsize=11)
    plt.ylabel("Simulated seconds (s)", fontsize=11)
    plt.xticks(lsq_sizes)
    plt.grid(True, alpha=0.25)
    plt.legend(title="ROB", fontsize=10, title_fontsize=11, ncol=1, frameon=True)

    # Build filename reflecting selected robs
    name_suffix = "_".join(str(r) for r in robs_to_plot)
    out_png = os.path.join(out_dir, f"seconds_vs_lsq_by_selected_robs_{name_suffix}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    if save_svg:
        plt.savefig(os.path.join(out_dir, f"seconds_vs_lsq_by_selected_robs_{name_suffix}.svg"))
    plt.close()


def plot_per_lsq(
    energy_data: Dict[int, Dict[int, float]],
    seconds_data: Dict[int, Dict[int, float]],
    rob_sizes: List[int],
    lsq_sizes: List[int],
    out_dir: str,
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    """
    For each LSQ size, generate two plots saved under <out_dir>/per_lsq/:
      - Total Energy vs ROB for that LSQ
      - Simulated seconds vs ROB for that LSQ

    This keeps per-LSQ views grouped under a dedicated subfolder so they
    are easy to browse. Filenames:
      energy_vs_rob_lsq_<lsq>.png
      seconds_vs_rob_lsq_<lsq>.png
    """
    per_lsq_dir = os.path.join(out_dir, "per_lsq")
    ensure_out_dir(per_lsq_dir)

    # Prepare cycles for consistent styling across LSQ plots
    color_cycle = cycle(DEFAULT_COLORS)
    marker_cycle = cycle(MARKERS)
    ls_cycle = cycle(LINESTYLES)

    for lsq in lsq_sizes:
        # ENERGY plot for this LSQ
        xs, ys = [], []
        for rob in rob_sizes:
            # Always attempt to read from the dict; an empty dict is fine
            v = energy_data.get(rob, {}).get(lsq)
            if v is not None:
                xs.append(rob)
                ys.append(v)
        if xs:
            color = next(color_cycle)
            marker = next(marker_cycle)
            linestyle = next(ls_cycle)
            plt.figure(figsize=(7, 4))
            plt.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=6,
                label=f"LSQ={lsq}",
            )
            plt.title(f"Total Energy vs ROB (LSQ={lsq})", fontsize=13, fontweight="bold")
            plt.xlabel("ROB size", fontsize=11)
            plt.ylabel("Total Energy (J)", fontsize=11)
            plt.xticks(rob_sizes)
            plt.grid(True, alpha=0.25)
            plt.legend().set_visible(False)
            out_png = os.path.join(per_lsq_dir, f"energy_vs_rob_lsq_{lsq}.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=dpi)
            if save_svg:
                plt.savefig(os.path.join(per_lsq_dir, f"energy_vs_rob_lsq_{lsq}.svg"))
            plt.close()

        # SECONDS plot for this LSQ
        xs, ys = [], []
        for rob in rob_sizes:
            # Always attempt to read from the dict; an empty dict is fine
            v = seconds_data.get(rob, {}).get(lsq)
            if v is not None:
                xs.append(rob)
                ys.append(v)
        if xs:
            color = next(color_cycle)
            marker = next(marker_cycle)
            linestyle = next(ls_cycle)
            plt.figure(figsize=(7, 4))
            plt.plot(
                xs,
                ys,
                marker=marker,
                linestyle=linestyle,
                color=color,
                linewidth=2,
                markersize=6,
                label=f"LSQ={lsq}",
            )
            plt.title(f"Simulated Seconds vs ROB (LSQ={lsq})", fontsize=13, fontweight="bold")
            plt.xlabel("ROB size", fontsize=11)
            plt.ylabel("Simulated seconds (s)", fontsize=11)
            plt.xticks(rob_sizes)
            plt.grid(True, alpha=0.25)
            plt.legend().set_visible(False)
            out_png = os.path.join(per_lsq_dir, f"seconds_vs_rob_lsq_{lsq}.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=dpi)
            if save_svg:
                plt.savefig(os.path.join(per_lsq_dir, f"seconds_vs_rob_lsq_{lsq}.svg"))
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot total energy and performance (simulated seconds) vs ROB/LSQ sizes from part_1 results.")
    parser.add_argument(
        "--base-dir",
        default="./part_1_results/aca",
        help="Base directory containing rob_*_lsq_* folders.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save plots (default: <base-dir>/plots).",
    )
    parser.add_argument("--show", action="store_true", help="Display plots interactively in addition to saving.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved PNGs.")
    parser.add_argument("--svg", action="store_true", help="Also save SVG versions.")
    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir or os.path.join(base_dir, "plots")
    ensure_out_dir(out_dir)

    energy_data, seconds_data, rob_sizes, lsq_sizes = collect_data(base_dir)
    if not energy_data and not seconds_data:
        print(f"No valid data found under: {base_dir}")
        return

    # Energy plots
    if energy_data:
        plot_power_vs_rob_by_lsq(energy_data, rob_sizes, lsq_sizes, out_dir, dpi=args.dpi, save_svg=args.svg)
        plot_power_vs_lsq_by_rob(energy_data, rob_sizes, lsq_sizes, out_dir, dpi=args.dpi, save_svg=args.svg)
        plot_power_heatmap(energy_data, rob_sizes, lsq_sizes, out_dir, dpi=args.dpi, save_svg=args.svg)
    else:
        print("Warning: No energy data found (missing power components or simulated seconds).")

    # Performance (simulated seconds) plots
    if seconds_data:
        plot_seconds_vs_rob_by_lsq(seconds_data, rob_sizes, lsq_sizes, out_dir, dpi=args.dpi, save_svg=args.svg)
        plot_seconds_vs_lsq_by_rob(seconds_data, rob_sizes, lsq_sizes, out_dir, dpi=args.dpi, save_svg=args.svg)
        plot_seconds_heatmap(seconds_data, rob_sizes, lsq_sizes, out_dir, dpi=args.dpi, save_svg=args.svg)
        # Also save a plot that compares only a subset of ROB sizes (128,256,512)
        plot_seconds_vs_lsq_by_selected_robs(seconds_data, rob_sizes, lsq_sizes, out_dir, selected_robs=(128, 256, 512), dpi=args.dpi, save_svg=args.svg)
    else:
        print("Warning: No simulated seconds data found.")

    # Per-LSQ focused plots (one plot per LSQ showing values vs ROB)
    plot_per_lsq(energy_data, seconds_data, rob_sizes, lsq_sizes, out_dir, dpi=args.dpi, save_svg=args.svg)

    if args.show:
        import matplotlib.image as mpimg
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        preview = [
            ("energy_vs_rob_by_lsq.png", "Energy vs ROB (by LSQ)"),
            ("energy_vs_lsq_by_rob.png", "Energy vs LSQ (by ROB)"),
            ("energy_heatmap.png", "Energy heatmap"),
            ("seconds_vs_rob_by_lsq.png", "Seconds vs ROB (by LSQ)"),
            ("seconds_vs_lsq_by_rob.png", "Seconds vs LSQ (by ROB)"),
            ("seconds_heatmap.png", "Seconds heatmap"),
        ]
        for ax, (fname, title) in zip(axes.flatten(), preview):
            p = os.path.join(out_dir, fname)
            if os.path.isfile(p):
                img = mpimg.imread(p)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(title)
            else:
                ax.axis("off")
                ax.set_title(f"{title} (missing)")
        plt.tight_layout()
        plt.show()

    print(f"Saved plots to: {out_dir}")
    print("Tip: pip install matplotlib if not installed.")


if __name__ == "__main__":
    main()
