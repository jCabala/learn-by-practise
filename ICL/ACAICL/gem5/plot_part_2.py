import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter, MaxNLocator
from itertools import cycle

# Small helper: consistent axis/legend styling used across plots
AX_TITLE_KW = {"fontsize": 14, "fontweight": "bold"}
AX_LABEL_KW = {"fontsize": 11}

# Example folder name:
# LROB32_IQ21_LSQ11__BP_L64_G128_BTB128_RAS16
FOLDER_PATTERN = re.compile(
    r"^(?:L?ROB)(\d+)_IQ(\d+)_LSQ(\d+)__BP_L(\d+)_G(\d+)_BTB(\d+)_RAS(\+?\d+)$"
)

PARAM_LABELS = {
    "rob": "ROB size",
    "iq": "IQ size",
    "lsq": "LSQ size",
    "l": "Local predictor size",
    "g": "Global predictor size",
    "btb": "BTB entries",
    "ras": "RAS entries",
}

# Match palettes and markers used in part 1 for consistent visuals
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
LINESTYLES = ["-"]


def parse_results_file(results_path: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Returns (total_energy (J), simulated_seconds).
    total_energy = (Subthreshold Leakage + Gate Leakage + Runtime Dynamic) * Simulated seconds.
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

    if sub is None or gate is None or runtime is None or sim_seconds is None:
        return None, sim_seconds

    total_power = sub + gate + runtime
    return total_power * sim_seconds, sim_seconds


def collect_samples(base_dir: str) -> List[Dict[str, float]]:
    """
    Scan base_dir for config folders and return a list of samples with parsed params and energy.
    Each sample dict contains: rob, iq, lsq, l, g, btb, ras, energy.
    """
    samples: List[Dict[str, float]] = []
    for name in os.listdir(base_dir):
        m = FOLDER_PATTERN.match(name)
        if not m:
            continue
        rob, iq, lsq, l, g, btb, ras = map(int, m.groups())
        results_path = os.path.join(base_dir, name, "results")
        energy, _ = parse_results_file(results_path)
        if energy is None:
            continue
        samples.append(
            {
                "rob": float(rob),
                "iq": float(iq),
                "lsq": float(lsq),
                "l": float(l),
                "g": float(g),
                "btb": float(btb),
                "ras": float(ras),
                "energy": float(energy),
            }
        )
    return samples


def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _window_tuple(s: Dict[str, float]) -> Tuple[float, float, float]:
    return (s["rob"], s["iq"], s["lsq"])


def _predictor_tuple(s: Dict[str, float]) -> Tuple[float, float, float, float]:
    return (s["l"], s["g"], s["btb"], s["ras"])


def _format_window_tuple(t: Tuple[float, float, float]) -> str:
    r, i, l = t
    return f"(ROB={int(r)},IQ={int(i)},LSQ={int(l)})"


def _format_predictor_tuple(t: Tuple[float, float, float, float]) -> str:
    l, g, b, r = t
    return f"(Local={int(l)},Global={int(g)},BTB={int(b)},RAS={int(r)})"


def plot_energy_vs_window_by_predictor(
    samples: List[Dict[str, float]],
    out_path: str,
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    # Aggregate energies by (W, P)
    agg: Dict[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]], List[float]] = defaultdict(list)
    windows_set, preds_set = set(), set()
    for s in samples:
        w, p = _window_tuple(s), _predictor_tuple(s)
        agg[(w, p)].append(s["energy"])
        windows_set.add(w)
        preds_set.add(p)

    windows = sorted(windows_set)  # strictly increasing => lexicographic order works
    preds = sorted(preds_set)

    x_positions = list(range(len(windows)))
    x_labels = [_format_window_tuple(w) for w in windows]

    plt.figure(figsize=(max(8, len(windows) * 0.8), 5))
    color_cycle = cycle(DEFAULT_COLORS)
    marker_cycle = cycle(MARKERS)
    ls_cycle = cycle(LINESTYLES)
    any_plot = False
    for p in preds:
        ys = []
        for w in windows:
            vals = agg.get((w, p))
            ys.append(float(np.mean(vals)) if vals else np.nan)
        if np.isfinite(ys).any():
            color = next(color_cycle)
            marker = next(marker_cycle)
            linestyle = next(ls_cycle)
            plt.plot(x_positions, ys, marker=marker, linestyle=linestyle, color=color, linewidth=2, markersize=6, label=_format_predictor_tuple(p))
            any_plot = True

    if not any_plot:
        plt.close()
        return

    plt.title("Total Energy vs Window Size vs Predictor Size")
    plt.xlabel("Window Size(ROB, IQ, LSQ)")
    plt.ylabel("Total Energy (J)")
    # smaller font for x-axis tick labels to avoid overlap on long labels
    plt.xticks(x_positions, x_labels, rotation=45, ha="right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    if save_svg:
        plt.savefig(os.path.splitext(out_path)[0] + ".svg")
    plt.close()


def plot_energy_vs_predictor_by_window(
    samples: List[Dict[str, float]],
    out_path: str,
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    # Aggregate energies by (P, W)
    agg: Dict[Tuple[Tuple[float, float, float, float], Tuple[float, float, float]], List[float]] = defaultdict(list)
    windows_set, preds_set = set(), set()
    for s in samples:
        w, p = _window_tuple(s), _predictor_tuple(s)
        agg[(p, w)].append(s["energy"])
        windows_set.add(w)
        preds_set.add(p)

    preds = sorted(preds_set)
    windows = sorted(windows_set)

    x_positions = list(range(len(preds)))
    x_labels = [_format_predictor_tuple(p) for p in preds]

    plt.figure(figsize=(max(8, len(preds) * 0.8), 5))
    color_cycle = cycle(DEFAULT_COLORS)
    marker_cycle = cycle(MARKERS)
    ls_cycle = cycle(LINESTYLES)
    any_plot = False
    for w in windows:
        ys = []
        for p in preds:
            vals = agg.get((p, w))
            ys.append(float(np.mean(vals)) if vals else np.nan)
        if np.isfinite(ys).any():
            color = next(color_cycle)
            marker = next(marker_cycle)
            linestyle = next(ls_cycle)
            plt.plot(x_positions, ys, marker=marker, linestyle=linestyle, color=color, linewidth=2, markersize=6, label=_format_window_tuple(w))
            any_plot = True

    if not any_plot:
        plt.close()
        return

    plt.title("Total Energy vs Predictor Size vs Window Size")
    plt.xlabel("Predictor Size (Local, Global, BTB, RAS)")
    plt.ylabel("Total Energy (J)")
    # smaller font for x-axis tick labels to avoid overlap on long labels
    plt.xticks(x_positions, x_labels, rotation=45, ha="right", fontsize=5)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    if save_svg:
        plt.savefig(os.path.splitext(out_path)[0] + ".svg")
    plt.close()


def plot_energy_vs_window_first_predictor(
    samples: List[Dict[str, float]],
    out_path: str,
    dpi: int = 150,
    save_svg: bool = False,
) -> None:
    """Plot Total Energy vs Window Size but only for the first sorted predictor.

    This mirrors `plot_energy_vs_window_by_predictor` but filters to the first
    predictor in lexicographic order (i.e., the first item of the sorted preds).
    """
    # Aggregate energies by (W, P)
    agg: Dict[Tuple[Tuple[float, float, float], Tuple[float, float, float, float]], List[float]] = defaultdict(list)
    windows_set, preds_set = set(), set()
    for s in samples:
        w, p = _window_tuple(s), _predictor_tuple(s)
        agg[(w, p)].append(s["energy"])
        windows_set.add(w)
        preds_set.add(p)

    windows = sorted(windows_set)
    preds = sorted(preds_set)

    if not preds:
        return

    # select only the first predictor
    first_pred = preds[0]

    x_positions = list(range(len(windows)))
    x_labels = [_format_window_tuple(w) for w in windows]

    plt.figure(figsize=(max(8, len(windows) * 0.8), 5))
    color_cycle = cycle(DEFAULT_COLORS)
    marker_cycle = cycle(MARKERS)
    ls_cycle = cycle(LINESTYLES)
    ys = []
    for w in windows:
        vals = agg.get((w, first_pred))
        ys.append(float(np.mean(vals)) if vals else np.nan)

    if not np.isfinite(ys).any():
        plt.close()
        return

    # pick a style for the single-line plot
    color = next(color_cycle)
    marker = next(marker_cycle)
    linestyle = next(ls_cycle)
    plt.plot(x_positions, ys, marker=marker, linestyle=linestyle, color=color, linewidth=2, markersize=6, label=_format_predictor_tuple(first_pred))
    plt.title("Total Energy vs Window Size for First Predictor Size")
    plt.xlabel("Window Size(ROB, IQ, LSQ)")
    plt.ylabel("Total Energy (J)")
    # smaller font for x-axis tick labels to avoid overlap on long labels
    plt.xticks(x_positions, x_labels, rotation=45, ha="right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    if save_svg:
        plt.savefig(os.path.splitext(out_path)[0] + ".svg")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot energy vs window and predictor tuples from part_2 results."
    )
    parser.add_argument(
        "--base-dir",
        default="./part_2_results/",
        help="Directory containing part_2 config folders.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save plots (default: <base-dir>/plots).",
    )
    parser.add_argument("--show", action="store_true", help="Display preview of saved plots.")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved PNGs.")
    parser.add_argument("--svg", action="store_true", help="Also save SVG versions.")
    args = parser.parse_args()

    base_dir = args.base_dir
    out_dir = args.out_dir or os.path.join(base_dir, "plots")
    ensure_out_dir(out_dir)

    samples = collect_samples(base_dir)
    if not samples:
        print(f"No valid data found under: {base_dir}")
        return

    # Tuple-based plots
    w_by_p = os.path.join(out_dir, "energy_vs_window_by_predictor.png")
    p_by_w = os.path.join(out_dir, "energy_vs_predictor_by_window.png")
    plot_energy_vs_window_by_predictor(samples, w_by_p, dpi=args.dpi, save_svg=args.svg)
    plot_energy_vs_predictor_by_window(samples, p_by_w, dpi=args.dpi, save_svg=args.svg)
    # also create a dedicated plot for only the first predictor (sorted order)
    first_pred_plot = os.path.join(out_dir, "energy_vs_window_first_predictor.png")
    plot_energy_vs_window_first_predictor(samples, first_pred_plot, dpi=args.dpi, save_svg=args.svg)

    if args.show:
        import matplotlib.image as mpimg
        figs = [os.path.basename(w_by_p), os.path.basename(p_by_w)]
        cols = 2
        rows = 1
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4))
        axes = np.array(axes).reshape(rows, cols)
        for ax, fname in zip(axes.flatten(), figs):
            p = os.path.join(out_dir, fname)
            if os.path.isfile(p):
                img = mpimg.imread(p)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(fname.replace("_", " ").replace(".png", ""))
            else:
                ax.axis("off")
                ax.set_title(f"{fname} (missing)")
        plt.tight_layout()
        plt.show()

    print(f"Saved plots to: {out_dir}")
    print("Tip: pip install matplotlib if not installed.")


if __name__ == "__main__":
    main()
