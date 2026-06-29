#!/usr/bin/env python3
"""
plot_psi.py
-----------
Visualisation of PSI (Phase Slope Index) connectivity results.

Reads HDF5 output from psi.py and produces two plot types:

1. --plot-group     (group)
   Signed PSI bar chart — mean +/- SEM per ROI pair x band, one panel
   per task. Bars above zero = first-named ROI leads; below zero =
   second-named ROI leads. Individual subject points overlaid.

2. --plot-matrix    (group)
   Directionality matrix heatmap — rows/cols = ROIs, colour = signed
   mean PSI (red = row leads column, blue = column leads row).
   One heatmap per band x task.

All plots saved to:
    derivatives/logs/plots/group/psi/

Usage
-----
    python code/plot_psi.py --root $MEGROOT --plot-group
    python code/plot_psi.py --root $MEGROOT --plot-matrix
    python code/plot_psi.py --root $MEGROOT --plot-group --plot-matrix \\
        --trials perceived
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from core import (
    ATLAS_CONFIGS,
    DEFAULT_ATLAS,
    TASKS,
    Paths,
    load_subjects,
    setup_logging,
    sub_id,
)
from psi import FREQ_BANDS
from connectivity_common import _get_roi_pairs

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")

TASK_COLORS = {
    "laser":    "#E63946",
    "pinprick": "#457B9D",
    "tactile":  "#2A9D8F",
}
TASK_LABELS = {
    "laser":    "Laser",
    "pinprick": "Pinprick",
    "tactile":  "Tactile",
}

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "figure.dpi":      150,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_psi_matrix(
    paths: Paths,
    subjects: list[str],
    tasks: list[str],
    bands: list[str],
    atlas_key: str,
    trial_filter: str = "all",
) -> tuple[dict, list[str]]:
    """Load PSI values from HDF5 files into a nested dict.

    Returns
    -------
    data[task][pair][band] = np.ndarray shape (n_subjects,), signed PSI
    all_pairs               = list of "roiA-roiB" pair strings
    """
    atlas_cfg = ATLAS_CONFIGS[atlas_key]
    roi_names = list(atlas_cfg["rois"].keys())
    all_pairs = [f"{a}-{b}" for a, b in _get_roi_pairs(roi_names)]
    suffix = "" if trial_filter == "all" else f"_{trial_filter}"

    data: dict = {
        task: {pair: {band: [] for band in bands} for pair in all_pairs}
        for task in tasks
    }
    n_found = 0

    for label in subjects:
        for task in tasks:
            fpath = (
                paths.deriv / "connectivity" / sub_id(label) / f"task-{task}"
                / f"{sub_id(label)}_task-{task}_psi_painmatrix{suffix}.h5"
            )
            if not fpath.exists() or fpath.stat().st_size == 0:
                for pair in all_pairs:
                    for band in bands:
                        data[task][pair][band].append(np.nan)
                continue

            with h5py.File(fpath, "r") as f:
                for pair in all_pairs:
                    a, b = pair.split("-")
                    grp_name = f"{a}__{b}"
                    for band in bands:
                        try:
                            val = float(f[grp_name][band].attrs["psi"])
                        except (KeyError, TypeError):
                            val = np.nan
                        data[task][pair][band].append(val)
            n_found += 1

    for task in tasks:
        for pair in all_pairs:
            for band in bands:
                data[task][pair][band] = np.array(data[task][pair][band], dtype=float)

    print(f"Loaded PSI: {n_found} subject x task files")
    return data, all_pairs


# ---------------------------------------------------------------------------
# Plot 1: Group bar chart, signed PSI per pair x band
# ---------------------------------------------------------------------------


def plot_psi_group_bar(
    data: dict, all_pairs: list[str], bands: list[str], tasks: list[str],
    out_dir: Path, logger,
) -> None:
    for band in bands:
        fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 6),
                                  squeeze=False)
        axes = axes[0]

        for ti, task in enumerate(tasks):
            ax = axes[ti]
            means, sems, valid_pairs = [], [], []
            for pair in all_pairs:
                vals = data[task][pair][band]
                valid = vals[~np.isnan(vals)]
                if len(valid) == 0:
                    continue
                means.append(valid.mean())
                sems.append(valid.std(ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0)
                valid_pairs.append((pair, valid))

            y = np.arange(len(valid_pairs))
            mean_vals = [m for m, _ in zip(means, valid_pairs)]
            colors = ["indianred" if m > 0 else "steelblue" for m in means]
            ax.barh(y, means, xerr=sems, color=colors, alpha=0.8, capsize=3)

            for yi, (pair, valid) in enumerate(valid_pairs):
                jitter = np.random.default_rng(0).uniform(-0.1, 0.1, size=len(valid))
                ax.scatter(valid, np.full(len(valid), yi) + jitter,
                           color="black", alpha=0.4, s=10, zorder=3)

            ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.set_yticks(y)
            ax.set_yticklabels([p for p, _ in valid_pairs], fontsize=8)
            ax.set_xlabel("PSI  (positive = first ROI leads)")
            ax.set_title(TASK_LABELS[task], color=TASK_COLORS[task],
                         fontweight="bold")

        fig.suptitle(
            f"PSI directionality  |  {band} "
            f"({FREQ_BANDS[band][0]}-{FREQ_BANDS[band][1]} Hz)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_path = out_dir / f"psi_group_bar_{band}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", out_path.name)


# ---------------------------------------------------------------------------
# Plot 2: Directionality matrix heatmap
# ---------------------------------------------------------------------------


def plot_psi_matrix(
    data: dict, all_pairs: list[str], roi_names: list[str], bands: list[str],
    tasks: list[str], out_dir: Path, logger,
) -> None:
    pair_to_idx = {f"{a}-{b}": (a, b) for a, b in _get_roi_pairs(roi_names)}
    n = len(roi_names)
    name_to_i = {name: i for i, name in enumerate(roi_names)}

    for band in bands:
        fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5.5),
                                  squeeze=False)
        axes = axes[0]

        for ti, task in enumerate(tasks):
            ax = axes[ti]
            mat = np.full((n, n), np.nan)
            for pair in all_pairs:
                a, b = pair_to_idx[pair]
                vals = data[task][pair][band]
                valid = vals[~np.isnan(vals)]
                if len(valid) == 0:
                    continue
                m = valid.mean()
                ia, ib = name_to_i[a], name_to_i[b]
                mat[ia, ib] = m
                mat[ib, ia] = -m  # antisymmetric: b->a is negative of a->b

            vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1.0
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(roi_names, fontsize=7, rotation=90)
            ax.set_yticklabels(roi_names, fontsize=7)
            ax.set_title(TASK_LABELS[task], color=TASK_COLORS[task],
                         fontweight="bold")
            fig.colorbar(im, ax=ax, shrink=0.8, label="PSI (row -> col)")

        fig.suptitle(
            f"PSI directionality matrix  |  {band} "
            f"({FREQ_BANDS[band][0]}-{FREQ_BANDS[band][1]} Hz)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = out_dir / f"psi_matrix_{band}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info("Saved: %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot PSI (Phase Slope Index) connectivity results."
    )
    parser.add_argument("--root",     type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--tasks",    nargs="+", default=None, choices=TASKS)
    parser.add_argument("--bands",    nargs="+", default=list(FREQ_BANDS.keys()),
                         choices=list(FREQ_BANDS.keys()))
    parser.add_argument("--atlas",    default=DEFAULT_ATLAS,
                         choices=list(ATLAS_CONFIGS.keys()))
    parser.add_argument(
        "--trials", default="all", choices=["all", "perceived", "not-perceived"],
        help="Which trial-filtered PSI results to load/plot "
             "(must match the --trials used when running psi.py).",
    )
    parser.add_argument("--plot-group",  action="store_true",
                         help="Signed PSI bar chart per pair x band.")
    parser.add_argument("--plot-matrix", action="store_true",
                         help="Directionality matrix heatmap per band.")
    args = parser.parse_args()

    paths = Paths(args.root)
    logger = setup_logging(paths, "plot_psi")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks = args.tasks if args.tasks else TASKS

    logger.info("Subjects : %d  %s", len(subjects), subjects)
    logger.info("Tasks    : %s", tasks)
    logger.info("Bands    : %s", args.bands)
    logger.info("Trials   : %s", args.trials)

    out_dir = paths.log_dir() / "plots" / "group" / "psi"
    out_dir.mkdir(parents=True, exist_ok=True)

    data, all_pairs = load_psi_matrix(
        paths, subjects, tasks, args.bands, args.atlas, args.trials
    )

    if not (args.plot_group or args.plot_matrix):
        logger.info("No plot type selected. Use --plot-group and/or --plot-matrix.")
        return

    if args.plot_group:
        logger.info("Generating group bar plots ...")
        plot_psi_group_bar(data, all_pairs, args.bands, tasks, out_dir, logger)

    if args.plot_matrix:
        logger.info("Generating directionality matrices ...")
        roi_names = list(ATLAS_CONFIGS[args.atlas]["rois"].keys())
        plot_psi_matrix(data, all_pairs, roi_names, args.bands, tasks, out_dir, logger)

    logger.info("All plots saved to: %s", out_dir)


if __name__ == "__main__":
    main()
