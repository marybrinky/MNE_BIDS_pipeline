#!/usr/bin/env python3
"""
plot_pac.py
-----------
Visualisation of PAC results for the laser-pain MEG study.

Reads HDF5 output from pac.py and produces three plot types:

1. Comodulogram
   2-D heatmap: phase-band (rows) × amplitude-band (columns).
   Colour encodes mean MI across subjects, one panel per task.
   Both directions per pair are shown side-by-side.

2. Directionality contrast
   For each ROI pair: bar chart of MI  A→B vs B→A, all band combos,
   tasks overlaid.  Highlights asymmetric coupling (potential driver–
   follower relationships).

3. Raincloud (z-score)
   Per direction × band combo: subject-level z-score distributions
   across tasks.  Identical style to plot_wpli.py (Allen et al. 2019).

Outputs
-------
    derivatives/logs/plots/group/pac/
        comodulogram_<pair>.png
        directionality_<pair>_<phase>_<amp>.png
        raincloud_<direction>_<combo>.png

Usage
-----
    python plot_pac.py
    python plot_pac.py --subjects 4382 4383
    python plot_pac.py --metric z_score      # default; or 'mi'
    python plot_pac.py --no-raincloud
"""

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from core import (
    ATLAS_CONFIGS,
    DEFAULT_ATLAS,
    TASKS,
    Paths,
    load_subjects,
    setup_logging,
    sub_id,
)
from pac import PAC_AMP_BANDS, PAC_PHASE_BANDS

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")

TASK_COLORS = {
    "laser": "#E63946",
    "pinprick": "#457B9D",
    "tactile": "#2A9D8F",
}
TASK_LABELS = {
    "laser": "Laser",
    "pinprick": "Pinprick",
    "tactile": "Tactile",
}
CONDITION_KEY = "stimulus"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_directions(roi_names: list[str]) -> list[str]:
    """Return direction strings for all ordered ROI pairs."""
    directions = []
    for i, a in enumerate(roi_names):
        for b in roi_names[i + 1 :]:
            directions.append(f"{a}_to_{b}")
            directions.append(f"{b}_to_{a}")
    return directions


def _all_combos(phase_bands: list[str], amp_bands: list[str]) -> list[str]:
    return [f"phase_{p}_amp_{a}" for p in phase_bands for a in amp_bands]


def _unidirectional_pairs(directions: list[str]) -> list[tuple[str, str]]:
    """Return unique (A_to_B, B_to_A) tuples (each unordered pair once)."""
    seen: set[frozenset] = set()
    pairs = []
    for d in directions:
        parts = d.split("_to_")
        if len(parts) != 2:
            continue
        key = frozenset(parts)
        if key not in seen:
            seen.add(key)
            a, b = parts
            rev = f"{b}_to_{a}"
            if rev in directions:
                pairs.append((d, rev))
    return pairs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_pac_matrix(
    paths: Paths,
    subjects: list[str],
    tasks: list[str],
    phase_bands: list[str],
    amp_bands: list[str],
    atlas_key: str,
    metric: str = "z_score",
) -> tuple[dict, list[str], list[str]]:
    """Load PAC values from HDF5 files into a nested dict.

    Returns
    -------
    data[task][direction][combo] = np.ndarray shape (n_subjects,)
    directions : list of direction strings present in the data
    combos     : list of combo strings
    """
    atlas_cfg = ATLAS_CONFIGS[atlas_key]
    roi_names = list(atlas_cfg["rois"].keys())
    directions = _all_directions(roi_names)
    combos = _all_combos(phase_bands, amp_bands)

    data: dict = {
        task: {d: {c: [] for c in combos} for d in directions} for task in tasks
    }
    n_found = 0

    for label in subjects:
        for task in tasks:
            fpath = (
                paths.deriv
                / "connectivity"
                / sub_id(label)
                / f"task-{task}"
                / f"{sub_id(label)}_task-{task}_pac_painmatrix.h5"
            )

            if not fpath.exists() or fpath.stat().st_size == 0:
                for direction in directions:
                    for combo in combos:
                        data[task][direction][combo].append(np.nan)
                continue

            with h5py.File(fpath, "r") as f:
                for direction in directions:
                    for combo in combos:
                        try:
                            val = f[direction][combo][CONDITION_KEY].attrs.get(
                                metric, np.nan
                            )
                            val = float(val)
                            if np.isnan(val):
                                val = np.nan
                        except (KeyError, TypeError):
                            val = np.nan
                        data[task][direction][combo].append(val)
            n_found += 1

    # Convert to arrays
    for task in tasks:
        for direction in directions:
            for combo in combos:
                data[task][direction][combo] = np.array(
                    data[task][direction][combo], dtype=float
                )

    print(f"Loaded PAC data ({metric}): {n_found} subject×task files found")
    return data, directions, combos


# ---------------------------------------------------------------------------
# Plot 1: Consolidated comodulogram (one figure per ROI pair)
# ---------------------------------------------------------------------------


def plot_comodulograms(
    data: dict,
    directions: list[str],
    combos: list[str],
    phase_bands: list[str],
    amp_bands: list[str],
    tasks: list[str],
    out_dir: Path,
    metric: str = "z_score",
    subject_label: str | None = None,
) -> None:
    """One figure per ROI pair showing the full PAC picture.

    Layout:
        rows = directions  (A→B top, B→A bottom)
        cols = tasks       (laser | pinprick | tactile)
        each cell = phase × amplitude comodulogram heatmap

    Produces 1 PNG per ROI pair — 10 files for 5 ROIs.
    """
    metric_label = "z-score" if metric == "z_score" else "MI"
    subj_str = f"sub-{subject_label}  |  " if subject_label else ""
    n_phase = len(phase_bands)
    n_amp = len(amp_bands)
    dir_pairs = _unidirectional_pairs(directions)

    for dir_ab, dir_ba in dir_pairs:
        rois_ab = dir_ab.split("_to_")
        pair_tag = f"{rois_ab[0]}-{rois_ab[1]}"

        n_rows = 2  # A→B and B→A
        n_cols = len(tasks)

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.5 * n_cols, 4.5 * n_rows),
            squeeze=False,
        )

        # Shared colour scale across both directions and all tasks
        all_vals = []
        for direc in [dir_ab, dir_ba]:
            for task in tasks:
                for combo in combos:
                    vals = data[task][direc][combo]
                    all_vals.extend(vals[~np.isnan(vals)].tolist())
        if all_vals:
            vmin = np.percentile(all_vals, 5)
            vmax = np.percentile(all_vals, 95)
        else:
            vmin, vmax = 0.0, 1.0

        for ri, (direc, row_label) in enumerate(
            [
                (dir_ab, f"{rois_ab[0]} → {rois_ab[1]}\n(phase drives amplitude)"),
                (dir_ba, f"{rois_ab[1]} → {rois_ab[0]}\n(phase drives amplitude)"),
            ]
        ):
            for ci, task in enumerate(tasks):
                ax = axes[ri, ci]

                # Build phase × amplitude matrix
                mat = np.full((n_phase, n_amp), np.nan)
                for pi, p_name in enumerate(phase_bands):
                    for ai, a_name in enumerate(amp_bands):
                        combo = f"phase_{p_name}_amp_{a_name}"
                        vals = data[task][direc][combo]
                        valid = vals[~np.isnan(vals)]
                        if len(valid) > 0:
                            mat[pi, ai] = float(np.mean(valid))

                im = ax.imshow(
                    mat,
                    aspect="auto",
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    cmap="YlOrRd",
                )

                # Grey out NaN cells
                nan_mask = np.isnan(mat)
                if nan_mask.any():
                    grey = np.zeros((*mat.shape, 4))
                    grey[nan_mask] = [0.85, 0.85, 0.85, 1.0]
                    ax.imshow(grey, aspect="auto", origin="lower")

                # Value annotations
                for pi in range(n_phase):
                    for ai in range(n_amp):
                        if not np.isnan(mat[pi, ai]):
                            ax.text(
                                ai,
                                pi,
                                f"{mat[pi, ai]:.2f}",
                                ha="center",
                                va="center",
                                fontsize=8,
                                color="black",
                            )

                ax.set_xticks(range(n_amp))
                ax.set_xticklabels(
                    [
                        f"{amp_bands[i]}\n({PAC_AMP_BANDS[amp_bands[i]][0]}–{PAC_AMP_BANDS[amp_bands[i]][1]} Hz)"
                        for i in range(n_amp)
                    ],
                    fontsize=8,
                )
                ax.set_yticks(range(n_phase))
                ax.set_yticklabels(
                    [
                        f"{phase_bands[i]}\n({PAC_PHASE_BANDS[phase_bands[i]][0]}–{PAC_PHASE_BANDS[phase_bands[i]][1]} Hz)"
                        for i in range(n_phase)
                    ],
                    fontsize=8,
                )

                if ri == 0:
                    ax.set_title(
                        TASK_LABELS[task],
                        fontsize=11,
                        fontweight="bold",
                        color=TASK_COLORS[task],
                        pad=8,
                    )
                if ci == 0:
                    ax.set_ylabel(row_label, fontsize=9)
                if ri == n_rows - 1:
                    ax.set_xlabel("Amplitude band", fontsize=9)

            # Shared colorbar per row — placed outside subplots
            cbar_ax = fig.add_axes([0.91, 0.55 - ri * 0.45, 0.02, 0.35])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(metric_label, fontsize=9)

        fig.suptitle(
            f"PAC comodulogram  —  {subj_str}{rois_ab[0]} ↔ {rois_ab[1]}",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout(rect=[0, 0, 0.88, 1])

        subj_prefix = f"sub-{subject_label}_" if subject_label else ""
        out_path = out_dir / f"{subj_prefix}comodulogram_{pair_tag}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 2: Directionality contrast  (A→B vs B→A per band combo per task)
# ---------------------------------------------------------------------------


def plot_directionality(
    data: dict,
    directions: list[str],
    combos: list[str],
    tasks: list[str],
    out_dir: Path,
    metric: str = "z_score",
    subject_label: str | None = None,
) -> None:
    """Bar chart comparing both directions for each ROI pair.

    One figure per unordered pair.
    Layout: rows = band combos, cols = tasks.
    Each cell = A→B vs B→A bars with SEM.
    Produces 1 PNG per ROI pair — 10 files for 5 ROIs.
    """
    metric_label = "z-score" if metric == "z_score" else "MI"
    subj_str = f"sub-{subject_label}  |  " if subject_label else ""
    subj_prefix = f"sub-{subject_label}_" if subject_label else ""
    dir_pairs = _unidirectional_pairs(directions)

    for dir_ab, dir_ba in dir_pairs:
        rois_ab = dir_ab.split("_to_")
        pair_tag = "-".join(rois_ab)

        n_rows = len(combos)
        n_cols = len(tasks)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.0 * n_cols, 3.5 * n_rows),
            squeeze=False,
            sharey="row",
        )

        for ri, combo in enumerate(combos):
            combo_label = combo.replace("phase_", "").replace("_amp_", " → ")

            for ci, task in enumerate(tasks):
                ax = axes[ri, ci]
                vals_ab = data[task][dir_ab][combo]
                vals_ba = data[task][dir_ba][combo]
                n_ab = int(np.sum(~np.isnan(vals_ab)))
                n_ba = int(np.sum(~np.isnan(vals_ba)))

                means = [np.nanmean(vals_ab), np.nanmean(vals_ba)]
                sems = [
                    np.nanstd(vals_ab) / max(1, n_ab**0.5),
                    np.nanstd(vals_ba) / max(1, n_ba**0.5),
                ]
                labels = [
                    f"{rois_ab[0]} → {rois_ab[1]}",
                    f"{rois_ab[1]} → {rois_ab[0]}",
                ]

                for xi, (mean, sem, alpha) in enumerate(zip(means, sems, [0.85, 0.45])):
                    ax.bar(
                        xi,
                        mean,
                        yerr=sem,
                        width=0.55,
                        color=TASK_COLORS[task],
                        alpha=alpha,
                        capsize=3,
                        error_kw={"elinewidth": 1.2},
                    )

                # Individual dots
                for xi, vals in enumerate([vals_ab, vals_ba]):
                    valid = vals[~np.isnan(vals)]
                    jitter = np.random.default_rng(0).uniform(
                        -0.1, 0.1, size=len(valid)
                    )
                    ax.scatter(
                        np.full(len(valid), xi) + jitter,
                        valid,
                        color="black",
                        alpha=0.5,
                        s=14,
                        zorder=3,
                    )

                ax.axhline(0, color="grey", linewidth=0.7, linestyle="--")
                ax.set_xticks([0, 1])
                ax.set_xticklabels(labels, fontsize=7, rotation=20, ha="right")

                if ri == 0:
                    ax.set_title(
                        TASK_LABELS[task],
                        fontsize=11,
                        fontweight="bold",
                        color=TASK_COLORS[task],
                    )
                if ci == 0:
                    ax.set_ylabel(
                        f"{combo_label} Hz\n{metric_label}",
                        fontsize=8,
                    )

        fig.suptitle(
            f"PAC directionality — {subj_str}{' ↔ '.join(rois_ab)}",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()

        out_path = out_dir / f"{subj_prefix}directionality_{pair_tag}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Plot 3: Raincloud  (subject-level metric per direction × combo across tasks)
# ---------------------------------------------------------------------------


def plot_raincloud(
    data: dict,
    direction: str,
    combo: str,
    tasks: list[str],
    out_dir: Path,
    metric: str = "z_score",
    subject_label: str | None = None,
) -> None:
    """Raincloud plot for one (direction, band combo) across tasks."""
    metric_label = "z-score" if metric == "z_score" else "MI"
    subj_str = f"sub-{subject_label}  |  " if subject_label else ""
    subj_prefix = f"sub-{subject_label}_" if subject_label else ""
    fig, ax = plt.subplots(figsize=(8, 4))

    y_positions = np.arange(len(tasks), dtype=float) * 1.4
    handles = []

    for yi, task in enumerate(tasks):
        vals = data[task][direction][combo]
        valid = vals[~np.isnan(vals)]
        y = y_positions[yi]
        color = TASK_COLORS[task]

        if len(valid) >= 3:
            kde = gaussian_kde(valid, bw_method=0.4)
            x_range = np.linspace(valid.min() - 0.5, valid.max() + 0.5, 300)
            density = kde(x_range)
            density = density / density.max() * 0.45  # half-violin height
            ax.fill_between(
                x_range,
                y + 0.05,
                y + 0.05 + density,
                color=color,
                alpha=0.4,
            )
            ax.plot(x_range, y + 0.05 + density, color=color, linewidth=0.8, alpha=0.7)

        # Box plot
        if len(valid) >= 2:
            q25, med, q75 = np.percentile(valid, [25, 50, 75])
            iqr = q75 - q25
            whisker_lo = max(valid.min(), q25 - 1.5 * iqr)
            whisker_hi = min(valid.max(), q75 + 1.5 * iqr)

            ax.plot([whisker_lo, q25], [y, y], color=color, linewidth=1.2)
            ax.plot([q75, whisker_hi], [y, y], color=color, linewidth=1.2)
            ax.barh(
                y,
                q75 - q25,
                height=0.18,
                left=q25,
                color=color,
                alpha=0.7,
                edgecolor=color,
            )
            ax.plot(
                med, y, marker="|", color="white", markersize=7, markeredgewidth=1.8
            )

        # Individual dots (jittered vertically)
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.08, 0.08, size=len(valid))
        ax.scatter(valid, y - 0.25 + jitter, color=color, alpha=0.6, s=18, zorder=3)

        handles.append(mpatches.Patch(color=color, label=TASK_LABELS[task], alpha=0.7))

    ax.set_yticks(y_positions)
    ax.set_yticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_xlabel(metric_label, fontsize=10)

    dir_label = direction.replace("_to_", " → ")
    combo_label = combo.replace("phase_", "").replace("_amp_", " → ")
    ax.set_title(
        f"PAC — {subj_str}{dir_label}\n{combo_label} Hz",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")

    tag = f"{direction}_{combo}".replace(" ", "_")
    out_path = out_dir / f"{subj_prefix}raincloud_{tag}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot PAC results for the laser-pain MEG study."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"BIDS project root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        metavar="LABEL",
        help="Bare subject labels (default: all in participants.tsv)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=TASKS,
        help=f"Tasks to include (default: all — {TASKS})",
    )
    parser.add_argument(
        "--phase-bands",
        nargs="+",
        default=list(PAC_PHASE_BANDS.keys()),
        choices=list(PAC_PHASE_BANDS.keys()),
    )
    parser.add_argument(
        "--amp-bands",
        nargs="+",
        default=list(PAC_AMP_BANDS.keys()),
        choices=list(PAC_AMP_BANDS.keys()),
    )
    parser.add_argument(
        "--atlas",
        default=DEFAULT_ATLAS,
        choices=list(ATLAS_CONFIGS.keys()),
    )
    parser.add_argument(
        "--metric",
        default="z_score",
        choices=["z_score", "mi"],
        help="Which PAC metric to plot: 'z_score' (default) or raw 'mi'",
    )
    parser.add_argument(
        "--no-comodulogram",
        action="store_true",
        help="Skip comodulogram plots",
    )
    parser.add_argument(
        "--no-directionality",
        action="store_true",
        help="Skip directionality contrast plots",
    )
    parser.add_argument(
        "--raincloud",
        action="store_true",
        help="Also generate raincloud plots (off by default — produces many files)",
    )
    args = parser.parse_args()

    paths = Paths(args.root)
    logger = setup_logging(paths, "plot_pac")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks = args.tasks if args.tasks else TASKS

    logger.info("Subjects : %d  %s", len(subjects), subjects)
    logger.info("Tasks    : %s", tasks)
    logger.info("Metric   : %s", args.metric)

    out_dir = paths.log_dir() / "plots" / "group" / "pac"
    out_dir.mkdir(parents=True, exist_ok=True)

    data, directions, combos = load_pac_matrix(
        paths,
        subjects,
        tasks,
        args.phase_bands,
        args.amp_bands,
        args.atlas,
        args.metric,
    )

    # Filter to directions with at least some data
    active_dirs = [
        d
        for d in directions
        if any(not np.all(np.isnan(data[t][d][c])) for t in tasks for c in combos)
    ]
    logger.info("Active directions: %d", len(active_dirs))

    if not active_dirs:
        logger.error("No PAC data found — run pac.py first")
        return

    if not args.no_comodulogram:
        logger.info("Generating comodulograms ...")
        plot_comodulograms(
            data,
            active_dirs,
            combos,
            args.phase_bands,
            args.amp_bands,
            tasks,
            out_dir,
            args.metric,
        )

    if not args.no_directionality:
        logger.info("Generating directionality contrast plots ...")
        plot_directionality(
            data,
            active_dirs,
            combos,
            tasks,
            out_dir,
            args.metric,
        )

    if args.raincloud:
        logger.info("Generating raincloud plots ...")
        for direction in active_dirs:
            for combo in combos:
                logger.info("Raincloud: %s / %s", direction, combo)
                plot_raincloud(data, direction, combo, tasks, out_dir, args.metric)

    logger.info("All plots saved to: %s", out_dir)


if __name__ == "__main__":
    main()
