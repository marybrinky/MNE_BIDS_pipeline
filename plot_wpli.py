#!/usr/bin/env python3
"""
plot_wpli.py
------------
Visualisation of WPLI connectivity results for the laser-pain MEG study.

Reads HDF5 output from wpli.py and produces five plot types:

1. --plot-circle   (per subject)
   Connectivity circle — MNE-style polar plot, one panel per task.
   Colour encodes raw WPLI value (0–1).

2. --plot-topo     (per subject)
   Anatomical head layout — ROIs at approximate anatomical positions,
   line thickness and colour encode raw WPLI.

3. --plot-heatmap  (group)
   Horizontal bar chart overview — mean ± SEM per ROI pair, rows = pairs,
   cols = tasks. Annotated with mean value on each bar.

4. --plot-raincloud (group)
   Half-violin + box + individual dots per task for each ROI pair × band.
   Standard Allen et al. (2019) raincloud style.

5. --plot-group    (group)
   Grouped bar chart — mean ± SEM with individual subject dots overlaid.

All group plots saved to:
    derivatives/logs/plots/group/wpli/

Per-subject plots shown interactively (plt.show()).

Usage
-----
    python code/plot_wpli.py --root $MEGROOT --subjects 4382 --plot-circle
    python code/plot_wpli.py --root $MEGROOT --subjects 4382 --plot-topo
    python code/plot_wpli.py --root $MEGROOT --plot-heatmap
    python code/plot_wpli.py --root $MEGROOT --plot-raincloud
    python code/plot_wpli.py --root $MEGROOT --plot-group
    python code/plot_wpli.py --root $MEGROOT --plot-heatmap --plot-raincloud --plot-group
"""

import argparse
from pathlib import Path

import h5py
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mne_connectivity.viz import plot_connectivity_circle
from scipy.stats import gaussian_kde, sem as scipy_sem

from core import (
    ATLAS_CONFIGS,
    DEFAULT_ATLAS,
    TASKS,
    Paths,
    get_roi_hemisphere_labels,
    load_subjects,
    setup_logging,
    sub_id,
)
from wpli import FREQ_BANDS, _exists, _get_roi_pairs

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")
CONDITION_KEY = "stimulus"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

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
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         150,
})

# Approximate anatomical positions for topo plot
# x: left(-) to right(+),  y: posterior(-) to anterior(+)
ROI_POSITIONS: dict[str, tuple[float, float]] = {
    "SI_l":       (-0.30, -0.65),
    "SII_r":      ( 0.88, -0.05),
    "SII_l":      (-0.88, -0.05),
    "Insula_r":   ( 0.65,  0.35),
    "Insula_l":   (-0.65,  0.35),
    "Ins_post_r": ( 0.88, -0.38),
    "Ins_post_l": (-0.88, -0.38),
    "Ins_ant_r":  ( 0.88,  0.28),
    "Ins_ant_l":  (-0.88,  0.28),
    "MCC_r":      ( 0.25,  0.18),
    "MCC_l":      (-0.25,  0.18),
    "ACC_r":      ( 0.25,  0.60),
    "ACC_l":      (-0.25,  0.60),
}

ROI_SHORT_LABELS: dict[str, str] = {
    "SI_l":       "SI",
    "SII_r":      "S2r",
    "SII_l":      "S2l",
    "Insula_r":   "Ir",
    "Insula_l":   "Il",
    "Ins_post_r": "pInsr",
    "Ins_post_l": "pInsl",
    "Ins_ant_r":  "aInsr",
    "Ins_ant_l":  "aInsl",
    "MCC_r":      "MCCr",
    "MCC_l":      "MCCl",
    "ACC_r":      "ACCr",
    "ACC_l":      "ACCl",
}


# ---------------------------------------------------------------------------
# Screen size helper
# ---------------------------------------------------------------------------


def _screen_inches() -> tuple[float, float]:
    try:
        import tkinter as _tk
        _r = _tk.Tk(); _r.withdraw()
        px_w = _r.winfo_screenwidth()
        px_h = _r.winfo_screenheight()
        dpi  = _r.winfo_fpixels("1i")
        _r.destroy()
        return (px_w / dpi) * 0.92, (px_h / dpi) * 0.88
    except Exception:
        return 16.0, 9.0


def _fit_figure_to_screen(fig) -> None:
    try:
        import tkinter as _tk
        _r = _tk.Tk(); _r.withdraw()
        px_w = _r.winfo_screenwidth()
        px_h = _r.winfo_screenheight()
        dpi  = _r.winfo_fpixels("1i")
        _r.destroy()
        sw = (px_w / dpi) * 0.92
        sh = (px_h / dpi) * 0.88
    except Exception:
        sw, sh = 16.0, 9.0

    fw, fh  = fig.get_size_inches()
    scale   = min(sw / fw, sh / fh, 1.0)
    if scale < 1.0:
        fig.set_size_inches(fw * scale, fh * scale, forward=True)


# ---------------------------------------------------------------------------
# Shared HDF5 reader
# ---------------------------------------------------------------------------


def _read_wpli_value(band_grp, cond: str) -> float:
    """Read raw WPLI from an HDF5 band group.

    Prefers the sub-group wpli attr; falls back to the flat band-level attr.
    """
    if cond in band_grp:
        val = band_grp[cond].attrs.get("wpli", np.nan)
        return float(val) if val is not None else np.nan
    # Flat attr fallback (written as -1.0 when NaN)
    val = band_grp.attrs.get(cond, -1.0)
    v   = float(val)
    return v if v >= 0 else np.nan


def load_wpli_matrix(
    paths: Paths,
    subjects: list[str],
    tasks: list[str],
    bands: list[str],
    atlas_key: str,
) -> tuple[dict, list[str]]:
    """Load WPLI values from HDF5 files into a nested dict.

    Returns
    -------
    data[task][pair][band] = np.ndarray shape (n_subjects,)
    all_pairs              = list of pair strings
    """
    atlas_cfg = ATLAS_CONFIGS[atlas_key]
    roi_names = list(atlas_cfg["rois"].keys())
    all_pairs = [f"{a}-{b}" for a, b in _get_roi_pairs(roi_names)]

    data: dict = {
        task: {pair: {band: [] for band in bands} for pair in all_pairs}
        for task in tasks
    }
    n_found = 0

    for label in subjects:
        for task in tasks:
            fpath = (
                paths.deriv / "connectivity" / sub_id(label) / f"task-{task}"
                / f"{sub_id(label)}_task-{task}_wpli_painmatrix.h5"
            )
            if not fpath.exists() or fpath.stat().st_size == 0:
                for pair in all_pairs:
                    for band in bands:
                        data[task][pair][band].append(np.nan)
                continue

            with h5py.File(fpath, "r") as f:
                for pair in all_pairs:
                    for band in bands:
                        try:
                            band_grp  = f[pair][band]
                            cond_keys = [k for k in band_grp.attrs.keys()
                                         if not k.startswith("_")]
                            cond = cond_keys[0] if cond_keys else CONDITION_KEY
                            val  = _read_wpli_value(band_grp, cond)
                        except (KeyError, TypeError):
                            val = np.nan
                        data[task][pair][band].append(val)
            n_found += 1

    for task in tasks:
        for pair in all_pairs:
            for band in bands:
                data[task][pair][band] = np.array(data[task][pair][band], dtype=float)

    print(f"Loaded WPLI: {n_found} subject×task files")
    return data, all_pairs


def _collect_subject_h5(
    paths: Paths, label: str, tasks: list[str], bands: list[str]
) -> tuple[dict[str, Path], list[float]]:
    """Return {task: h5_path} and list of all WPLI values for colour scaling."""
    task_data:  dict[str, Path] = {}
    all_values: list[float]     = []
    for task in tasks:
        h5 = (paths.deriv / "connectivity" / sub_id(label) / f"task-{task}"
              / f"{sub_id(label)}_task-{task}_wpli_painmatrix.h5")
        if not _exists(h5):
            continue
        task_data[task] = h5
        with h5py.File(h5, "r") as f:
            for pair in [k for k in f.keys() if k != "roi_time_courses"]:
                for band in bands:
                    if band not in f[pair]:
                        continue
                    bg = f[pair][band]
                    cond_keys = [k for k in bg.attrs.keys() if not k.startswith("_")]
                    if not cond_keys:
                        continue
                    v = _read_wpli_value(bg, cond_keys[0])
                    if not np.isnan(v):
                        all_values.append(v)
    return task_data, all_values


def _wpli_scale(all_values: list[float]):
    """Return (vmin, vmax, norm, cmap) for WPLI [0, 1] data."""
    vmin = max(0.0, float(np.min(all_values)) - 0.01)
    vmax = min(1.0, float(np.max(all_values)) + 0.01)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd
    return vmin, vmax, norm, cmap


# ---------------------------------------------------------------------------
# Plot 1: Connectivity circle  (per subject)
# ---------------------------------------------------------------------------


def plot_wpli_circle(
    paths: Paths,
    label: str,
    tasks: list[str],
    bands: list[str],
    logger,
    atlas_key: str = DEFAULT_ATLAS,
) -> None:
    """MNE connectivity circle — one figure per band, one column per task."""
    hemi_labels = get_roi_hemisphere_labels(atlas_key)

    task_data, all_values = _collect_subject_h5(paths, label, tasks, bands)
    if not task_data or not all_values:
        logger.warning("[sub-%s]  No valid WPLI values for circle plot", label)
        return

    vmin, vmax, _, _ = _wpli_scale(all_values)
    available_tasks = list(task_data.keys())

    for band in bands:
        panel_w = min(5.5, 16.0 / len(available_tasks))
        fig, axes = plt.subplots(
            1, len(available_tasks),
            figsize=(panel_w * len(available_tasks), panel_w),
            subplot_kw=dict(polar=True),
        )
        if len(available_tasks) == 1:
            axes = [axes]

        fig.canvas.manager.set_window_title(f"sub-{label} | {band} | WPLI")

        for ax, task in zip(axes, available_tasks):
            with h5py.File(task_data[task], "r") as f:
                pairs = [k for k in f.keys() if k != "roi_time_courses"]
                roi_names: list[str] = []
                for pair in pairs:
                    a, b = pair.split("-", 1)
                    if a not in roi_names: roi_names.append(a)
                    if b not in roi_names: roi_names.append(b)
                node_labels = [ROI_SHORT_LABELS.get(r, hemi_labels.get(r, r)) for r in roi_names]

                if band not in f[pairs[0]]:
                    ax.set_visible(False)
                    continue

                bg0       = f[pairs[0]][band]
                cond_keys = [k for k in bg0.attrs.keys() if not k.startswith("_")]
                cond      = cond_keys[0] if cond_keys else CONDITION_KEY

                con = np.zeros((len(roi_names), len(roi_names)))
                for pair in pairs:
                    if band not in f[pair]: continue
                    val = _read_wpli_value(f[pair][band], cond)
                    if np.isnan(val): continue
                    a, b = pair.split("-", 1)
                    if a not in roi_names or b not in roi_names: continue
                    i, j = roi_names.index(a), roi_names.index(b)
                    con[i, j] = con[j, i] = val

            plot_connectivity_circle(
                con, node_names=node_labels, ax=ax,
                show=False, vmin=vmin, vmax=vmax, colormap="hot_r",
            )
            ax.set_title(task, color="black", pad=20, fontsize=11, fontweight="bold")

        fig.suptitle(
            f"sub-{label}  |  {band} ({FREQ_BANDS[band][0]}–{FREQ_BANDS[band][1]} Hz)"
            f"  |  WPLI: {vmin:.3f}–{vmax:.3f}",
            fontsize=10, y=0.98,
        )
        plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2: Topographic head layout  (per subject)
# ---------------------------------------------------------------------------


def plot_wpli_topo(
    paths: Paths,
    label: str,
    tasks: list[str],
    bands: list[str],
    logger,
    atlas_key: str = DEFAULT_ATLAS,
) -> None:
    """Anatomical head layout — ROIs at approximate anatomical positions."""
    hemi_labels = get_roi_hemisphere_labels(atlas_key)

    task_data, all_values = _collect_subject_h5(paths, label, tasks, bands)
    if not task_data or not all_values:
        logger.warning("[sub-%s]  No valid WPLI values for topo plot", label)
        return

    vmin, vmax, norm, cmap = _wpli_scale(all_values)
    available_tasks = list(task_data.keys())

    for band in bands:
        panel_w = min(5.0, 15.0 / len(available_tasks))
        fig, axes = plt.subplots(1, len(available_tasks),
                                 figsize=(panel_w * len(available_tasks), panel_w * 0.9))
        if len(available_tasks) == 1:
            axes = [axes]
        fig.canvas.manager.set_window_title(f"Topo WPLI | sub-{label} | {band}")

        for ax, task in zip(axes, available_tasks):
            ax.set_aspect("equal")
            ax.axis("off")
            ax.add_patch(plt.Circle((0,0), 1.0, fill=False, color="black", linewidth=1.5))
            ax.plot([0.0,-0.08,0.08,0.0], [1.15,1.0,1.0,1.15], color="black", linewidth=1.5, zorder=2)
            for cx, t1, t2 in [(-1.02,90,270),(1.02,270,90)]:
                ax.add_patch(mpatches.Arc((cx,0),0.1,0.2,angle=0,theta1=t1,theta2=t2,color="black",lw=1.5))

            with h5py.File(task_data[task], "r") as f:
                pairs = [k for k in f.keys() if k != "roi_time_courses"]
                roi_names_in_file: list[str] = []
                for pair in pairs:
                    a, b = pair.split("-", 1)
                    if a not in roi_names_in_file: roi_names_in_file.append(a)
                    if b not in roi_names_in_file: roi_names_in_file.append(b)

                if band not in f[pairs[0]]:
                    ax.set_title(f"{task}\n(no {band} data)", fontsize=9)
                    continue

                bg0       = f[pairs[0]][band]
                cond_keys = [k for k in bg0.attrs.keys() if not k.startswith("_")]
                cond      = cond_keys[0] if cond_keys else CONDITION_KEY

                for pair in pairs:
                    if band not in f[pair]: continue
                    val = _read_wpli_value(f[pair][band], cond)
                    if np.isnan(val): continue
                    a, b = pair.split("-", 1)
                    if a not in ROI_POSITIONS or b not in ROI_POSITIONS: continue
                    x1, y1 = ROI_POSITIONS[a]
                    x2, y2 = ROI_POSITIONS[b]
                    color    = cmap(norm(val))
                    lw       = 0.4 + 3.0 * (val / max(vmax, 1e-6))
                    alpha    = max(0.15, min(0.9, 0.2 + 0.7 * (val / max(vmax, 1e-6))))
                    ax.plot([x1,x2],[y1,y2], color=color, linewidth=lw,
                            alpha=alpha, solid_capstyle="round", zorder=1)

            for roi_name in roi_names_in_file:
                if roi_name not in ROI_POSITIONS: continue
                x, y = ROI_POSITIONS[roi_name]
                short = ROI_SHORT_LABELS.get(roi_name, roi_name)
                ax.plot(x, y, "o", markersize=14, color="steelblue",
                        zorder=2, markeredgecolor="black", markeredgewidth=1.2)
                ax.text(x, y + 0.10, short, fontsize=8, ha="center", va="bottom",
                        color="black", fontweight="bold", zorder=4,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.3, 1.3)
            ax.set_title(task, fontsize=10, fontweight="bold", pad=8)

        fig.subplots_adjust(bottom=0.15)
        cbar_ax = fig.add_axes([0.25, 0.05, 0.50, 0.03])
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("WPLI", fontsize=9)

        fig.suptitle(
            f"sub-{label}  |  {band} ({FREQ_BANDS[band][0]}–{FREQ_BANDS[band][1]} Hz)"
            f"  |  WPLI: {vmin:.3f}–{vmax:.3f}",
            fontsize=10, y=0.98,
        )
        plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3: Heatmap overview  (group)
# ---------------------------------------------------------------------------


def plot_heatmap_overview(
    data: dict,
    all_pairs: list[str],
    tasks: list[str],
    bands: list[str],
    out_dir: Path,
    subjects: list[str] | None = None,
) -> None:
    """Horizontal bar chart — mean ± SEM per pair, rows=pairs, cols=tasks×bands."""
    n_pairs  = len(all_pairs)
    n_rows   = len(bands)
    n_cols   = len(tasks)
    label_fs = max(6, min(10, int(220 / n_pairs)))

    fig_w = min(max(4 * n_cols, 10), 14)
    fig_h = min(max(2.5, n_pairs * 0.30) * n_rows, 7)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        sharey=True, sharex=False,
    )
    if n_rows == 1: axes = axes[np.newaxis, :]
    if n_cols == 1: axes = axes[:, np.newaxis]

    all_vals = [
        v for task in tasks for pair in all_pairs for band in bands
        for v in data[task][pair][band] if not np.isnan(v)
    ]
    if not all_vals:
        print("No data for heatmap overview.")
        return

    vmax = min(1.0, np.percentile(all_vals, 95))
    xlim = (0, vmax * 1.2)

    for ri, band in enumerate(bands):
        for ci, task in enumerate(tasks):
            ax    = axes[ri, ci]
            means = [np.nanmean(data[task][p][band]) for p in all_pairs]
            ns    = [int(np.sum(~np.isnan(data[task][p][band]))) for p in all_pairs]
            sems  = [np.nanstd(data[task][p][band]) / max(1, ns[i]**0.5)
                     for i, p in enumerate(all_pairs)]

            y_pos = np.arange(n_pairs)
            bars  = ax.barh(
                y_pos, means, xerr=sems, height=0.65,
                color=TASK_COLORS[task], alpha=0.75,
                error_kw={"elinewidth": 1.0, "capsize": 2},
            )
            for bar, mean, n in zip(bars, means, ns):
                if not np.isnan(mean):
                    ax.text(
                        mean + vmax * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{mean:.3f}",
                        va="center", ha="left",
                        fontsize=max(5, label_fs-1), color="dimgray",
                    )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(all_pairs, fontsize=label_fs)
            ax.set_xlim(*xlim)
            ax.set_xlabel("WPLI", fontsize=9)
            ax.set_title(
                f"{TASK_LABELS[task]}  —  {band} ({FREQ_BANDS[band][0]}–{FREQ_BANDS[band][1]} Hz)",
                fontsize=10, fontweight="bold", color=TASK_COLORS[task], pad=6,
            )
            ax.invert_yaxis()
            ax.tick_params(axis="y", length=0)

    n_sub = len(subjects) if subjects else 0
    subj_str = ", ".join(f"sub-{s}" for s in subjects) if subjects else "unknown"
    fig.suptitle(
        f"WPLI — Pain matrix connectivity  |  N={n_sub}\n"
        f"Included: {subj_str}",
        fontsize=10, fontweight="bold", y=1.005
    )
    plt.tight_layout(h_pad=1.5, w_pad=1.0)

    out_path = out_dir / "heatmap_overview.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path.name}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Raincloud  (group)
# ---------------------------------------------------------------------------


def _half_violin(ax, data_1d, x_pos, color, width=0.25, side="right"):
    clean = data_1d[~np.isnan(data_1d)]
    if len(clean) < 3:
        return
    kde     = gaussian_kde(clean, bw_method="scott")
    y_range = np.linspace(clean.min()-0.01, clean.max()+0.01, 200)
    density = kde(y_range)
    density = density / density.max() * width
    if side == "right":
        ax.fill_betweenx(y_range, x_pos, x_pos+density, color=color, alpha=0.35, linewidth=0)
        ax.plot(x_pos+density, y_range, color=color, linewidth=0.8)
    else:
        ax.fill_betweenx(y_range, x_pos-density, x_pos, color=color, alpha=0.35, linewidth=0)
        ax.plot(x_pos-density, y_range, color=color, linewidth=0.8)


def plot_raincloud(
    data: dict,
    all_pairs: list[str],
    tasks: list[str],
    bands: list[str],
    out_dir: Path,
    subjects: list[str] | None = None,
) -> None:
    """Raincloud plots — one PNG per ROI pair × band."""
    for pair in all_pairs:
        for band in bands:
            fig, ax = plt.subplots(figsize=(6, 4))
            x_positions = {task: i for i, task in enumerate(tasks)}
            rng         = np.random.default_rng(42)

            for task in tasks:
                x     = x_positions[task]
                vals  = data[task][pair][band]
                clean = vals[~np.isnan(vals)]
                if len(clean) == 0:
                    continue
                color = TASK_COLORS[task]

                _half_violin(ax, clean, x, color, width=0.3, side="right")

                ax.boxplot(
                    clean, positions=[x-0.15], widths=0.12,
                    patch_artist=True, showfliers=False,
                    medianprops={"color":"white","linewidth":2},
                    boxprops={"facecolor":color,"alpha":0.7},
                    whiskerprops={"color":color}, capprops={"color":color},
                )

                jitter = rng.uniform(-0.06, 0.06, size=len(clean))
                ax.scatter(x-0.32+jitter, clean, color=color, alpha=0.7,
                           s=25, zorder=3, edgecolors="white", linewidths=0.5)

            ax.set_xticks(list(x_positions.values()))
            ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=11)
            ax.set_ylabel("WPLI", fontsize=11)
            ax.set_ylim(bottom=0)
            n_sub = len(subjects) if subjects else 0
            subj_str = ", ".join(f"sub-{s}" for s in subjects) if subjects else "unknown"
            ax.set_title(
                f"{pair}   |   {band} ({FREQ_BANDS[band][0]}–{FREQ_BANDS[band][1]} Hz)"
                f"  |  N={n_sub}\nIncluded: {subj_str}",
                fontsize=9, fontweight="bold",
            )
            ax.set_xlim(-0.6, len(tasks)-0.4)

            patches = [mpatches.Patch(color=TASK_COLORS[t], label=TASK_LABELS[t], alpha=0.75)
                       for t in tasks]
            ax.legend(handles=patches, loc="upper right", fontsize=9, framealpha=0.5)
            plt.tight_layout()

            safe_pair = pair.replace(" ", "_")
            out_path  = out_dir / f"raincloud_{safe_pair}_{band}.png"
            fig.savefig(out_path, bbox_inches="tight", facecolor="white")
            print(f"Saved: {out_path.name}")
            plt.show()
            plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Group bar chart with subject dots  (group)
# ---------------------------------------------------------------------------


def plot_wpli_group(
    data: dict,
    all_pairs: list[str],
    tasks: list[str],
    bands: list[str],
    subjects: list[str],
    out_dir: Path,
) -> None:
    """Grouped bar chart — mean ± SEM per pair, individual subject dots overlaid."""
    for band in bands:
        n_pairs   = len(all_pairs)
        n_tasks   = len(tasks)
        bar_width = 0.22
        x         = np.arange(n_pairs)

        fig_w = min(max(7, n_pairs * 1.2), 14)
        fig, ax = plt.subplots(figsize=(fig_w, 4.5))

        for t_idx, task in enumerate(tasks):
            means, sems = [], []
            for pair in all_pairs:
                vals = data[task][pair][band]
                v    = vals[~np.isnan(vals)]
                means.append(float(np.mean(v)) if len(v) > 0 else 0.0)
                sems.append(float(scipy_sem(v)) if len(v) > 1 else 0.0)

            offset = (t_idx - n_tasks/2 + 0.5) * bar_width
            bar_x  = x + offset
            color  = TASK_COLORS.get(task, f"C{t_idx}")

            ax.bar(bar_x, means, bar_width, label=TASK_LABELS[task], color=color,
                   alpha=0.75, yerr=sems, capsize=4,
                   error_kw=dict(elinewidth=1.2, ecolor="black"))

            rng = np.random.default_rng(t_idx)
            for pi, pair in enumerate(all_pairs):
                v = data[task][pair][band]
                v = v[~np.isnan(v)]
                if len(v) == 0: continue
                jitter = rng.uniform(-bar_width*0.3, bar_width*0.3, size=len(v))
                ax.scatter(np.full(len(v), bar_x[pi])+jitter, v,
                           color=color, s=18, alpha=0.7, zorder=3, edgecolors="none")

        ax.set_ylim(bottom=0)
        ax.set_xticks(x)
        ax.set_xticklabels(all_pairs, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("WPLI (mean ± SEM)")
        ax.set_xlabel("ROI pair")
        ax.legend(title="Task", loc="upper right")

        n_included = len(subjects)
        subj_str = ", ".join(f"sub-{s}" for s in subjects)
        ax.set_title(
            f"Group WPLI  |  {band} ({FREQ_BANDS[band][0]}–{FREQ_BANDS[band][1]} Hz)"
            f"  |  N={n_included}\n"
            f"Included: {subj_str}",
            fontsize=9,
        )
        fig.canvas.manager.set_window_title(f"Group WPLI | {band}")
        plt.tight_layout()

        out_path = out_dir / f"group_bar_{band}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_path.name}")
        plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Plot WPLI connectivity results — laser-pain MEG study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Plot arguments (combine freely):\n"
            "  --plot-circle      Per-subject connectivity circle (interactive)\n"
            "  --plot-topo        Per-subject anatomical head layout (interactive)\n"
            "  --plot-heatmap     Group horizontal bar chart overview (saved PNG)\n"
            "  --plot-raincloud   Group raincloud per pair × band (saved PNG)\n"
            "  --plot-group       Group bar chart with subject dots (saved PNG)\n"
        ),
    )
    parser.add_argument("--root",     type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--tasks",    nargs="+", default=None, choices=TASKS)
    parser.add_argument("--bands",    nargs="+", default=list(FREQ_BANDS.keys()),
                        choices=list(FREQ_BANDS.keys()))
    parser.add_argument("--atlas",    default=DEFAULT_ATLAS, choices=list(ATLAS_CONFIGS.keys()))

    parser.add_argument("--plot-circle",    action="store_true",
                        help="Per-subject connectivity circle (MNE polar plot).")
    parser.add_argument("--plot-topo",      action="store_true",
                        help="Per-subject anatomical head layout.")
    parser.add_argument("--plot-heatmap",   action="store_true",
                        help="Group horizontal bar chart overview.")
    parser.add_argument("--plot-raincloud", action="store_true",
                        help="Group raincloud plots per pair × band.")
    parser.add_argument("--plot-group",     action="store_true",
                        help="Group bar chart with individual subject dots.")
    parser.add_argument("--plot-show", action="store_true",
                        help="Open the plots folder in Finder without recomputing.")

    args = parser.parse_args()

    paths    = Paths(args.root)
    logger   = setup_logging(paths, "plot_wpli")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks    = args.tasks    if args.tasks    else TASKS

    out_dir = paths.log_dir() / "plots" / "group" / "wpli"
    out_dir.mkdir(parents=True, exist_ok=True)

    any_plot = (args.plot_circle or args.plot_topo or args.plot_heatmap
                or args.plot_raincloud or args.plot_group)

    if args.plot_show and not any_plot:
        import subprocess
        logger.info("Opening plots folder: %s", out_dir)
        subprocess.run(["open", str(out_dir)])
        return

    if args.plot_show and any_plot:
        import subprocess
        files_to_open = []
        if args.plot_heatmap:
            f = out_dir / "heatmap_overview.png"
            if f.exists(): files_to_open.append(f)
        if args.plot_group:
            for band in args.bands:
                f = out_dir / f"group_bar_{band}.png"
                if f.exists(): files_to_open.append(f)
        if args.plot_raincloud:
            files_to_open.append(out_dir)
        if args.plot_circle or args.plot_topo:
            files_to_open.append(out_dir)
        if files_to_open:
            for p in files_to_open:
                subprocess.run(["open", str(p)])
        else:
            logger.warning("No saved plots found — run without --plot-show first.")
        return

    if not any_plot:
        parser.print_help()
        return

    logger.info("Subjects : %d  %s", len(subjects), subjects)
    logger.info("Tasks    : %s", tasks)
    logger.info("Bands    : %s", args.bands)

    # ── Per-subject plots ────────────────────────────────────────────────────
    if args.plot_circle or args.plot_topo:
        for label in subjects:
            if args.plot_circle:
                logger.info("[sub-%s]  Circle plot ...", label)
                plot_wpli_circle(paths, label, tasks, args.bands, logger,
                                 atlas_key=args.atlas)
            if args.plot_topo:
                logger.info("[sub-%s]  Topo plot ...", label)
                plot_wpli_topo(paths, label, tasks, args.bands, logger,
                               atlas_key=args.atlas)

    # ── Group plots ──────────────────────────────────────────────────────────
    if args.plot_heatmap or args.plot_raincloud or args.plot_group:
        data, all_pairs = load_wpli_matrix(
            paths, subjects, tasks, args.bands, args.atlas
        )
        active_pairs = [
            p for p in all_pairs
            if any(not np.all(np.isnan(data[t][p][b]))
                   for t in tasks for b in args.bands)
        ]
        if not active_pairs:
            logger.error("No WPLI data found — run wpli.py first.")
            return
        logger.info("Active ROI pairs: %s", active_pairs)

        if args.plot_heatmap:
            logger.info("Generating heatmap overview ...")
            plot_heatmap_overview(data, active_pairs, tasks, args.bands, out_dir, subjects)

        if args.plot_raincloud:
            logger.info("Generating raincloud plots ...")
            plot_raincloud(data, active_pairs, tasks, args.bands, out_dir, subjects)

        if args.plot_group:
            logger.info("Generating group bar chart ...")
            plot_wpli_group(data, active_pairs, tasks, args.bands, subjects, out_dir)

        logger.info("Plots saved to: %s", out_dir)


if __name__ == "__main__":
    main()
