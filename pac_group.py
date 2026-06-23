#!/usr/bin/env python3
"""
pac_group.py
------------
Group-level Phase-Amplitude Coupling (PAC) analysis for the laser-pain MEG study.

Reads per-subject HDF5 files produced by pac.py, averages across subjects,
and provides outlier / artifact detection via three complementary methods:

1.  Strip plots          — subject-level dots alongside group mean ± SEM;
                           outliers flagged visually and with subject labels.
2.  Leave-one-out (LOO)  — recomputes group mean with each subject removed;
                           subjects whose removal changes the mean by > LOO_THRESHOLD
                           SDs are flagged automatically.
3.  Subject z-scoring    — subjects whose individual value lies > OUTLIER_Z SDs
                           from the group mean are flagged.

Outputs
-------
    derivatives/logs/plots/group/pac/
        group_comodulogram_<pair>.png
        group_directionality_<pair>.png
        group_stripplot_<direction>_<combo>.png
        group_loo_<direction>_<combo>.png
        group_outlier_report.tsv         ← summary of all flagged subjects

Usage
-----
    python code/pac_group.py --root $MEGROOT
    python code/pac_group.py --root $MEGROOT --subjects 4382 4383
    python code/pac_group.py --root $MEGROOT --metric mi
    python code/pac_group.py --root $MEGROOT --no-loo        # skip LOO (slow for large N)
    python code/pac_group.py --root $MEGROOT --outlier-z 2.5
"""

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
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
from pac import PAC_AMP_BANDS, PAC_PHASE_BANDS
from plot_pac import (
    TASK_COLORS,
    TASK_LABELS,
    _all_combos,
    _all_directions,
    _unidirectional_pairs,
    load_pac_matrix,
    plot_comodulograms,
    plot_directionality,
)

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

OUTLIER_Z      = 3.0   # flag subject if |z_subject| > this relative to group
LOO_THRESHOLD  = 1.0   # flag subject if LOO change > this × group SD

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size":   11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":   True,
        "grid.alpha":  0.3,
        "figure.dpi":  150,
    }
)


# ---------------------------------------------------------------------------
# Outlier detection helpers
# ---------------------------------------------------------------------------


def _flag_outliers_zscore(
    vals: np.ndarray,
    subjects: list[str],
    threshold: float = OUTLIER_Z,
) -> list[tuple[str, float]]:
    """Return (subject_label, z_value) for subjects > threshold SDs from mean.

    Parameters
    ----------
    vals      : shape (n_subjects,)  — may contain NaN (skipped)
    subjects  : matching list of subject label strings
    threshold : SD threshold

    Returns
    -------
    list of (label, z) tuples for flagged subjects
    """
    valid_mask = ~np.isnan(vals)
    if valid_mask.sum() < 3:
        return []
    mu  = np.nanmean(vals)
    sd  = np.nanstd(vals)
    if sd < 1e-12:
        return []
    flagged = []
    for i, (v, label) in enumerate(zip(vals, subjects)):
        if np.isnan(v):
            continue
        z = (v - mu) / sd
        if abs(z) > threshold:
            flagged.append((label, float(z)))
    return flagged


def _loo_influence(
    vals: np.ndarray,
    subjects: list[str],
    threshold: float = LOO_THRESHOLD,
) -> list[tuple[str, float]]:
    """Leave-one-out: flag subjects whose removal shifts the mean > threshold × SD.

    Parameters
    ----------
    vals      : shape (n_subjects,)  — may contain NaN
    subjects  : matching subject labels
    threshold : LOO_THRESHOLD (in units of group SD)

    Returns
    -------
    list of (label, delta_mean_in_SDs) tuples for flagged subjects
    """
    valid_mask = ~np.isnan(vals)
    n_valid = valid_mask.sum()
    if n_valid < 4:
        return []

    full_mean = np.nanmean(vals)
    full_sd   = np.nanstd(vals)
    if full_sd < 1e-12:
        return []

    flagged = []
    for i, label in enumerate(subjects):
        if np.isnan(vals[i]):
            continue
        loo_vals = np.concatenate([vals[:i], vals[i+1:]])
        loo_mean = np.nanmean(loo_vals)
        delta_sd = abs(loo_mean - full_mean) / full_sd
        if delta_sd > threshold:
            flagged.append((label, float(delta_sd)))
    return flagged


# ---------------------------------------------------------------------------
# Plot: group strip plot with SEM and outlier markers
# ---------------------------------------------------------------------------


def plot_group_stripplot(
    data: dict,
    direction: str,
    combo: str,
    tasks: list[str],
    subjects: list[str],
    out_dir: Path,
    metric: str = "z_score",
    outlier_z: float = OUTLIER_Z,
    loo_threshold: float = LOO_THRESHOLD,
    run_loo: bool = True,
) -> list[dict]:
    """Strip plot: each subject as a dot, group mean ± SEM, outliers labelled.

    Returns
    -------
    list of outlier-record dicts (for the group report)
    """
    metric_label = "z-score" if metric == "z_score" else "MI"
    fig, ax = plt.subplots(figsize=(9, 4.5))

    y_positions = np.arange(len(tasks), dtype=float) * 1.6
    outlier_records = []

    for yi, task in enumerate(tasks):
        vals    = data[task][direction][combo]          # shape (n_subjects,)
        y       = y_positions[yi]
        color   = TASK_COLORS[task]
        valid   = ~np.isnan(vals)
        n_valid = valid.sum()

        # --- group stats ---
        mu  = np.nanmean(vals) if n_valid > 0 else np.nan
        sem = (np.nanstd(vals) / np.sqrt(n_valid)) if n_valid > 1 else np.nan

        # --- outlier flags ---
        z_flagged   = _flag_outliers_zscore(vals, subjects, outlier_z)
        loo_flagged = _loo_influence(vals, subjects, loo_threshold) if run_loo else []
        all_flagged_labels = set(l for l, _ in z_flagged) | set(l for l, _ in loo_flagged)

        for sub_label, val in zip(subjects, vals):
            if np.isnan(val):
                continue
            is_outlier = sub_label in all_flagged_labels
            ax.scatter(
                val,
                y + np.random.default_rng(hash(sub_label) % (2**32)).uniform(-0.12, 0.12),
                color  = "#D62828" if is_outlier else color,
                s      = 55 if is_outlier else 30,
                alpha  = 0.9 if is_outlier else 0.55,
                zorder = 4,
                marker = "D" if is_outlier else "o",
            )
            if is_outlier:
                ax.annotate(
                    sub_label,
                    xy    = (val, y),
                    xytext= (4, 6),
                    textcoords = "offset points",
                    fontsize   = 7,
                    color      = "#D62828",
                )

        # --- mean ± SEM ---
        if not np.isnan(mu):
            ax.errorbar(
                mu, y - 0.38,
                xerr      = sem if not np.isnan(sem) else 0,
                fmt       = "s",
                color     = color,
                markersize= 8,
                capsize   = 5,
                linewidth = 2,
                zorder    = 5,
                label     = f"{TASK_LABELS[task]}  mean={mu:.3f}  SEM={sem:.3f}  n={n_valid}",
            )

        # --- collect outlier records ---
        for sub_label, z_val in z_flagged:
            outlier_records.append({
                "subject":   sub_label,
                "task":      task,
                "direction": direction,
                "combo":     combo,
                "method":    "z-score",
                "value":     float(vals[subjects.index(sub_label)]),
                "flag":      f"z={z_val:+.2f}",
            })
        for sub_label, delta in loo_flagged:
            outlier_records.append({
                "subject":   sub_label,
                "task":      task,
                "direction": direction,
                "combo":     combo,
                "method":    "LOO",
                "value":     float(vals[subjects.index(sub_label)]),
                "flag":      f"delta={delta:.2f}SD",
            })

    ax.set_yticks(y_positions - 0.38)
    ax.set_yticklabels([TASK_LABELS[t] for t in tasks])
    ax.set_xlabel(metric_label, fontsize=10)
    ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")

    dir_label   = direction.replace("_to_", " → ")
    combo_label = combo.replace("phase_", "").replace("_amp_", " → ")
    ax.set_title(
        f"Group PAC — {dir_label}\n{combo_label} Hz  "
        f"(◆ = outlier  |  z-threshold={outlier_z:.1f})",
        fontsize=11,
        fontweight="bold",
    )

    # legend with mean / SEM per task
    handles = [
        mpatches.Patch(color=TASK_COLORS[t], label=f"{TASK_LABELS[t]}", alpha=0.7)
        for t in tasks
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    tag      = f"{direction}_{combo}".replace(" ", "_")
    out_path = out_dir / f"group_stripplot_{tag}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path.name}")
    return outlier_records


# ---------------------------------------------------------------------------
# Plot: LOO influence bar chart
# ---------------------------------------------------------------------------


def plot_loo(
    data: dict,
    direction: str,
    combo: str,
    tasks: list[str],
    subjects: list[str],
    out_dir: Path,
    metric: str = "z_score",
    loo_threshold: float = LOO_THRESHOLD,
) -> None:
    """Bar chart of LOO delta (in SDs) per subject, one panel per task.

    Bars exceeding loo_threshold are highlighted in red.
    """
    metric_label = "z-score" if metric == "z_score" else "MI"
    n_tasks = len(tasks)
    fig, axes = plt.subplots(
        1, n_tasks,
        figsize   = (4.5 * n_tasks, max(3.5, 0.3 * len(subjects))),
        sharey    = True,
        squeeze   = False,
    )

    for ci, task in enumerate(tasks):
        ax   = axes[0, ci]
        vals = data[task][direction][combo]

        valid_mask = ~np.isnan(vals)
        n_valid    = valid_mask.sum()
        if n_valid < 4:
            ax.text(0.5, 0.5, "n < 4\nLOO skipped",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="grey")
            ax.set_title(TASK_LABELS[task], fontsize=11,
                         fontweight="bold", color=TASK_COLORS[task])
            continue

        full_mean = np.nanmean(vals)
        full_sd   = np.nanstd(vals)

        deltas = []
        labels_plot = []
        for i, label in enumerate(subjects):
            if np.isnan(vals[i]):
                continue
            loo_vals  = np.concatenate([vals[:i], vals[i+1:]])
            loo_mean  = np.nanmean(loo_vals)
            delta_sd  = (loo_mean - full_mean) / (full_sd if full_sd > 1e-12 else 1.0)
            deltas.append(delta_sd)
            labels_plot.append(label)

        y_pos  = np.arange(len(deltas))
        colors = ["#D62828" if abs(d) > loo_threshold else TASK_COLORS[task]
                  for d in deltas]

        ax.barh(y_pos, deltas, color=colors, alpha=0.75, edgecolor="none")
        ax.axvline(0,             color="grey",   linewidth=0.8, linestyle="--")
        ax.axvline( loo_threshold, color="#D62828", linewidth=0.8, linestyle=":")
        ax.axvline(-loo_threshold, color="#D62828", linewidth=0.8, linestyle=":",
                   label=f"threshold ±{loo_threshold} SD")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_plot, fontsize=7)
        ax.set_xlabel("LOO Δmean (SDs)", fontsize=9)
        ax.set_title(
            TASK_LABELS[task],
            fontsize=11,
            fontweight="bold",
            color=TASK_COLORS[task],
        )
        if ci == 0:
            ax.legend(fontsize=7, loc="lower right")

    dir_label   = direction.replace("_to_", " → ")
    combo_label = combo.replace("phase_", "").replace("_amp_", " → ")
    fig.suptitle(
        f"LOO influence — {dir_label} | {combo_label} Hz",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    tag      = f"{direction}_{combo}".replace(" ", "_")
    out_path = out_dir / f"group_loo_{tag}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Outlier report TSV
# ---------------------------------------------------------------------------


def save_outlier_report(records: list[dict], out_dir: Path) -> None:
    """Write all flagged subjects to a TSV for easy inspection."""
    if not records:
        print("No outliers flagged — report not written.")
        return

    out_path = out_dir / "group_outlier_report.tsv"
    fields   = ["subject", "task", "direction", "combo", "method", "value", "flag"]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    # Print summary to terminal
    unique_subs = sorted({r["subject"] for r in records})
    print(f"\nOutlier report: {len(records)} flags across {len(unique_subs)} subject(s)")
    print(f"Flagged subjects: {unique_subs}")
    print(f"Saved: {out_path.name}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Group-level PAC analysis: average across subjects, "
            "with strip plots, LOO analysis, and outlier detection."
        )
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
        metavar="TASK",
        help=f"Tasks to process (default: all — {TASKS})",
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
        help="PAC metric: 'z_score' (default) or raw 'mi'",
    )
    parser.add_argument(
        "--outlier-z",
        type=float,
        default=OUTLIER_Z,
        help=f"Z-score threshold for outlier flagging (default: {OUTLIER_Z})",
    )
    parser.add_argument(
        "--loo-threshold",
        type=float,
        default=LOO_THRESHOLD,
        help=(
            f"LOO threshold in group SDs (default: {LOO_THRESHOLD}). "
            "Subjects whose removal shifts the mean by more than this are flagged."
        ),
    )
    parser.add_argument(
        "--no-loo",
        action="store_true",
        help="Skip leave-one-out plots (faster for large N).",
    )
    parser.add_argument(
        "--no-comodulogram",
        action="store_true",
        help="Skip group comodulogram plots.",
    )
    parser.add_argument(
        "--no-directionality",
        action="store_true",
        help="Skip group directionality plots.",
    )
    parser.add_argument(
        "--no-stripplot",
        action="store_true",
        help="Skip strip plots.",
    )
    parser.add_argument(
        "--plot-show",
        action="store_true",
        help="Open the plots folder in Finder after saving.",
    )
    args = parser.parse_args()

    paths    = Paths(args.root)
    logger   = setup_logging(paths, "pac_group")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks    = args.tasks    if args.tasks    else TASKS

    logger.info("Subjects     : %d  %s", len(subjects), subjects)
    logger.info("Tasks        : %s",     tasks)
    logger.info("Metric       : %s",     args.metric)
    logger.info("Outlier z    : %.1f",   args.outlier_z)
    logger.info("LOO threshold: %.2f SD", args.loo_threshold)
    logger.info("Run LOO      : %s",     not args.no_loo)

    out_dir = paths.log_dir() / "plots" / "group" / "pac"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # --plot-show: just open the folder, no recomputation
    # ------------------------------------------------------------------
    if args.plot_show:
        import subprocess
        logger.info("Opening plots folder: %s", out_dir)
        subprocess.run(["open", str(out_dir)])
        return

    # ------------------------------------------------------------------
    # Load all subject PAC values
    # ------------------------------------------------------------------
    data, directions, combos = load_pac_matrix(
        paths,
        subjects,
        tasks,
        args.phase_bands,
        args.amp_bands,
        args.atlas,
        args.metric,
    )

    active_dirs = [
        d for d in directions
        if any(
            not np.all(np.isnan(data[t][d][c]))
            for t in tasks for c in combos
        )
    ]
    logger.info("Active directions: %d", len(active_dirs))

    if not active_dirs:
        logger.error("No PAC data found — run pac.py first.")
        return

    # ------------------------------------------------------------------
    # Group comodulograms (reuse plot_pac helpers, pass all subjects)
    # ------------------------------------------------------------------
    if not args.no_comodulogram:
        logger.info("Generating group comodulograms ...")
        plot_comodulograms(
            data,
            active_dirs,
            combos,
            args.phase_bands,
            args.amp_bands,
            tasks,
            out_dir,
            args.metric,
            subject_label=None,    # None → "group" title
        )

    # ------------------------------------------------------------------
    # Group directionality plots
    # ------------------------------------------------------------------
    if not args.no_directionality:
        logger.info("Generating group directionality plots ...")
        plot_directionality(
            data,
            active_dirs,
            combos,
            tasks,
            out_dir,
            args.metric,
            subject_label=None,
        )

    # ------------------------------------------------------------------
    # Strip plots + outlier detection + optional LOO
    # ------------------------------------------------------------------
    all_outlier_records: list[dict] = []

    if not args.no_stripplot:
        logger.info("Generating group strip plots ...")
        for direction in active_dirs:
            for combo in combos:
                records = plot_group_stripplot(
                    data,
                    direction,
                    combo,
                    tasks,
                    subjects,
                    out_dir,
                    metric       = args.metric,
                    outlier_z    = args.outlier_z,
                    loo_threshold= args.loo_threshold,
                    run_loo      = not args.no_loo,
                )
                all_outlier_records.extend(records)

                if not args.no_loo:
                    plot_loo(
                        data,
                        direction,
                        combo,
                        tasks,
                        subjects,
                        out_dir,
                        metric        = args.metric,
                        loo_threshold = args.loo_threshold,
                    )

    # ------------------------------------------------------------------
    # Save outlier report
    # ------------------------------------------------------------------
    save_outlier_report(all_outlier_records, out_dir)

    logger.info("All group outputs saved to: %s", out_dir)
    logger.info("─────────────────────────────────────────────")
    logger.info("Done.")




if __name__ == "__main__":
    main()
