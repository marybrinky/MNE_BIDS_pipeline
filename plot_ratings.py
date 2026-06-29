#!/usr/bin/env python3
"""
plot_ratings.py
----------------
Group-level visualisation of behavioural pain/sensation ratings.

Produces, for {laser, pinprick} (pain scale) and {tactile} (sensation
scale) separately:
  - bar plots of mean rating per subject + group mean (raw and log1p)
  - all-trials vs perceived-only (rating > 0) variants
  - a correlation plot (log-rating vs raw-rating, sanity check of the
    Stevens transform; first-half vs second-half trial correlation as
    a simple within-subject reliability check)
  - for laser only: breakdown of trial counts by quality category
    (brennend, spitz, spitz-brennend, warm, nichts)

Ratings are read the same way as epoch.py: triggercheck JSON first,
behavioural mat file as fallback.

Usage
-----
    python plot_ratings.py --root $MEGROOT
    python plot_ratings.py --root $MEGROOT --subjects 1409 3691
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAIN_TASKS = ["laser", "pinprick"]      # 0 = no pain, 100 = worst pain
SENSATION_TASKS = ["tactile"]           # 0 = nothing, 50 = first stim, 100 = max

QUALITY_LABELS = ["s", "sb", "b", "w", "n"]   # spitz, spitz-brennend, brennend, warm, nichts
QUALITY_NAMES = {
    "s": "sharp (spitz)", "sb": "sharp-burning (spitz-brennend)",
    "b": "burning (brennend)", "w": "warm (warm)", "n": "none (nichts)",
}
TASK_COLORS = {"laser": "indianred", "pinprick": "steelblue", "tactile": "seagreen"}
NONPAIN_LABELS = {"w", "n"}   # warm + nichts = non-painful


def sub_id(label: str) -> str:
    return label if label.startswith("sub-") else f"sub-{label}"


# ---------------------------------------------------------------------------
# Rating loaders (mirrors epoch.py logic)
# ---------------------------------------------------------------------------

def read_ratings_from_json(json_path: Path, task: str):
    with json_path.open() as f:
        tc = json.load(f)
    stim_key = "is_laser" if task == "laser" else "is_stim"
    ratings, qualities = [], []
    for t in tc.get("trials", []):
        if not t.get(stim_key, False):
            continue
        val = t.get("intensity_mat", t.get("intensity_fif"))
        ratings.append(None if (val is None or val == -1) else float(val))
        qualities.append(t.get("quality_fif"))
    return ratings, qualities


def _unwrap_cell(cell):
    """Recursively unwrap nested arrays/lists down to a scalar string."""
    while hasattr(cell, "__len__") and not isinstance(cell, str) and len(cell) > 0:
        try:
            cell = cell.flat[0] if hasattr(cell, "flat") else cell[0]
        except (IndexError, AttributeError):
            break
    return re.sub(r"""[\[\]'"]""", "", str(cell).strip()).strip()


def read_ratings_from_mat(mat_path: Path, task: str = "pinprick"):
    """Read intensity (and, for laser, quality) ratings from a mat file.

    For laser files, responses has shape (3, n): row 0 = intensity,
    row 1 = quality letter (s/sb/b/w/n), row 2 = unused (button colour).
    For pinprick/tactile, responses has shape (1, n): intensity only.
    """
    import scipy.io
    mat = scipy.io.loadmat(str(mat_path))
    r = mat["response"][0, 0]
    resps = r["responses"]
    n = resps.shape[1]
    has_quality = task == "laser" and resps.shape[0] >= 2

    ratings, qualities = [], []
    for i in range(n):
        val = _unwrap_cell(resps[0, i])
        if "miss" in val.lower() or val in ("", "nan"):
            ratings.append(None)
        else:
            try:
                ratings.append(float(val))
            except ValueError:
                ratings.append(None)

        if has_quality:
            qval = _unwrap_cell(resps[1, i]).lower()
            qualities.append(qval if qval in QUALITY_LABELS else None)
        else:
            qualities.append(None)
    return ratings, qualities


def load_ratings(root: Path, label: str, task: str):
    """Returns (ratings_list, qualities_list); ratings_list has None for miss."""
    json_path = (root / "derivatives" / "trigger_check" / sub_id(label)
                 / f"{sub_id(label)}_task-{task}_triggercheck.json")
    if json_path.exists():
        return read_ratings_from_json(json_path, task)

    mat_path = (root / "rawdata" / sub_id(label) / "beh"
                / f"{sub_id(label)}_task-{task}_ratings.mat")
    if mat_path.exists():
        return read_ratings_from_mat(mat_path, task)
    return [], []


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def clean(ratings):
    return np.array([r for r in ratings if r is not None], dtype=float)


def perceived_only(ratings):
    return np.array([r for r in ratings if r is not None and r > 0], dtype=float)


def log1p_transform(arr):
    return np.log1p(arr)


# ---------------------------------------------------------------------------
# Plot 1: per-subject + group bar plot (raw and log1p), all vs perceived
# ---------------------------------------------------------------------------

def plot_bar_group(root, subjects, tasks, out_dir, label_str, perceived=False):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    n_tasks = len(tasks)
    bar_width = 0.8 / max(n_tasks, 1)

    for ax, transform, unit in zip(
        axes, [lambda x: x, log1p_transform],
        ["rating (0\u2013100 NRS)", "log1p(rating)  [0\u2013100 NRS]"]
    ):
        # x_base groups by subject; tasks for the same subject sit directly
        # next to each other at that subject's x position (offset by task)
        x_base = np.arange(len(subjects))
        for ti, task in enumerate(tasks):
            means, sems = [], []
            for s in subjects:
                ratings, _ = load_ratings(root, s, task)
                arr = perceived_only(ratings) if perceived else clean(ratings)
                if arr.size == 0:
                    means.append(np.nan)
                    sems.append(0)
                    continue
                vals = transform(arr)
                means.append(vals.mean())
                sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            offset = (ti - (n_tasks - 1) / 2) * bar_width
            ax.bar(x_base + offset, means, yerr=sems, width=bar_width, capsize=2,
                   color=TASK_COLORS.get(task, "gray"), alpha=0.85, label=task)

        ax.set_xticks(x_base)
        ax.set_xticklabels(subjects, rotation=90, fontsize=6)
        ax.set_title(f"{unit.split('  [')[0]} \u2014 {label_str} "
                      f"({'perceived' if perceived else 'all trials'})")
        ax.set_ylabel(unit)
        ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.suptitle("Error bars = SEM (standard error of the mean)",
                 fontsize=9, y=0.985)
    fname = out_dir / f"ratings_bar_{label_str}_{'perceived' if perceived else 'all'}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 2: correlation — raw vs log1p (sanity), and first-half vs second-half
# ---------------------------------------------------------------------------

def plot_correlation(root, subjects, tasks, out_dir, label_str, perceived=False):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) raw vs log1p per-subject mean — shows the Stevens compression effect
    raw_means, log_means = [], []
    for task in tasks:
        for s in subjects:
            ratings, _ = load_ratings(root, s, task)
            arr = perceived_only(ratings) if perceived else clean(ratings)
            if arr.size == 0:
                continue
            raw_means.append(arr.mean())
            log_means.append(log1p_transform(arr).mean())
    axes[0].scatter(raw_means, log_means, alpha=0.7, color="darkorange")
    axes[0].set_xlabel("mean raw rating")
    axes[0].set_ylabel("mean log1p(rating)")
    axes[0].set_title("Stevens compression: raw vs log1p")

    # (b) first-half vs second-half trial mean per subject (within-session reliability)
    first_half, second_half = [], []
    for task in tasks:
        for s in subjects:
            ratings, _ = load_ratings(root, s, task)
            arr = perceived_only(ratings) if perceived else clean(ratings)
            if arr.size < 4:
                continue
            half = len(arr) // 2
            first_half.append(arr[:half].mean())
            second_half.append(arr[half:].mean())
    axes[1].scatter(first_half, second_half, alpha=0.7, color="seagreen")
    lims = [0, 100]
    axes[1].plot(lims, lims, "k--", alpha=0.4)
    axes[1].set_xlabel("mean rating — first half of trials")
    axes[1].set_ylabel("mean rating — second half of trials")
    axes[1].set_title("Within-session reliability")

    fig.suptitle(f"{label_str} ({'perceived' if perceived else 'all trials'})")
    fig.tight_layout()
    fname = out_dir / f"ratings_corr_{label_str}_{'perceived' if perceived else 'all'}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 3: laser quality category breakdown
# ---------------------------------------------------------------------------

def plot_laser_quality(root, subjects, out_dir):
    counts = {q: 0 for q in QUALITY_LABELS}
    n_subjects_included = 0
    for s in subjects:
        _, qualities = load_ratings(root, s, "laser")
        valid = [q for q in qualities if q in counts]
        if valid:
            n_subjects_included += 1
        for q in valid:
            counts[q] += 1

    total = sum(counts.values())
    pain_total = sum(counts[q] for q in QUALITY_LABELS if q not in NONPAIN_LABELS)
    nonpain_total = sum(counts[q] for q in NONPAIN_LABELS)
    pct_pain = 100 * pain_total / total if total else 0
    pct_nonpain = 100 * nonpain_total / total if total else 0

    fig, ax = plt.subplots(figsize=(7, 5.5))
    labels = [QUALITY_NAMES[q] for q in QUALITY_LABELS]
    values = [counts[q] for q in QUALITY_LABELS]
    bars = ax.bar(labels, values, color="indianred", alpha=0.8)

    for bar, q in zip(bars, QUALITY_LABELS):
        pct = 100 * counts[q] / total if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                 f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("trial count")
    ax.set_title(
        f"Laser quality category distribution (n = {n_subjects_included} subjects)\n"
        f"Pain (sharp/sharp-burning/burning): {pct_pain:.1f}%   |   "
        f"Non-pain (warm/none): {pct_nonpain:.1f}%"
    )
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fname = out_dir / "laser_quality_distribution.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


# ---------------------------------------------------------------------------
# Plot 4: Laser vs pinprick — per-subject and group-level significance
# ---------------------------------------------------------------------------

def plot_laser_vs_pinprick(root, subjects, out_dir, perceived=False):
    from scipy import stats

    subj_results = []  # (subject, mean_laser, mean_pinprick, p_value)
    laser_means, pp_means = [], []

    for s in subjects:
        lr, _ = load_ratings(root, s, "laser")
        pr, _ = load_ratings(root, s, "pinprick")
        l_arr = perceived_only(lr) if perceived else clean(lr)
        p_arr = perceived_only(pr) if perceived else clean(pr)
        if l_arr.size < 2 or p_arr.size < 2:
            continue
        # Per-subject: Mann-Whitney U (trials are independent, not paired by index
        # since trial order/intensity differs between tasks)
        _, pval = stats.mannwhitneyu(l_arr, p_arr, alternative="two-sided")
        subj_results.append((s, l_arr.mean(), p_arr.mean(), pval))
        laser_means.append(l_arr.mean())
        pp_means.append(p_arr.mean())

    # Group-level: paired test on per-subject means (same subjects did both tasks)
    laser_means = np.array(laser_means)
    pp_means = np.array(pp_means)
    if len(laser_means) > 1:
        t_stat, t_p = stats.ttest_rel(laser_means, pp_means)
        w_stat, w_p = stats.wilcoxon(laser_means, pp_means)
    else:
        t_p = w_p = np.nan

    n_sig = sum(1 for _, _, _, p in subj_results if p < 0.05)

    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(len(subj_results))
    width = 0.35
    l_vals = [r[1] for r in subj_results]
    p_vals = [r[2] for r in subj_results]
    sig = [r[3] < 0.05 for r in subj_results]

    ax.bar(x - width / 2, l_vals, width, label="laser", color=TASK_COLORS["laser"], alpha=0.85)
    ax.bar(x + width / 2, p_vals, width, label="pinprick", color=TASK_COLORS["pinprick"], alpha=0.85)
    for i, is_sig in enumerate(sig):
        if is_sig:
            ax.text(i, max(l_vals[i], p_vals[i]) + 1, "*", ha="center", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in subj_results], rotation=90, fontsize=7)
    ax.set_ylabel("mean rating (0\u2013100 NRS)")
    ax.set_title(
        f"Laser vs pinprick per subject ({'perceived' if perceived else 'all trials'})\n"
        f"* = p<0.05 (Mann-Whitney U, within subject)   |   "
        f"{n_sig}/{len(subj_results)} subjects significant\n"
        f"Group-level (paired across subjects): "
        f"paired t-test p={t_p:.4f}, Wilcoxon p={w_p:.4f}"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = out_dir / f"laser_vs_pinprick_{'perceived' if perceived else 'all'}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")




def main():
    parser = argparse.ArgumentParser(description="Plot behavioural rating data")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=None,
                         help="Bare subject labels (default: all in participants.tsv)")
    args = parser.parse_args()

    root = args.root
    if args.subjects:
        subjects = args.subjects
    else:
        tsv = root / "rawdata" / "participants.tsv"
        subjects = []
        with open(tsv) as f:
            next(f)
            for line in f:
                pid = line.split("\t")[0].strip()
                if pid and pid != "sub-P01":
                    subjects.append(pid.replace("sub-", ""))

    out_dir = root / "derivatives" / "logs" / "plots" / "ratings"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Subjects: {subjects}")

    # Nociceptive tasks (laser + pinprick) — same 0-100 NRS scale.
    # Labelled "nociception" rather than "pain" since non-painful
    # ratings (0, "warm") are included too.
    for perceived in (False, True):
        plot_bar_group(root, subjects, PAIN_TASKS, out_dir, "nociception", perceived)
        plot_correlation(root, subjects, PAIN_TASKS, out_dir, "nociception", perceived)

    # Tactile — separate scale, kept apart
    for perceived in (False, True):
        plot_bar_group(root, subjects, SENSATION_TASKS, out_dir, "tactile", perceived)
        plot_correlation(root, subjects, SENSATION_TASKS, out_dir, "tactile", perceived)

    # Laser vs pinprick: per-subject + group-level significance
    for perceived in (False, True):
        plot_laser_vs_pinprick(root, subjects, out_dir, perceived)

    # Laser quality breakdown
    plot_laser_quality(root, subjects, out_dir)

    print(f"\nAll plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
