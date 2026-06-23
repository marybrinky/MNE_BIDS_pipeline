#!/usr/bin/env python3
"""
wpli.py
-------
Weighted Phase Lag Index (WPLI) connectivity between pain matrix regions
for the laser-pain MEG study.  Compute only — see plot_wpli.py for plots.

Pipeline
--------
1. Load single-trial epochs
2. Apply dSPM inverse to each trial -> single-trial STCs
3. Extract ROI time courses (HCPMMP1, bilateral hemisphere-specific ROIs)
4. Bandpass filter to each frequency band
5. Compute WPLI between all ROI pairs using Hilbert-based estimator
6. Save per-subject/task/condition/pair/band results to HDF5

WPLI estimator
--------------
    WPLI = |E[Im(C_xy)]| / E[|Im(C_xy)|]

where C_xy = conj(X) * Y is the cross-spectrum across trials using
the analytic (Hilbert) signal. This is the debiased formulation
(Vinck et al. 2011), appropriate for event-related designs.

No surrogate statistics are computed. WPLI is bounded [0, 1] and does
not follow a Gaussian distribution, so z-scores are not appropriate.
Use permutation tests at the group level if statistical inference is needed.

HDF5 layout
-----------
    /{pair}/{band}/
        attrs: {cond} = raw WPLI
        /{cond}/
            attrs: wpli
    /roi_time_courses/
        data, times, roi_names, conditions

Usage
-----
    python code/wpli.py --root $MEGROOT --subjects 4382
    python code/wpli.py --root $MEGROOT --bands theta
    python code/wpli.py --root $MEGROOT --overwrite
    python code/wpli.py --root $MEGROOT --setup-parcellation --subjects 4382
"""

import argparse
import sys
from pathlib import Path

import h5py
import mne
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from core import (
    ATLAS_CONFIGS,
    DEFAULT_ATLAS,
    DEFAULT_EPOCH_CONFIG,
    EPOCH_CONFIGS,
    TASKS,
    Paths,
    load_subjects,
    setup_logging,
    sub_id,
)

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")


# ---------------------------------------------------------------------------
# Trial selection via ratings TSV
# ---------------------------------------------------------------------------


def load_trial_mask(
    paths: Paths,
    label: str,
    task: str,
    trial_filter: str,
    n_epochs: int,
    logger,
) -> np.ndarray | None:
    """Boolean mask of trials to keep based on ratings TSV.

    Parameters
    ----------
    trial_filter : str
        "all"       — keep all trials (default, returns None = no filtering)
        "perceived" — keep only trials with intensity > 0 and not "miss"

    Returns None when no filtering is needed or ratings file is missing.
    """
    import csv
    tag = f"sub-{label} / {task}"

    if trial_filter == "all":
        return None

    tsv = (
        paths.deriv / "ratings" / sub_id(label)
        / f"{sub_id(label)}_task-{task}_ratings.tsv"
    )
    if not tsv.exists():
        logger.warning(
            "[%s]  Ratings TSV not found — using all trials. "
            "Run match_ratings.py first to enable --trials perceived.", tag
        )
        return None

    with open(tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows   = list(reader)

    if len(rows) != n_epochs:
        logger.warning(
            "[%s]  Ratings TSV has %d rows but epochs has %d trials — "
            "skipping filter.", tag, len(rows), n_epochs
        )
        return None

    mask   = np.zeros(n_epochs, dtype=bool)
    n_kept = 0
    for i, row in enumerate(rows):
        intensity = str(row.get("intensity", "")).strip().lower()
        matched   = str(row.get("matched",   "")).strip().lower()
        if matched != "true":
            continue
        if intensity in ("", "nan", "miss", "none"):
            continue
        try:
            if float(intensity) > 0:
                mask[i] = True
                n_kept  += 1
        except ValueError:
            continue

    logger.info("[%s]  Trial filter '%s': %d / %d trials kept",
                tag, trial_filter, n_kept, n_epochs)
    return mask

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SNR          = 3.0
LAMBDA2      = 1.0 / SNR**2
FILTER_ORDER = 4

FREQ_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
}


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------


def _get_roi_definitions(atlas_key: str = DEFAULT_ATLAS) -> dict[str, list[str]]:
    return ATLAS_CONFIGS[atlas_key]["rois"]


def _get_roi_pairs(roi_names: list[str]) -> list[tuple[str, str]]:
    """Predefined bilateral pain-matrix pairs, or all combinations."""
    coarse_pairs = [
        ("SI_l",    "SII_r"),
        ("SI_l",    "SII_l"),
        ("SI_l",    "Insula_r"),
        ("SI_l",    "Insula_l"),
        ("SII_r",   "SII_l"),
        ("SII_r",   "Insula_r"),
        ("SII_r",   "Insula_l"),
        ("SII_l",   "Insula_r"),
        ("SII_l",   "Insula_l"),
        ("Insula_r","Insula_l"),
    ]
    coarse_rois = {"SI_r","SI_l","SII_r","SII_l","Insula_r","Insula_l","ACC"}
    if set(roi_names).issubset(coarse_rois):
        return [(a,b) for a,b in coarse_pairs if a in roi_names and b in roi_names]
    pairs = []
    for i, a in enumerate(roi_names):
        for b in roi_names[i+1:]:
            pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _bandpass(signal: np.ndarray, lo: float, hi: float, sfreq: float) -> np.ndarray:
    nyq = sfreq / 2.0
    b, a = butter(FILTER_ORDER, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# Step 1: Load ROI labels
# ---------------------------------------------------------------------------


def load_roi_labels(
    paths: Paths, label: str, logger, atlas_key: str = DEFAULT_ATLAS
) -> dict[str, list]:
    tag       = f"sub-{label}"
    atlas_cfg = ATLAS_CONFIGS[atlas_key]
    parc      = atlas_cfg["parc"]
    roi_defs  = atlas_cfg["rois"]

    all_labels = mne.read_labels_from_annot(
        sub_id(label), parc=parc,
        subjects_dir=str(paths.freesurfer_dir()), verbose=False,
    )
    logger.info("[%s]  Loaded %d labels from %s", tag, len(all_labels), parc)

    base_to_label: dict[str, mne.Label] = {}
    for lbl in all_labels:
        base = lbl.name.removesuffix("-lh").removesuffix("-rh")
        base_to_label[base] = lbl

    roi_label_map: dict[str, list] = {}
    for roi_name, label_names in roi_defs.items():
        matched = []
        for ln in label_names:
            base = ln.removesuffix("-lh").removesuffix("-rh")
            if base in base_to_label:
                matched.append(base_to_label[base])
            else:
                logger.warning("[%s]  Label not found in %s: %s", tag, parc, ln)
        if matched:
            roi_label_map[roi_name] = matched
            hemi_counts = {
                "lh": sum(1 for l in matched if l.name.endswith("-lh")),
                "rh": sum(1 for l in matched if l.name.endswith("-rh")),
            }
            logger.info("[%s]  ROI '%s': %d labels %s", tag, roi_name, len(matched), hemi_counts)
        else:
            logger.warning("[%s]  ROI '%s': NO labels matched", tag, roi_name)
    return roi_label_map


# ---------------------------------------------------------------------------
# Step 2: Extract ROI time courses
# ---------------------------------------------------------------------------


def extract_roi_time_courses(
    paths: Paths,
    label: str,
    task: str,
    epoch_config: str,
    inv: mne.minimum_norm.InverseOperator,
    roi_label_map: dict[str, list],
    logger,
    trial_filter: str = "all",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]] | None:
    """Apply inverse to single trials → mean-flip ROI time courses.

    Returns (data, times, roi_names, conditions) or None on failure.
    data shape: (n_rois, n_epochs, n_times)
    """
    tag      = f"sub-{label} / {task}"
    cfg      = EPOCH_CONFIGS[epoch_config]
    epo_file = paths.epochs(label, task, desc=f"{cfg['desc']}-preproc")

    if not _exists(epo_file):
        logger.warning("[%s]  Epochs not found: %s", tag, epo_file.name)
        return None
    src_file = paths.src(label)
    if not _exists(src_file):
        logger.warning("[%s]  Source space not found", tag)
        return None

    logger.info("[%s]  Loading epochs ...", tag)
    epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)
    src    = mne.read_source_spaces(str(src_file), verbose=False)

    common_chs = [ch for ch in inv["info"]["ch_names"] if ch in set(epochs.ch_names)]
    epochs     = epochs.pick(common_chs, verbose=False)
    logger.info("[%s]  Channels after alignment: %d", tag, len(common_chs))

    # Apply trial filter from ratings TSV
    trial_mask = load_trial_mask(paths, label, task, trial_filter, len(epochs), logger)
    if trial_mask is not None:
        epochs = epochs[trial_mask]
        if len(epochs) < 5:
            logger.warning("[%s]  Too few epochs after filtering (%d) — skipping",
                           tag, len(epochs))
            return None

    logger.info("[%s]  Applying inverse to %d single trials ...", tag, len(epochs))
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs, inv, lambda2=LAMBDA2, method="dSPM",
        pick_ori="normal", verbose=False,
    )

    stc_subject = stcs[0].subject
    for roi_labels in roi_label_map.values():
        for lbl in roi_labels:
            lbl.subject = stc_subject

    roi_names = list(roi_label_map.keys())
    times     = stcs[0].times
    data      = np.zeros((len(roi_names), len(stcs), len(times)), dtype=np.float32)

    for ei, stc in enumerate(stcs):
        for ri, roi_name in enumerate(roi_names):
            tc = mne.extract_label_time_course(
                stc, roi_label_map[roi_name], src, mode="mean_flip", verbose=False,
            )
            data[ri, ei, :] = tc.mean(axis=0)

    id_to_name = {v: k for k, v in epochs.event_id.items()}
    conditions = [id_to_name.get(epochs.events[i, 2], "unknown") for i in range(len(stcs))]

    logger.info("[%s]  ROI time courses: shape %s", tag, data.shape)
    return data, times, roi_names, conditions


# ---------------------------------------------------------------------------
# Step 3: WPLI (raw, no surrogates)
# ---------------------------------------------------------------------------


def _compute_wpli_from_analytic(analytic_a: np.ndarray, analytic_b: np.ndarray) -> float:
    im_cross = np.imag(np.conj(analytic_a) * analytic_b)
    num = np.abs(np.mean(im_cross, axis=0)).mean()
    den = np.mean(np.abs(im_cross), axis=0).mean()
    return float(num / den) if den > 1e-12 else 0.0


def compute_wpli(
    tc_a: np.ndarray, tc_b: np.ndarray, sfreq: float, band: tuple[float, float]
) -> float:
    """Raw WPLI (Vinck et al. 2011). No surrogates."""
    lo, hi = band
    n = tc_a.shape[0]
    aa = np.array([hilbert(_bandpass(tc_a[i], lo, hi, sfreq)) for i in range(n)])
    ab = np.array([hilbert(_bandpass(tc_b[i], lo, hi, sfreq)) for i in range(n)])
    return _compute_wpli_from_analytic(aa, ab)


# ---------------------------------------------------------------------------
# Step 0: Parcellation setup
# ---------------------------------------------------------------------------


def setup_parcellation(
    paths: Paths, label: str, atlas_key: str = DEFAULT_ATLAS,
    overwrite: bool = False, logger=None,
) -> bool:
    tag          = f"sub-{label}"
    atlas_cfg    = ATLAS_CONFIGS[atlas_key]
    parc         = atlas_cfg["parc"]
    subjects_dir = str(paths.freesurfer_dir())
    subject      = sub_id(label)

    lh = paths.freesurfer_dir() / subject / "label" / f"lh.{parc}.annot"
    rh = paths.freesurfer_dir() / subject / "label" / f"rh.{parc}.annot"
    if lh.exists() and rh.exists() and not overwrite:
        logger.info("[%s]  %s already exists — skipping", tag, parc)
        return True

    logger.info("[%s]  Setting up %s ...", tag, parc)
    try:
        mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=False)
        fsavg = mne.read_labels_from_annot("fsaverage", parc=parc, subjects_dir=subjects_dir, verbose=False)
        morphed = mne.morph_labels(fsavg, subject_to=subject, subject_from="fsaverage", subjects_dir=subjects_dir)
        mne.write_labels_to_annot(morphed, subject=subject, parc=parc, subjects_dir=subjects_dir, overwrite=True, verbose=False)
        logger.info("[%s]  Written: lh.%s.annot, rh.%s.annot", tag, parc, parc)
        return True
    except Exception as e:
        logger.error("[%s]  Parcellation failed: %s", tag, e, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Step 4: Per-subject/task orchestration
# ---------------------------------------------------------------------------


def wpli_one(
    paths: Paths, label: str, task: str, epoch_config: str,
    bands: list[str], atlas_key: str = DEFAULT_ATLAS,
    overwrite: bool = False, trial_filter: str = "all", logger=None,
) -> bool:
    tag      = f"sub-{label} / {task}"
    suffix   = "" if trial_filter == "all" else f"_{trial_filter}"
    out_dir  = paths.deriv / "connectivity" / sub_id(label) / f"task-{task}"
    out_file = out_dir / f"{sub_id(label)}_task-{task}_wpli_painmatrix{suffix}.h5"

    if _exists(out_file) and not overwrite:
        logger.info("[%s]  SKIP — WPLI file exists", tag)
        return True

    inv_file = paths.inv(label, task)
    if not _exists(inv_file):
        logger.warning("[%s]  Inverse operator not found: %s", tag, inv_file)
        return False

    inv           = mne.minimum_norm.read_inverse_operator(str(inv_file), verbose=False)
    roi_label_map = load_roi_labels(paths, label, logger, atlas_key=atlas_key)
    if len(roi_label_map) < 2:
        logger.warning("[%s]  Too few ROIs — skipping", tag)
        return False

    result = extract_roi_time_courses(paths, label, task, epoch_config, inv, roi_label_map, logger, trial_filter=trial_filter)
    if result is None:
        return False

    data, times, roi_names, conditions = result
    sfreq             = round(1.0 / float(times[1] - times[0]))
    unique_conditions = sorted(set(conditions))
    roi_pairs         = _get_roi_pairs(list(roi_label_map.keys()))

    logger.info("[%s]  WPLI: %d pairs x %d bands x %d conditions",
                tag, len(roi_pairs), len(bands), len(unique_conditions))

    wpli_results: dict[str, dict[str, dict[str, float]]] = {}
    for roi_a, roi_b in roi_pairs:
        if roi_a not in roi_names or roi_b not in roi_names:
            logger.warning("[%s]  Pair %s-%s: ROI missing", tag, roi_a, roi_b)
            continue
        idx_a     = roi_names.index(roi_a)
        idx_b     = roi_names.index(roi_b)
        pair_name = f"{roi_a}-{roi_b}"
        wpli_results[pair_name] = {}

        for band_name in bands:
            band = FREQ_BANDS[band_name]
            wpli_results[pair_name][band_name] = {}
            for cond in unique_conditions:
                mask = np.array([c == cond for c in conditions])
                tc_a = data[idx_a, mask, :].astype(np.float64)
                tc_b = data[idx_b, mask, :].astype(np.float64)
                if tc_a.shape[0] < 5:
                    logger.warning("[%s]  %s/%s/%s: %d trials — skip",
                                   tag, pair_name, band_name, cond, tc_a.shape[0])
                    wpli_results[pair_name][band_name][cond] = np.nan
                    continue
                wpli_val = compute_wpli(tc_a, tc_b, sfreq, band)
                wpli_results[pair_name][band_name][cond] = wpli_val
                logger.info("[%s]  %s  %s  %-20s  WPLI=%.4f  (n=%d)",
                            tag, pair_name, band_name, cond, wpli_val, tc_a.shape[0])

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as f:
        f.attrs.update(dict(subject=sub_id(label), task=task, epoch_config=epoch_config,
                            atlas=atlas_key, method="WPLI",
                            reference="Vinck et al. 2011 NeuroImage",
                            inverse="dSPM", sfreq=sfreq))
        for pair_name, band_dict in wpli_results.items():
            grp = f.create_group(pair_name)
            for band_name, cond_dict in band_dict.items():
                bgrp = grp.create_group(band_name)
                for cond, wpli_val in cond_dict.items():
                    safe_val = float(wpli_val) if not np.isnan(wpli_val) else -1.0
                    bgrp.attrs[cond] = safe_val          # flat attr for backward compat
                    sgrp = bgrp.create_group(cond)
                    sgrp.attrs["wpli"] = float(wpli_val) if not np.isnan(wpli_val) else np.nan
        tc = f.create_group("roi_time_courses")
        tc.create_dataset("data",       data=data,                         compression="gzip", compression_opts=4)
        tc.create_dataset("times",      data=times)
        tc.create_dataset("roi_names",  data=np.array(roi_names,  dtype="S"))
        tc.create_dataset("conditions", data=np.array(conditions, dtype="S"))

    logger.info("[%s]  Saved: %s", tag, out_file.name)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="WPLI connectivity — laser-pain MEG study (compute only).\n"
                    "For plots run: python code/plot_wpli.py"
    )
    parser.add_argument("--root",         type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects",     nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--tasks",        nargs="+", default=None, choices=TASKS)
    parser.add_argument("--epoch-config", default=DEFAULT_EPOCH_CONFIG, choices=list(EPOCH_CONFIGS.keys()))
    parser.add_argument("--bands",        nargs="+", default=list(FREQ_BANDS.keys()), choices=list(FREQ_BANDS.keys()))
    parser.add_argument("--atlas",        default=DEFAULT_ATLAS, choices=list(ATLAS_CONFIGS.keys()))
    parser.add_argument("--overwrite",    action="store_true")
    parser.add_argument(
        "--trials", default="all", choices=["all", "perceived"],
        help=(
            "'all' = all epochs (default). "
            "'perceived' = only trials rated > 0 and not 'miss'. "
            "Requires match_ratings.py to have been run first. "
            "Output file gets suffix _perceived so both versions coexist."
        ),
    )
    parser.add_argument("--setup-parcellation", action="store_true",
                        help="Morph HCPMMP1 from fsaverage to each subject (run once).")
    args = parser.parse_args()

    paths    = Paths(args.root)
    logger   = setup_logging(paths, "wpli")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks    = args.tasks    if args.tasks    else TASKS

    if args.setup_parcellation:
        logger.info("-- Parcellation setup --  Atlas: %s  Subjects: %s", args.atlas, subjects)
        n_ok = n_fail = 0
        for label in subjects:
            ok = setup_parcellation(paths, label, atlas_key=args.atlas, overwrite=args.overwrite, logger=logger)
            if ok: n_ok += 1
            else:  n_fail += 1
        logger.info("Done.  OK: %d  |  Failed: %d", n_ok, n_fail)
        if n_fail: sys.exit(1)
        return

    logger.info("Subjects     : %s", subjects)
    logger.info("Tasks        : %s", tasks)
    logger.info("Epoch config : %s", args.epoch_config)
    logger.info("Bands        : %s", args.bands)
    logger.info("Atlas        : %s", args.atlas)
    logger.info("Overwrite    : %s", args.overwrite)
    logger.info("Trials       : %s", args.trials)

    n_ok = n_fail = 0
    for label in subjects:
        for task in tasks:
            try:
                ok = wpli_one(paths, label, task, epoch_config=args.epoch_config,
                              bands=args.bands, atlas_key=args.atlas,
                              overwrite=args.overwrite, trial_filter=args.trials, logger=logger)
                if ok: n_ok += 1
                else:  n_fail += 1
            except Exception as e:
                logger.error("[sub-%s / %s]  FAILED: %s", label, task, e, exc_info=True)
                n_fail += 1

    logger.info("─────────────────────────────────────────────")
    logger.info("Done.  Success: %d  |  Failed: %d", n_ok, n_fail)
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
