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
    python code/setup_parcellation.py --root $MEGROOT --subjects 4382
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
from connectivity_common import (
    LAMBDA2,
    SNR,
    extract_roi_time_courses,
    load_roi_labels,
    load_trial_mask,
    _bandpass,
    _exists,
    _get_roi_pairs,
)

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")

FREQ_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
}


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
        "--trials", default="all", choices=["all", "perceived", "not-perceived"],
        help=(
            "'all' = all epochs (default). "
            "'perceived' = only trials rated > 0 and not 'miss'. "
            "'not-perceived' = only trials rated exactly 0 and not 'miss'. "
            "Requires match_ratings.py to have been run first. "
            "Output file gets a matching suffix so all versions coexist."
        ),
    )
    args = parser.parse_args()

    paths    = Paths(args.root)
    logger   = setup_logging(paths, "wpli")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks    = args.tasks    if args.tasks    else TASKS

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
