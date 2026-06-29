#!/usr/bin/env python3
"""
psi.py
------
Phase Slope Index (PSI) connectivity between pain-matrix ROIs.

Reuses the exact same single-trial source-localisation pipeline as
wpli.py (load epochs -> apply dSPM inverse -> extract ROI time courses),
then computes PSI on those ROI time courses instead of WPLI.

PSI estimates the DIRECTION of information flow between two signals by
looking at how their phase difference changes across neighbouring
frequencies (Nolte et al., 2008). Unlike WPLI, which only measures
connectivity strength, PSI's sign tells you which ROI tends to LEAD the
other within a frequency band.

    PSI_ij > 0  ->  ROI i leads ROI j  (flow i -> j)
    PSI_ij < 0  ->  ROI j leads ROI i  (flow j -> i)

Sources
-------
- Nolte G, Ziehe A, Nikulin VV, Schlögl A, Krämer N, Brismar T, Müller KR.
  "Robustly estimating the flow direction of information in complex
  physical systems." Physical Review Letters, 100(23), 234101 (2008).
  Defines PSI and its estimator from the cross-spectral density, and
  recommends fitting the slope across a frequency BAND rather than a
  single frequency bin (a single frequency cannot constrain a slope).

- MNE-Connectivity documentation for phase_slope_index:
  https://mne.tools/mne-connectivity/stable/generated/
  mne_connectivity.phase_slope_index.html
  Implementation used directly here (mode="multitaper", fmin/fmax).

- This project's wpli.py: ROI extraction, inverse-operator loading, and
  HDF5 output conventions are reused unchanged so PSI results sit
  alongside WPLI results in the same per-subject connectivity folder.

Usage
-----
    python code/psi.py --root $MEGROOT --subjects 4382
    python code/psi.py --root $MEGROOT --bands theta alpha
"""

import argparse
import sys
from pathlib import Path

import h5py
import mne
import numpy as np

try:
    from mne_connectivity import phase_slope_index
except ImportError:
    sys.exit(
        "mne-connectivity is required for PSI.\n"
        "Install with: pip install mne-connectivity"
    )

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

# wpli.py imports these from itself; reused here for psi.py's own use
from connectivity_common import load_roi_labels, extract_roi_time_courses, _exists, _get_roi_pairs

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")

# Same bands as wpli.py by default -- PSI needs a band (not a single
# frequency) to fit a phase slope, per Nolte et al. (2008).
FREQ_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
}


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------


def compute_psi_pair(
    tc_a: np.ndarray, tc_b: np.ndarray, sfreq: float, band: tuple[float, float]
) -> float:
    """PSI for one ROI pair across trials, in one frequency band.

    tc_a, tc_b : (n_trials, n_times) ROI time courses
    Returns a single scalar PSI value (sign = direction, a->b positive).
    """
    lo, hi = band
    # mne_connectivity.phase_slope_index expects data shaped
    # (n_epochs, n_signals, n_times); stack the two ROIs as 2 "channels"
    data = np.stack([tc_a, tc_b], axis=1)  # (n_trials, 2, n_times)

    psi_con = phase_slope_index(
        data,
        sfreq=sfreq,
        mode="multitaper",
        fmin=lo,
        fmax=hi,
        verbose=False,
    )
    # dense output: (2, 2, n_bands) -- PSI[0,1] = signal 0 -> signal 1
    mat = psi_con.get_data(output="dense")[:, :, 0]
    return float(mat[0, 1])


# ---------------------------------------------------------------------------
# Per-subject pipeline
# ---------------------------------------------------------------------------


def psi_one(
    paths: Paths, label: str, task: str, epoch_config: str,
    bands: list[str], atlas_key: str = DEFAULT_ATLAS,
    overwrite: bool = False, trial_filter: str = "all", logger=None,
) -> bool:
    tag      = f"sub-{label} / {task}"
    suffix   = "" if trial_filter == "all" else f"_{trial_filter}"
    out_dir  = paths.deriv / "connectivity" / sub_id(label) / f"task-{task}"
    out_file = out_dir / f"{sub_id(label)}_task-{task}_psi_painmatrix{suffix}.h5"

    if _exists(out_file) and not overwrite:
        logger.info("[%s]  SKIP — PSI file exists", tag)
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

    result = extract_roi_time_courses(
        paths, label, task, epoch_config, inv, roi_label_map, logger,
        trial_filter=trial_filter,
    )
    if result is None:
        return False
    data, times, roi_names, conditions = result   # data: (n_rois, n_trials, n_times)

    sfreq = 1.0 / (times[1] - times[0])
    roi_pairs = _get_roi_pairs(roi_names)
    name_to_idx = {n: i for i, n in enumerate(roi_names)}

    logger.info(
        "[%s]  Computing PSI: %d ROI pairs, %d bands, %d trials",
        tag, len(roi_pairs), len(bands), data.shape[1]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_file, "w") as f:
        for roi_a, roi_b in roi_pairs:
            ia, ib = name_to_idx[roi_a], name_to_idx[roi_b]
            tc_a, tc_b = data[ia], data[ib]   # (n_trials, n_times)
            grp = f.create_group(f"{roi_a}__{roi_b}")
            for band_name in bands:
                band = FREQ_BANDS[band_name]
                psi_val = compute_psi_pair(tc_a, tc_b, sfreq, band)
                band_grp = grp.create_group(band_name)
                band_grp.attrs["psi"] = psi_val
                band_grp.attrs["fmin"] = band[0]
                band_grp.attrs["fmax"] = band[1]
                band_grp.attrs["direction"] = (
                    f"{roi_a}->{roi_b}" if psi_val > 0 else f"{roi_b}->{roi_a}"
                )

        f.attrs["roi_names"] = roi_names
        f.attrs["sfreq"] = sfreq
        f.attrs["n_trials"] = data.shape[1]

    logger.info("[%s]  Saved: %s", tag, out_file.name)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="PSI (Phase Slope Index) connectivity — directional "
                     "complement to wpli.py, laser-pain MEG study."
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
        help="'all' = all epochs. 'perceived' = only trials rated > 0. "
             "'not-perceived' = only trials rated exactly 0.",
    )
    args = parser.parse_args()

    paths    = Paths(args.root)
    logger   = setup_logging(paths, "psi")
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
                ok = psi_one(paths, label, task, epoch_config=args.epoch_config,
                             bands=args.bands, atlas_key=args.atlas,
                             overwrite=args.overwrite, trial_filter=args.trials,
                             logger=logger)
                if ok: n_ok += 1
                else:  n_fail += 1
            except Exception as e:
                logger.error("[sub-%s / %s]  FAILED: %s", label, task, e, exc_info=True)
                n_fail += 1

    logger.info("Done.  OK: %d  |  Failed: %d", n_ok, n_fail)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
