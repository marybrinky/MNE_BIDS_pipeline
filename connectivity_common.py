#!/usr/bin/env python3
"""
connectivity_common.py
-----------------------
Shared infrastructure for all connectivity scripts (wpli.py, pac.py,
psi.py). None of these three scripts depend on each other — they all
import their shared building blocks from here, sitting at the same
level.

Contains
--------
- Trial-selection: load_trial_mask() reads ratings the same way as
  epoch.py (triggercheck JSON first, behavioural mat file fallback).
- ROI setup: setup_parcellation(), load_roi_labels(), _get_roi_pairs().
- ROI time-course extraction: extract_roi_time_courses() — applies the
  dSPM inverse to single-trial epochs and extracts mean-flip ROI time
  courses, used as the common input to WPLI / PAC / PSI.
- Small utilities: _exists(), _bandpass().

Usage
-----
This module is not run directly. wpli.py / pac.py / psi.py import from
it, e.g.:

    from connectivity_common import (
        load_trial_mask, load_roi_labels, extract_roi_time_courses,
        setup_parcellation, _get_roi_pairs, _exists, _bandpass,
        SNR, LAMBDA2, FILTER_ORDER,
    )
"""

import json as _json
import re as _re
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, filtfilt

from core import (
    ATLAS_CONFIGS,
    DEFAULT_ATLAS,
    EPOCH_CONFIGS,
    Paths,
    sub_id,
)

SNR          = 3.0
LAMBDA2      = 1.0 / SNR**2
FILTER_ORDER = 4


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _bandpass(signal: np.ndarray, lo: float, hi: float, sfreq: float) -> np.ndarray:
    nyq = sfreq / 2.0
    b, a = butter(FILTER_ORDER, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# ROI pair definitions
# ---------------------------------------------------------------------------


def _get_roi_definitions(atlas_key: str = DEFAULT_ATLAS) -> dict[str, list[str]]:
    return ATLAS_CONFIGS[atlas_key]["rois"]


def _get_roi_pairs(roi_names: list[str]) -> list[tuple[str, str]]:
    """Predefined bilateral pain-matrix pairs, or all combinations."""
    coarse_pairs = [
        ("SI_l",     "SII_r"),
        ("SI_l",     "SII_l"),
        ("SI_l",     "Insula_r"),
        ("SI_l",     "Insula_l"),
        ("SII_r",    "SII_l"),
        ("SII_r",    "Insula_r"),
        ("SII_r",    "Insula_l"),
        ("SII_l",    "Insula_r"),
        ("SII_l",    "Insula_l"),
        ("Insula_r", "Insula_l"),
    ]
    coarse_rois = {"SI_r", "SI_l", "SII_r", "SII_l", "Insula_r", "Insula_l", "ACC"}
    if set(roi_names).issubset(coarse_rois):
        return [(a, b) for a, b in coarse_pairs if a in roi_names and b in roi_names]
    pairs = []
    for i, a in enumerate(roi_names):
        for b in roi_names[i + 1:]:
            pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Trial selection — reads ratings the same way as epoch.py
# ---------------------------------------------------------------------------


def _unwrap_cell(cell):
    """Recursively unwrap nested arrays/lists down to a scalar string."""
    while hasattr(cell, "__len__") and not isinstance(cell, str) and len(cell) > 0:
        try:
            cell = cell.flat[0] if hasattr(cell, "flat") else cell[0]
        except (IndexError, AttributeError):
            break
    return _re.sub(r"""[\[\]'"]""", "", str(cell).strip()).strip()


def _read_ratings_from_json(json_path: Path, task: str) -> list:
    with json_path.open() as f:
        tc = _json.load(f)
    stim_key = "is_laser" if task == "laser" else "is_stim"
    ratings = []
    for t in tc.get("trials", []):
        if not t.get(stim_key, False):
            continue
        val = t.get("intensity_mat", t.get("intensity_fif"))
        ratings.append(None if (val is None or val == -1) else float(val))
    return ratings


def _read_ratings_from_mat(mat_path: Path, task: str = "pinprick") -> list:
    """row 0 = intensity for all tasks; row 1 = quality letter for laser only."""
    import scipy.io
    mat = scipy.io.loadmat(str(mat_path))
    r = mat["response"][0, 0]
    resps = r["responses"]
    n = resps.shape[1]
    ratings = []
    for i in range(n):
        val = _unwrap_cell(resps[0, i])
        if "miss" in val.lower() or val in ("", "nan"):
            ratings.append(None)
        else:
            try:
                ratings.append(float(val))
            except ValueError:
                ratings.append(None)
    return ratings


def load_trial_mask(
    paths: Paths,
    label: str,
    task: str,
    trial_filter: str,
    n_epochs: int,
    logger,
) -> np.ndarray | None:
    """Boolean mask of trials to keep, read directly from the same rating
    sources as epoch.py — triggercheck JSON first (corrected indexing for
    compound-trigger subjects), behavioural mat file as fallback.

    Parameters
    ----------
    trial_filter : str
        "all"           — keep all trials (returns None = no filtering)
        "perceived"     — keep only trials with intensity > 0
        "not-perceived" — keep only trials with intensity == 0

    Returns None when no filtering is needed or no rating source is found.
    """
    tag = f"sub-{label} / {task}"

    if trial_filter == "all":
        return None

    json_path = (
        paths.deriv / "trigger_check" / sub_id(label)
        / f"{sub_id(label)}_task-{task}_triggercheck.json"
    )
    ratings = None
    if json_path.exists():
        try:
            ratings = _read_ratings_from_json(json_path, task)
            logger.info("[%s]  Ratings from triggercheck JSON (%d trials)",
                        tag, len(ratings))
        except Exception as e:
            logger.warning("[%s]  JSON rating read failed: %s — trying mat", tag, e)
            ratings = None

    if ratings is None:
        mat_path = (
            paths.raw / sub_id(label) / "beh"
            / f"{sub_id(label)}_task-{task}_ratings.mat"
        )
        if mat_path.exists():
            try:
                ratings = _read_ratings_from_mat(mat_path, task)
                logger.info("[%s]  Ratings from mat file (%d trials)",
                            tag, len(ratings))
            except Exception as e:
                logger.warning("[%s]  Mat rating read failed: %s — using all trials",
                                tag, e)
                return None
        else:
            logger.warning("[%s]  No rating source found — using all trials", tag)
            return None

    if len(ratings) < n_epochs:
        logger.warning(
            "[%s]  Rating count (%d) is less than required index range (%d) "
            "— skipping filter.", tag, len(ratings), n_epochs
        )
        return None

    mask   = np.zeros(n_epochs, dtype=bool)
    n_kept = 0
    for i, val in enumerate(ratings):
        if val is None:
            continue
        if trial_filter == "perceived" and val > 0:
            mask[i] = True
            n_kept  += 1
        elif trial_filter == "not-perceived" and val == 0:
            mask[i] = True
            n_kept  += 1

    logger.info("[%s]  Trial filter '%s': %d / %d trials kept",
                tag, trial_filter, n_kept, n_epochs)
    return mask


# ---------------------------------------------------------------------------
# ROI setup
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
# ROI time-course extraction (the common input to WPLI / PAC / PSI)
# ---------------------------------------------------------------------------


def extract_roi_time_courses(
    paths: Paths,
    label: str,
    task: str,
    epoch_config: str,
    inv: "mne.minimum_norm.InverseOperator",
    roi_label_map: dict[str, list],
    logger,
    trial_filter: str = "all",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]] | None:
    """Apply inverse to single trials -> mean-flip ROI time courses.

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

    # Apply trial filter from ratings (JSON/mat). Ratings are indexed by the
    # ORIGINAL trial number (1..n_total), while `epochs` here has already
    # had amplitude-rejected trials removed. epochs.selection records which
    # original indices survived, so we build the mask on the full original
    # count and then subselect via epochs.selection — same approach as
    # epoch.py's perceived filter.
    n_total_for_ratings = int(epochs.selection.max()) + 1 if len(epochs) else len(epochs)
    full_mask = load_trial_mask(paths, label, task, trial_filter,
                                 n_total_for_ratings, logger)
    if full_mask is not None:
        trial_mask = np.array([full_mask[i] for i in epochs.selection], dtype=bool)
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
