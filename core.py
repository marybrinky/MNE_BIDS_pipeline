#!/usr/bin/env python3
"""
core.py
-------
Shared utilities for the laser-pain MEG pipeline.

Provides:
    - Project-wide constants (tasks, ROIs, atlas configs)
    - FILTER_CONFIGS  — filter band variants
    - EPOCH_CONFIGS   — epoching variants
    - ATLAS_CONFIGS   — parcellation variants
    - Path construction via the Paths helper class
    - Subject list loading from rawdata/participants.tsv
    - Standardised logging setup
    - Bad channel I/O helpers
    - Trigger decoding for the Heidelberg Neuromag/TRIUX 6-channel STI system

All other pipeline scripts import from this module; nothing else should
hard-code paths or subject labels.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import mne
import numpy as np
import pandas as pd

# =============================================================================
# Project settings
# =============================================================================

PROJECT_NAME = "laser_pain"

TASKS = ["laser", "pinprick", "tactile"]

EPOCH_CONFIGS = {
    "sweep": {
        "tmin": -0.5,
        "tmax": 1.5,
        "desc": "sweep",
        "prep_desc": "preproc",
        # triggers: default codes used when no subject-specific override exists.
        # Supported formats (same as RHT):
        #   int          -> exact code match
        #   {"min": N}   -> all codes >= N
        #   {"min": N, "max": M}  -> range
        "triggers": {
            "stimulus": 1,  # fallback default
        },
        "baseline": (
            -0.2,
            -0.1,
        ),  # default, used if no task override, changed by Marie 260427
        "reject": {"grad": 6000e-13},
        "flat": {"grad": 1e-13},
        "n_expected": 50,
        "task_overrides": {
            "laser": {"baseline": (-0.1, 0.0)},
            "pinprick": {"baseline": (-0.2, -0.1)},
            "tactile": {"baseline": (-0.2, -0.1)},
        },
    },
    "sweep-tfr": {
        "tmin": -0.5,
        "tmax": 1.5,
        "desc": "broadband",
        "prep_desc": "broadband-preproc",
        "triggers": {
            "stimulus": 1,
        },
        "baseline": (-0.2, -0.1),
        "reject": {"grad": 6000e-13},
        "flat": {"grad": 1e-13},
        "n_expected": 50,
        "task_overrides": {
            "laser": {"baseline": (-0.1, 0.0)},
            "pinprick": {"baseline": (-0.2, -0.1)},
            "tactile": {"baseline": (-0.2, -0.1)},
        },
    },
}

DEFAULT_EPOCH_CONFIG = "sweep"

# ---------------------------------------------------------------------------
# Subject- and task-specific trigger overrides
# ---------------------------------------------------------------------------
# Add one entry per subject whose trigger codes differ from the defaults above.
# Structure: { label: { task: { condition_name: code_or_spec } } }
#
# Rules:
#   - Only tasks/conditions that differ from EPOCH_CONFIGS need an entry.
#   - Missing tasks fall back to EPOCH_CONFIGS["sweep"]["triggers"].
#   - Condition names must match those in EPOCH_CONFIGS["triggers"] so that
#     downstream scripts (source.py, contrast.py) see a consistent label.
#
# Example entry format:
#   "9999": {
#       "laser":    {"stimulus": 4},
#       "pinprick": {"stimulus": 8},
#       "tactile":  {"stimulus": 8},
#   },
# ---------------------------------------------------------------------------
# Trigger strategy for compound subjects (1409, 2827, 3691, etc.)
# ---------------------------------------------------------------------------
#
# Previously these subjects used a compound trigger (code 99 = code 4
# followed by code 11 within 1.5s). This was unreliable because:
#   - the laser was accidentally plugged into STI channel 3 (bit 2 = code 4)
#   - response buttons shared the same STI lines, causing bit overlap
#   - the laser sometimes appeared as code 3, 5, 6, or 7 instead of 4
#
# New approach (2026-06-01):
#   - Trigger structure verified manually in MATLAB (check_laser_trigger.m)
#   - Results saved as per-subject JSON sidecar files:
#       derivatives/trigger_check/sub-{label}/sub-{label}_task-laser_triggercheck.json
#   - JSON contains laser_bundle_indices: the 1-based bundle numbers
#     (anchored on code-11) that are confirmed real laser stimuli
#   - Pipeline reads JSON and uses only those code-4 times as onsets
#   - Non-laser bundles (bit-overlap false triggers) are skipped
#   - Standard subjects (code 32) are unaffected
#
# Subjects still needing JSON: 2827 3691 3847 4011 4163 4245 4365 4508 4574 4575
# Run check_laser_trigger.py, verify in MATLAB, save JSON.

TRIGGERCHECK_SUBJECTS: set[str] = {
    "1409",
    "2827",
    "3691",
    "3847",
    "4011",
    "4163",
    "4245",
    "4365",
    "4508",
    "4574",
    "4575",
}


def load_triggercheck(paths: "Paths", label: str, task: str = "laser") -> dict | None:
    """Load the trigger check JSON sidecar for a subject/task.

    Returns the parsed dict, or None if the file does not exist.

    File location:
        derivatives/trigger_check/sub-{label}/sub-{label}_task-{task}_triggercheck.json
    """
    fpath = (
        paths.deriv
        / "trigger_check"
        / f"sub-{label}"
        / f"sub-{label}_task-{task}_triggercheck.json"
    )
    if not fpath.exists():
        return None
    with fpath.open(encoding="utf-8") as f:
        return json.load(f)


def get_laser_bundle_indices(paths: "Paths", label: str) -> list[int] | None:
    """Return the 1-based bundle indices confirmed as real laser trials.

    For triggercheck subjects reads from the JSON sidecar.
    Returns None if no JSON exists (caller should warn and fall back).
    Returns None for standard subjects (not in TRIGGERCHECK_SUBJECTS).
    """
    if label not in TRIGGERCHECK_SUBJECTS:
        return None
    tc = load_triggercheck(paths, label, task="laser")
    if tc is None:
        return None
    return tc.get("laser_bundle_indices", None)


SUBJECT_TRIGGERS: dict[str, dict[str, dict[str, int | dict]]] = {
    # Triggercheck subjects: laser stimulus = code 4, filtered by JSON
    "1409": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "2827": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "3691": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "3847": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4011": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4163": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4166": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
    "4245": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4277": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4365": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4382": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4399": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4508": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4574": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4575": {
        "laser": {"stimulus": 4},
        "pinprick": {"stimulus": 8},
        "tactile": {"stimulus": 8},
    },
    "4605": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
    "4654": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
    "4815": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
    "4999": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
    "5001": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
    "5004": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
    "5026": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
    "5031": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
    "5038": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
    "5039": {
        "laser": {"stimulus": 32},
        "pinprick": {"stimulus": 1},
        "tactile": {"stimulus": 1},
    },
}


def get_triggers_for(
    label: str,
    task: str,
    epoch_config: str = "sweep",
) -> dict[str, int | dict]:
    """Return the effective trigger map for a given subject and task.

    Looks up SUBJECT_TRIGGERS[label][task] first; falls back to
    EPOCH_CONFIGS[epoch_config]["triggers"] if no override exists.

    Parameters
    ----------
    label        : str  Bare subject label, e.g. "4382".
    task         : str  BIDS task label, e.g. "laser".
    epoch_config : str  Key in EPOCH_CONFIGS (default "sweep").

    Returns
    -------
    dict  {condition_name: code_or_spec}
    """
    default = EPOCH_CONFIGS[epoch_config]["triggers"]
    subject_overrides = SUBJECT_TRIGGERS.get(label, {})
    raw_triggers = subject_overrides.get(
        task, default
    )  #  raw_triggers and return function added by Marie 260427 for laser trigger fix
    return dict(raw_triggers)


DEFAULT_EPOCH_CONFIG = "sweep"

FILTER_CONFIGS = {
    "preproc": {"l_freq": 1.0, "h_freq": 40.0, "notch": None},  # for evokeds
    "broadband": {
        "l_freq": 1.0,
        "h_freq": 90.0,
        "notch": [50.0, 100.0],
    },  # for tfr and gamma analysis
}

ATLAS_CONFIGS = {
    "hcpmmp1": {
        "parc": "HCPMMP1",
        "rois": {
            "SI_l": ["L_1_ROI", "L_2_ROI", "L_3a_ROI", "L_3b_ROI"],
            "SII_r": ["R_OP1_ROI", "R_OP4_ROI", "R_43_ROI"],
            "SII_l": ["L_OP1_ROI", "L_OP4_ROI", "L_43_ROI"],
            "Insula_r": ["R_Ig_ROI", "R_PoI1_ROI", "R_PoI2_ROI"],
            "Insula_l": ["L_Ig_ROI", "L_PoI1_ROI", "L_PoI2_ROI"],
            "ACC": ["L_24_ROI", "L_p24pr_ROI", "R_24_ROI", "R_p24pr_ROI"],
        },
    },
    "hcpmmp1_fine": {
        "parc": "HCPMMP1",
        "rois": {
            # S1 left only — cutaneous forearm/arm representation
            "SI_l": ["L_3b_ROI", "L_1_ROI", "L_3a_ROI", "L_2_ROI"],
            # SII bilateral
            "SII_r": ["R_OP1_ROI", "R_OP4_ROI", "R_43_ROI"],
            "SII_l": ["L_OP1_ROI", "L_OP4_ROI", "L_43_ROI"],
            # Posterior insula bilateral (nociception, interoception)
            "Ins_post_r": ["R_Ig_ROI", "R_PoI1_ROI", "R_PoI2_ROI", "R_MI_ROI"],
            "Ins_post_l": ["L_Ig_ROI", "L_PoI1_ROI", "L_PoI2_ROI", "L_MI_ROI"],
            # Anterior insula bilateral (affective, salience)
            "Ins_ant_r": ["R_AVI_ROI", "R_AAIC_ROI", "R_AId_ROI"],
            "Ins_ant_l": ["L_AVI_ROI", "L_AAIC_ROI", "L_AId_ROI"],
            # Mid-cingulate bilateral (most pain-specific cingulate region)
            "MCC_r": ["R_24dd_ROI", "R_24dv_ROI"],
            "MCC_l": ["L_24dd_ROI", "L_24dv_ROI"],
            # ACC bilateral (affective pain component)
            "ACC_r": ["R_24_ROI", "R_p24pr_ROI"],
            "ACC_l": ["L_24_ROI", "L_p24pr_ROI"],
        },
    },
    "aparcsub": {
        "parc": "aparc_sub",
        "rois": {
            "SI": ["lh.postcentral"],
            "SII": ["lh.superiortemporal"],
            "Insula": ["lh.insula"],
            "ACC": ["lh.caudalanteriorcingulate", "lh.rostralanteriorcingulate"],
        },
    },
}

DEFAULT_ATLAS = "hcpmmp1_fine"

ROI_NAMES = sorted({roi for cfg in ATLAS_CONFIGS.values() for roi in cfg["rois"]})


def get_roi_hemisphere_labels(atlas_key: str = "hcpmmp1") -> dict[str, str]:
    """Return a display label for each ROI indicating its hemisphere coverage.

    If the ROI name already ends with '_r' or '_l', the name is self-explanatory
    and no hemisphere annotation is added.

    Otherwise, reads parcel names from ATLAS_CONFIGS and determines automatically:
      - All parcels start with 'R_' or 'rh.' → append '(rh)'
      - All parcels start with 'L_' or 'lh.' → append '(lh)'
      - Mixed                                 → append '(bilat.)'

    Returns
    -------
    dict {roi_name: display_label}
        e.g. {"SI_l": "SI_l", "ACC": "ACC (bilat.)", ...}
    """
    rois = ATLAS_CONFIGS[atlas_key]["rois"]
    labels = {}
    for roi_name, parcels in rois.items():
        # If name already encodes hemisphere, use as-is
        lower = roi_name.lower()
        if lower.endswith("_r") or lower.endswith("_l"):
            labels[roi_name] = roi_name
            continue
        # Otherwise determine from parcel names
        has_l = any(p.startswith("L_") or p.startswith("lh.") for p in parcels)
        has_r = any(p.startswith("R_") or p.startswith("rh.") for p in parcels)
        if has_l and has_r:
            hemi = "bilat."
        elif has_r:
            hemi = "rh"
        elif has_l:
            hemi = "lh"
        else:
            hemi = "?"
        labels[roi_name] = f"{roi_name} ({hemi})"
    return labels


# =============================================================================
# Subject ID helpers
# =============================================================================


def sub_id(label: str) -> str:
    return label if label.startswith("sub-") else f"sub-{label}"


def sub_label(bids_id: str) -> str:
    return bids_id.removeprefix("sub-")


# =============================================================================
# Path helpers
# =============================================================================


class Paths:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.raw = self.root / "rawdata"
        self.deriv = self.root / "derivatives"

    def raw_meg(self, label: str, task: str) -> Path:
        return self.raw / sub_id(label) / "meg" / f"{sub_id(label)}_task-{task}_meg.fif"

    def events_tsv(self, label: str, task: str) -> Path:
        return (
            self.raw / sub_id(label) / "meg" / f"{sub_id(label)}_task-{task}_events.tsv"
        )

    def prep_dir(self, label: str, task: str) -> Path:
        return self.deriv / "prep" / sub_id(label) / "meg"

    def prep_raw(self, label: str, task: str, desc: str = "preproc") -> Path:
        return (
            self.prep_dir(label, task)
            / f"{sub_id(label)}_task-{task}_desc-{desc}_meg.fif"
        )

    def ica_file(self, label: str, task: str) -> Path:
        return self.prep_dir(label, task) / f"{sub_id(label)}_task-{task}_ica.fif"

    def bads_file(self, label: str, task: str) -> Path:
        return self.deriv / "bads" / f"{sub_id(label)}_task-{task}_bads.json"

    def epochs_dir(self, label: str, task: str) -> Path:
        return self.deriv / "epochs" / sub_id(label) / "meg"

    def epochs(self, label: str, task: str, desc: str = "preproc") -> Path:
        return (
            self.epochs_dir(label, task)
            / f"{sub_id(label)}_task-{task}_desc-{desc}_epo.fif"
        )

    def source_dir(self, label: str) -> Path:
        return self.deriv / "source" / sub_id(label)

    def bem_dir(self, label: str) -> Path:
        return self.source_dir(label) / "bem"

    def bem_sol(self, label: str) -> Path:
        return self.bem_dir(label) / f"{sub_id(label)}-20480-bem-sol.fif"

    def src(self, label: str) -> Path:
        return self.source_dir(label) / f"{sub_id(label)}-src.fif"

    def trans(self, label: str, task: str) -> Path:
        return self.deriv / "trans" / f"{sub_id(label)}_task-{task}-trans.fif"

    def stc_dir(self, label: str, task: str) -> Path:
        return self.source_dir(label) / f"task-{task}" / "meg" / "stc"

    def fwd(self, label: str, task: str) -> Path:
        return (
            self.source_dir(label)
            / f"task-{task}"
            / "meg"
            / f"{sub_id(label)}_task-{task}_fwd.fif"
        )

    def noise_cov(self, label: str, task: str) -> Path:
        return (
            self.source_dir(label)
            / f"task-{task}"
            / "meg"
            / f"{sub_id(label)}_task-{task}_cov.fif"
        )

    def inv(self, label: str, task: str) -> Path:
        return (
            self.source_dir(label)
            / f"task-{task}"
            / "meg"
            / f"{sub_id(label)}_task-{task}_inv.fif"
        )

    def roi_dir(self, label: str, task: str) -> Path:
        return self.source_dir(label) / f"task-{task}" / "meg" / "roi"

    def roi_ave(
        self, label: str, task: str, roi: str, atlas: str, epoch_config: str
    ) -> Path:
        return (
            self.roi_dir(label, task)
            / f"{sub_id(label)}_task-{task}_{roi}_{atlas}_{epoch_config}_ave.h5"
        )

    def roi_epo(self, label: str, task: str, atlas: str, epoch_config: str) -> Path:
        return (
            self.roi_dir(label, task)
            / f"{sub_id(label)}_task-{task}_{atlas}_{epoch_config}_epo.h5"
        )

    def connectivity_dir(self, label: str, task: str) -> Path:
        return self.deriv / "connectivity" / sub_id(label) / f"task-{task}"

    def log_dir(self) -> Path:
        return self.deriv / "logs"

    def freesurfer_dir(self) -> Path:
        return self.deriv / "freesurfer"


# =============================================================================
# Subject list
# =============================================================================


def load_subjects(paths: Paths) -> list[str]:
    tsv = paths.raw / "participants.tsv"
    if not tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found: {tsv}")
    df = pd.read_csv(tsv, sep="\t", dtype=str)
    if "participant_id" not in df.columns:
        raise ValueError(f"No participant_id column in {tsv}")
    return [sub_label(pid) for pid in df["participant_id"].dropna()]


# =============================================================================
# Bad channel I/O
# =============================================================================


def load_bads(paths: Paths, label: str, task: str) -> list[str]:
    """Load bad channels from derivatives/bads/<sub>_task-<task>_bads.json.

    Returns an empty list if the file does not exist.
    """
    fpath = paths.bads_file(label, task)
    if not fpath.exists():
        return []
    with fpath.open(encoding="utf-8") as f:
        return json.load(f).get("bads", [])


def save_bads(paths: Paths, label: str, task: str, bads: list[str]) -> None:
    """Write bad channels to derivatives/bads/<sub>_task-<task>_bads.json.

    Merges with existing entries (union, sorted).
    """
    fpath = paths.bads_file(label, task)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    existing: list[str] = []
    if fpath.exists():
        with fpath.open(encoding="utf-8") as f:
            existing = json.load(f).get("bads", [])
    merged = sorted(set(existing) | set(bads))
    with fpath.open("w", encoding="utf-8") as f:
        json.dump({"bads": merged}, f, indent=2)


# =============================================================================
# Logging
# =============================================================================


def setup_logging(paths: Paths, script_name: str) -> logging.Logger:
    log_dir = paths.log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(script_name)
    logger.info("Log file: %s", log_file)
    return logger


# =============================================================================
# Trigger decoding — Heidelberg Neuromag/TRIUX 6-channel STI system
# =============================================================================
#
# The Neuromag system encodes trigger codes across 6 binary STI channels
# (STI 001 – STI 006).  Each channel contributes one bit to the final
# decimal code:
#
#     STI 001 → bit 0 → value   1
#     STI 002 → bit 1 → value   2
#     STI 003 → bit 2 → value   4
#     STI 004 → bit 3 → value   8
#     STI 005 → bit 4 → value  16
#     STI 006 → bit 5 → value  32
#
# For the laser-pain paradigm only STI 001 is expected to fire (code = 1),
# but using the full 6-channel decoder ensures correctness if the stimulus
# computer or trigger box is ever reconfigured.
#
# Because hardware does not always switch all channels at exactly the same
# sample, a ±1-sample tolerance window is applied when combining channels.

STI_CHANNELS = [f"STI 00{i}" for i in range(1, 7)]
BIT_WEIGHTS = np.array([2**i for i in range(6)], dtype=np.int32)  # 1,2,4,8,16,32
TOLERANCE_SMP = 1  # ±1 sample jitter tolerance
MIN_DURATION = 0.002  # seconds — ignore pulses shorter than 2 ms (artefacts)
SHORTEST_EVT = 1  # samples


def _events_per_channel(raw: mne.io.BaseRaw) -> list[np.ndarray]:
    """Return rising-edge sample indices for each of the 6 STI channels."""
    result = []
    for ch in STI_CHANNELS:
        if ch not in raw.ch_names:
            result.append(np.array([], dtype=np.int32))
            continue
        try:
            evs = mne.find_events(
                raw,
                stim_channel=ch,
                min_duration=MIN_DURATION,
                shortest_event=SHORTEST_EVT,
                verbose=False,
            )[:, 0].astype(np.int32)
        except Exception:
            evs = np.array([], dtype=np.int32)

        # Remove pairs only 1 sample apart (hardware glitch)
        if len(evs) > 1:
            too_close = np.where(np.diff(evs) <= 1)[0]
            if len(too_close):
                evs = np.delete(evs, too_close)

        result.append(evs)
    return result


def _expand_with_tolerance(samples: np.ndarray, tol: int = TOLERANCE_SMP) -> np.ndarray:
    """Expand a sorted sample array by ±tol to create a tolerance window."""
    if len(samples) == 0:
        return np.array([], dtype=np.int32)
    offsets = np.arange(-tol, tol + 1, dtype=np.int32)
    expanded = (samples[:, np.newaxis] + offsets).ravel()
    return np.unique(expanded).astype(np.int32)


def _remove_duplicates(samples: np.ndarray, tol: int = TOLERANCE_SMP) -> np.ndarray:
    """Keep only the first sample of any cluster within ±tol."""
    if len(samples) == 0:
        return samples
    keep = np.ones(len(samples), dtype=bool)
    for i in range(1, len(samples)):
        if samples[i] - samples[i - 1] <= tol:
            keep[i] = False
    return samples[keep]


def _canonical_sample(candidates: np.ndarray, channel_samples: list[np.ndarray]) -> int:
    """Return the median sample from a cluster of candidate event times."""
    hits = []
    lo, hi = candidates.min(), candidates.max()
    for ch_smp in channel_samples:
        hits.extend(ch_smp[(ch_smp >= lo) & (ch_smp <= hi)].tolist())
    return int(np.median(hits)) if hits else int(candidates[0])


def get_triggers_from_raw(
    raw: mne.io.BaseRaw,
    adjust_ms: float = 0.0,
    min_duration: float = MIN_DURATION,
    shortest_event: int = SHORTEST_EVT,
    tolerance_smp: int = TOLERANCE_SMP,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Decode composite trigger codes from the 6 binary STI channels.

    Algorithm
    ---------
    1. Detect rising edges per STI channel with mne.find_events.
    2. Expand each channel's events by ±tolerance_smp samples.
    3. For each possible combination of channels (6-bit = 64 possible codes),
       find samples present in ALL channels of that combination and absent
       in higher-order combinations (to avoid double-counting).
    4. Apply optional latency correction (adjust_ms).

    Parameters
    ----------
    raw           : mne.io.BaseRaw
    adjust_ms     : float
        Latency correction in milliseconds (positive = shift later).
    min_duration  : float
        Minimum STI pulse duration in seconds.
    shortest_event: int
        Minimum event duration in samples.
    tolerance_smp : int
        ±sample tolerance for combining channels (default 1).

    Returns
    -------
    digital_trigger : np.ndarray, shape (n_samples,)
        Trigger code at each sample (0 where no event).
    trigger_events_df : pd.DataFrame
        Columns: sample_index, trigger_value
    """
    if not raw.preload:
        raw.load_data(verbose=False)

    ch_events = _events_per_channel(raw)
    ch_exp = [_expand_with_tolerance(e, tolerance_smp) for e in ch_events]

    found_samples: set[int] = set()
    event_list: list[tuple[int, int]] = []

    active_channels = [i for i, e in enumerate(ch_events) if len(e) > 0]

    for n_bits in range(len(active_channels), 0, -1):
        for combo in combinations(active_channels, n_bits):
            intersection = ch_exp[combo[0]].copy()
            for i in combo[1:]:
                intersection = np.intersect1d(intersection, ch_exp[i])
            if len(intersection) == 0:
                continue

            intersection = np.array(
                [s for s in intersection if s not in found_samples],
                dtype=np.int32,
            )
            if len(intersection) == 0:
                continue

            intersection = _remove_duplicates(np.sort(intersection), tolerance_smp)
            code = int(sum(BIT_WEIGHTS[i] for i in combo))

            for smp in intersection:
                if smp not in found_samples:
                    window = np.arange(
                        smp - tolerance_smp, smp + tolerance_smp + 1, dtype=np.int32
                    )
                    canon = _canonical_sample(window, [ch_events[i] for i in combo])
                    event_list.append((canon, code))
                    # Mark ALL individual channel samples that fired in this
                    # window as found, not just the ±tolerance window around
                    # the canonical. This prevents sub-combinations of the same
                    # hardware event from appearing as separate events.
                    for ch_i in combo:
                        for ch_smp in ch_events[ch_i]:
                            if abs(int(ch_smp) - int(smp)) <= tolerance_smp:
                                for s in range(
                                    int(ch_smp) - tolerance_smp,
                                    int(ch_smp) + tolerance_smp + 1,
                                ):
                                    found_samples.add(s)
                    # Also mark the canonical window
                    for s in range(canon - tolerance_smp, canon + tolerance_smp + 1):
                        found_samples.add(s)

    if not event_list:
        return (
            np.zeros(raw.n_times, dtype=int),
            pd.DataFrame(columns=["sample_index", "trigger_value"]),
        )

    # Merge events within tolerance_smp of each other — keep only the highest
    # composite code. This replicates MATLAB loadmeg.m line 316-319 which
    # groups all channels firing within TRIGmaxDT=1 sample into one trigword.
    # Without this, sub-combinations of a composite trigger (e.g. code 4 from
    # a hardware pulse that also triggers code 12 = 4+8) appear as separate
    # events.
    if event_list:
        event_list_sorted = sorted(event_list)  # sort by sample
        merged = []
        i = 0
        while i < len(event_list_sorted):
            smp, code = event_list_sorted[i]
            j = i + 1
            # Collect all events at the exact same sample (simultaneous hardware triggers)
            while j < len(event_list_sorted) and event_list_sorted[j][0] == smp:
                code = code | event_list_sorted[j][1]  # bitwise OR = combine
                j += 1
            merged.append((smp, code))
            i = j
        event_list = merged

    event_arr = np.array(sorted(event_list), dtype=np.int32)

    if adjust_ms != 0.0:
        shift = int(round(adjust_ms * 1e-3 * raw.info["sfreq"]))
        event_arr[:, 0] += shift
        event_arr = event_arr[(event_arr[:, 0] >= 0) & (event_arr[:, 0] < raw.n_times)]

    event_arr[:, 0] = np.clip(event_arr[:, 0], 0, raw.n_times - 1)

    digital_trigger = np.zeros(raw.n_times, dtype=int)
    digital_trigger[event_arr[:, 0]] = event_arr[:, 1]

    trigger_events_df = pd.DataFrame(
        {
            "sample_index": event_arr[:, 0],
            "trigger_value": event_arr[:, 1],
        }
    )

    codes, counts = np.unique(event_arr[:, 1], return_counts=True)
    print("  Trigger codes found:")
    for code, count in zip(codes, counts):
        bits = [i for i in range(6) if code & (1 << i)]
        print(f"    code {code:3d}  (bits {bits})  →  {count} events")

    return digital_trigger, trigger_events_df
