#!/usr/bin/env python3
"""
ratings_io.py
--------------
Single shared source for reading behavioural ratings, used by epoch.py,
plot_ratings.py, and trial_counts.py. Previously this logic was
duplicated almost identically across all three — this module replaces
those duplicates.

Rating source priority (matches the project's standard convention):
    1. Triggercheck JSON (derivatives/trigger_check/sub-{label}/) — used
       when present; contains corrected trial indexing for compound
       trigger subjects.
    2. Behavioural mat file (rawdata/sub-{label}/beh/) — fallback for
       all other subjects.

Handles the nested-array mat parsing fix (e.g. a cell stored as ['70']
instead of the bare string '70').
"""

import json
import re
from pathlib import Path

from core import Paths, sub_id


def _unwrap_cell(cell):
    """Recursively unwrap nested arrays/lists down to a scalar string."""
    while hasattr(cell, "__len__") and not isinstance(cell, str) and len(cell) > 0:
        try:
            cell = cell.flat[0] if hasattr(cell, "flat") else cell[0]
        except (IndexError, AttributeError):
            break
    return re.sub(r"""[\[\]'"]""", "", str(cell).strip()).strip()


def read_ratings_from_json(json_path: Path, task: str) -> list:
    """Returns a list of ratings only (float or None), discarding the
    quality field. Thin wrapper around read_ratings_and_quality_from_json
    to avoid duplicating the parsing loop."""
    ratings, _ = read_ratings_and_quality_from_json(json_path, task)
    return ratings


def read_ratings_from_mat(mat_path: Path) -> list:
    """Returns a list of ratings only (float or None for miss), discarding
    the quality field. Thin wrapper around read_ratings_and_quality_from_mat."""
    ratings, _ = read_ratings_and_quality_from_mat(mat_path, task="pinprick")
    return ratings


def load_ratings(paths: Paths, label: str, task: str) -> list:
    """Load the full rating list (ratings only, no quality) for a
    subject/task, JSON first then mat fallback. Returns [] if neither
    source is found."""
    ratings, _ = load_ratings_and_quality(paths, label, task)
    return ratings


QUALITY_LABELS = {"s", "sb", "b", "w", "n"}  # spitz, spitz-brennend, brennend, warm, nichts


def read_ratings_and_quality_from_json(json_path: Path, task: str):
    """Same as read_ratings_from_json but also returns the quality letter
    per trial (laser only; None for other tasks)."""
    with json_path.open() as f:
        tc = json.load(f)
    stim_key = "is_laser" if task == "laser" else "is_stim"
    ratings, qualities = [], []
    for t in tc.get("trials", []):
        if not t.get(stim_key, False):
            # Non-stim bundle: either no stimulation actually occurred
            # (e.g. robot missed the arm, flag_reason "miss_in_mat") or
            # the trigger/timing for the trial couldn't be recovered
            # (e.g. "pinprick_trigger_missing"). Either way there's no
            # usable stimulation event, so record it as a miss rather
            # than dropping it from the trial list entirely - dropping
            # it silently shrinks n_total and hides these trials from
            # n_miss.
            ratings.append(None)
            qualities.append(None)
            continue
        val = t.get("intensity_mat", t.get("intensity_fif"))
        ratings.append(None if (val is None or val == -1) else float(val))
        qualities.append(t.get("quality_fif"))
    return ratings, qualities


def read_ratings_and_quality_from_mat(mat_path: Path, task: str = "pinprick"):
    """Same as read_ratings_from_mat but also returns the quality letter
    per trial. For laser files, responses has shape (3, n): row 0 =
    intensity, row 1 = quality letter (s/sb/b/w/n), row 2 = unused.
    For pinprick/tactile, responses has shape (1, n): intensity only."""
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


def load_ratings_and_quality(paths: Paths, label: str, task: str):
    """Returns (ratings_list, qualities_list); ratings_list has None for
    miss. qualities_list is all None for non-laser tasks."""
    json_path = (paths.deriv / "trigger_check" / sub_id(label)
                 / f"{sub_id(label)}_task-{task}_triggercheck.json")
    if json_path.exists():
        try:
            return read_ratings_and_quality_from_json(json_path, task)
        except Exception:
            pass

    mat_path = (paths.raw / sub_id(label) / "beh"
                / f"{sub_id(label)}_task-{task}_ratings.mat")
    if mat_path.exists():
        try:
            return read_ratings_and_quality_from_mat(mat_path, task)
        except Exception:
            pass
    return [], []
