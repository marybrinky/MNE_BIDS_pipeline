#!/usr/bin/env python3 # changed by Marie 260427 to solve the laser trigger issue and to add butterfly plotting function
"""
epoch.py
--------
Trigger decoding and epoching for the laser-pain MEG study.

Steps
-----
1. Load preprocessed raw from derivatives/prep/
2. Decode composite trigger codes from the 6 STI channels (via core.py)
3. Build MNE events array from the decoded triggers
4. Epoch the data according to EPOCH_CONFIGS in core.py
5. Apply peak-to-peak and flat-signal rejection
6. Save epochs to derivatives/epochs/

Trigger decoding
----------------
Instead of calling mne.find_events() directly (which fails on the short
artefact pulses produced by the Heidelberg STI lines), epoch.py uses
get_triggers_from_raw() from core.py.  This function reads all 6 STI
channels individually, applies a ±1-sample tolerance window to handle
inter-channel jitter, and assembles composite codes via bit-wise OR —
the same validated approach used in the RHT pipeline.

Usage
-----
    # Inspect trigger codes before committing to EPOCH_CONFIGS
    python epoch.py --subjects 4382 --show-triggers

    # All subjects, default epoch config
    python epoch.py

    # Specific task only, no amplitude rejection
    python epoch.py --tasks laser --no-reject

    # Overwrite existing
    python epoch.py --overwrite

    # Plot butterfly from existing epoched data
    python epoch.py --subjects 4382 --tasks laser --plot-butterfly
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

import json

from core import (
    DEFAULT_EPOCH_CONFIG,
    EPOCH_CONFIGS,
    TASKS,
    TRIGGERCHECK_SUBJECTS,
    Paths,
    get_triggers_for,
    get_triggers_from_raw,
    load_subjects,
    setup_logging,
    sub_id,
)

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")


def _read_ratings_from_json(json_path: Path, task: str) -> list[float | None]:
    """Read per-trial intensity ratings from a triggercheck JSON sidecar.

    Returns a list of intensity values (float or None for miss/-1) ordered
    by trial index (stim bundles only, in bundle order).
    For laser JSONs: uses intensity_fif from is_laser=True bundles.
    For pinprick/tactile JSONs: uses intensity_mat from is_stim=True bundles.
    """
    with json_path.open() as f:
        tc = json.load(f)

    trials = tc.get("trials", [])
    ratings = []

    # Determine which field marks a valid stimulus trial
    stim_key = "is_laser" if task == "laser" else "is_stim"
    # Prefer intensity_mat if present, fall back to intensity_fif
    for t in trials:
        if not t.get(stim_key, False):
            continue
        val = t.get("intensity_mat", t.get("intensity_fif"))
        if val is None or val == -1:
            ratings.append(None)
        else:
            try:
                ratings.append(float(val))
            except (TypeError, ValueError):
                ratings.append(None)
    return ratings


def _read_ratings_from_mat(mat_path: Path) -> list[float | None]:
    """Read intensity ratings from a behavioural mat file.

    Returns a list of 50 floats (None for miss entries).

    Some trials store the response wrapped in an extra array layer
    (e.g. trial cell containing ['70'] instead of the bare string '70'),
    which can produce a literal "['70']" string when stringified directly.
    We unwrap recursively and strip brackets/quotes before parsing.
    """
    import scipy.io
    import re
    mat   = scipy.io.loadmat(str(mat_path))
    r     = mat["response"][0, 0]
    resps = r["responses"]
    n     = resps.shape[1]
    ratings = []
    for i in range(n):
        cell = resps[0, i]
        # Recursively unwrap nested arrays/lists down to a scalar
        while hasattr(cell, "__len__") and not isinstance(cell, str) and len(cell) > 0:
            try:
                cell = cell.flat[0] if hasattr(cell, "flat") else cell[0]
            except (IndexError, AttributeError):
                break
        val = str(cell).strip()
        # Strip any stray brackets/quotes left over from str() of an array
        val = re.sub(r"""[\[\]'"]""", "", val).strip()
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
    """Boolean mask of trials to keep based on behavioural ratings.

    trial_filter : "perceived"  keeps only trials rated > 0 (opt-in via --perceived)
                   "all"        keeps all trials (default)

    Rating source priority:
      1. Triggercheck JSON  (derivatives/trigger_check/sub-{label}/
                             sub-{label}_task-{task}_triggercheck.json)
         — used when present; contains corrected trial indexing
      2. Behavioural mat file  (rawdata/sub-{label}/beh/
                                sub-{label}_task-{task}_ratings.mat)
         — fallback for all subjects without a JSON

    The Nth rating maps to the Nth epoch in the events array.
    Miss trials (-1 / "miss") are always excluded regardless of filter.
    """
    tag = f"sub-{label} / {task}"

    if trial_filter == "all":
        logger.info("[%s]  Trial filter: all (no filtering)", tag)
        return None

    # --- Source 1: triggercheck JSON -------------------------------------
    json_path = (
        paths.deriv / "trigger_check" / sub_id(label)
        / f"{sub_id(label)}_task-{task}_triggercheck.json"
    )
    ratings = None

    if json_path.exists():
        try:
            ratings = _read_ratings_from_json(json_path, task)
            logger.info(
                "[%s]  Ratings from triggercheck JSON (%d stim trials found)",
                tag, len(ratings)
            )
        except Exception as e:
            logger.warning("[%s]  JSON rating read failed: %s — trying mat", tag, e)
            ratings = None

    # --- Source 2: mat file fallback -------------------------------------
    if ratings is None:
        mat_path = (
            paths.raw / sub_id(label) / "beh"
            / f"{sub_id(label)}_task-{task}_ratings.mat"
        )
        if mat_path.exists():
            try:
                ratings = _read_ratings_from_mat(mat_path)
                logger.info(
                    "[%s]  Ratings from mat file (%d trials found)",
                    tag, len(ratings)
                )
            except Exception as e:
                logger.warning("[%s]  Mat rating read failed: %s — using all trials", tag, e)
                return None
        else:
            logger.warning(
                "[%s]  No rating source found (no JSON, no mat file) "
                "— using all trials", tag
            )
            return None

    # --- Build mask ------------------------------------------------------
    if len(ratings) != n_epochs:
        logger.warning(
            "[%s]  Rating count mismatch: %d ratings but %d epochs "
            "— skipping filter",
            tag, len(ratings), n_epochs
        )
        return None

    mask   = np.zeros(n_epochs, dtype=bool)
    n_kept = 0
    for i, val in enumerate(ratings):
        if val is None:
            continue   # miss — always excluded
        if val > 0:
            mask[i] = True
            n_kept  += 1

    logger.info(
        "[%s]  Trial filter 'perceived': %d / %d trials kept (intensity > 0)",
        tag, n_kept, n_epochs
    )
    return mask


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Trigger inspection
# ---------------------------------------------------------------------------


def show_triggers(paths: Paths, label: str, task: str, logger) -> pd.DataFrame:
    """Decode triggers from the preprocessed (or raw) file and log a summary."""
    prep_file = paths.prep_raw(label, task, desc="preproc")
    src_file = prep_file if _exists(prep_file) else paths.raw_meg(label, task)

    if not src_file.exists():
        logger.warning(
            "[sub-%s / %s]  No file found for trigger inspection", label, task
        )
        return pd.DataFrame()

    logger.info("[sub-%s / %s]  Decoding triggers from: %s", label, task, src_file.name)
    raw = mne.io.read_raw_fif(src_file, preload=True, verbose=False)
    _, events_df = get_triggers_from_raw(raw)

    if events_df.empty:
        logger.warning("[sub-%s / %s]  No triggers found", label, task)
        return events_df

    events_df["time_s"] = events_df["sample_index"] / raw.info["sfreq"]
    trigger_map = get_triggers_for(label, task)
    logger.info("[sub-%s / %s]  Active trigger map: %s", label, task, trigger_map)
    for code, grp in events_df.groupby("trigger_value"):
        logger.info(
            "[sub-%s / %s]  code %3d → %d events  (first %.3f s, last %.3f s)",
            label,
            task,
            code,
            len(grp),
            grp["time_s"].iloc[0],
            grp["time_s"].iloc[-1],
        )
    return events_df


# ---------------------------------------------------------------------------
# Triggercheck: filter code-4 events to confirmed laser bundles from JSON
# ---------------------------------------------------------------------------


def filter_laser_events_by_json(
    events: np.ndarray,
    events_df,
    paths: Paths,
    label: str,
    logger,
    tag: str,
) -> tuple[np.ndarray, list[int]]:
    """Match the full JSON trigger raster onto the fif trigger stream.

    The JSON contains all bundles in recording order, each with a triggers
    array e.g. [4, 11, 2, 1, 3]. These are concatenated (skipping 123
    placeholders) into one long template and matched against the fif trigger
    sequence in a single forward pass.

    This is robust: the full recording context constrains the match so even
    ambiguous individual patterns are correctly located.

    Returns (events_array, confirmed_bundle_numbers) where
    confirmed_bundle_numbers[i] is the JSON bundle number for the ith
    confirmed laser epoch, in recording order.
    """
    json_path = (
        paths.deriv / "trigger_check" / sub_id(label)
        / f"{sub_id(label)}_task-laser_triggercheck.json"
    )

    if not json_path.exists():
        logger.warning(
            "[%s]  No triggercheck JSON — using ALL code-4 events", tag
        )
        return events, []

    with json_path.open() as f:
        tc = json.load(f)

    trials = tc.get("trials", [])

    # All fif triggers sorted by time
    all_samples = events_df["sample_index"].values.astype(int)
    all_codes   = events_df["trigger_value"].values.astype(int)
    sort_idx    = np.argsort(all_samples)
    all_samples = all_samples[sort_idx]
    all_codes   = all_codes[sort_idx]

    # Build the full template from all bundles in order, skipping 123s.
    # For each real (non-123) trigger in each bundle, record:
    #   - the trigger code
    #   - which bundle it belongs to
    #   - whether it is the FIRST trigger of the bundle (= stim onset)
    #   - whether the bundle is a laser trial
    template_codes    = []
    template_bundle   = []
    template_is_first = []
    template_is_laser = []

    for trial in trials:
        bundle_num = trial["bundle"]
        is_laser   = trial.get("is_laser", False)
        pattern    = [c for c in trial.get("triggers", []) if c != 123]
        for k, code in enumerate(pattern):
            template_codes.append(code)
            template_bundle.append(bundle_num)
            template_is_first.append(k == 0)
            template_is_laser.append(is_laser)

    template_codes = np.array(template_codes, dtype=int)
    n_template     = len(template_codes)

    if n_template == 0:
        logger.warning("[%s]  JSON template is empty", tag)
        return np.zeros((0, 3), dtype=np.int32), []

    # Single forward pass: scan fif for the template
    # Collect the fif position of each template element
    fif_positions = []   # fif index for each template position
    fif_ptr       = 0    # current position in fif stream

    for t_idx in range(n_template):
        expected_code = template_codes[t_idx]
        # Advance fif_ptr until we find this code
        found = False
        while fif_ptr < len(all_codes):
            if all_codes[fif_ptr] == expected_code:
                fif_positions.append(fif_ptr)
                fif_ptr += 1
                found = True
                break
            fif_ptr += 1
        if not found:
            logger.warning(
                "[%s]  Template position %d (code %d, bundle %d) not found "
                "in fif from position %d onward — stopping match",
                tag, t_idx, expected_code, template_bundle[t_idx], fif_ptr
            )
            # Use what we have so far
            template_codes    = template_codes[:t_idx]
            template_bundle   = template_bundle[:t_idx]
            template_is_first = template_is_first[:t_idx]
            template_is_laser = template_is_laser[:t_idx]
            break

    logger.info(
        "[%s]  Template match: %d / %d trigger codes matched in fif",
        tag, len(fif_positions), n_template
    )

    # Extract confirmed laser onsets: first trigger of each is_laser bundle
    confirmed_onsets  = []
    confirmed_bundles = []

    for i, fif_idx in enumerate(fif_positions):
        if template_is_first[i] and template_is_laser[i]:
            confirmed_onsets.append(int(all_samples[fif_idx]))
            confirmed_bundles.append(int(template_bundle[i]))

    if not confirmed_onsets:
        logger.warning("[%s]  No confirmed laser onsets found", tag)
        return np.zeros((0, 3), dtype=np.int32), []

    new_events = np.zeros((len(confirmed_onsets), 3), dtype=np.int32)
    new_events[:, 0] = confirmed_onsets
    new_events[:, 2] = 4

    n_nonlaser = len([t for t in trials if not t.get("is_laser", False)])
    logger.info(
        "[%s]  Triggercheck filter: %d confirmed laser onsets "
        "(%d bundles excluded as non-laser)",
        tag, len(confirmed_onsets), n_nonlaser,
    )
    return new_events, confirmed_bundles


# ---------------------------------------------------------------------------
# Trigger spec resolver + events array builder
# ---------------------------------------------------------------------------


def _resolve_trigger_spec(
    spec: int | dict | None,
    available_codes: np.ndarray,
    *,
    events_df: pd.DataFrame | None = None,
    sfreq: float | None = None,
) -> list[int] | np.ndarray:
    """Resolve a trigger specification to a list of matching integer codes,
    or a compound events array for compound trigger specs.

    Supported formats:
        None                        ->  not yet defined, returns []
        int                         ->  exact match, e.g. 1
        {"min": 50}                 ->  all codes >= 50
        {"max": 10}                 ->  all codes <= 10
        {"min": 4, "max": 8}        ->  all codes in [4, 8]
        {"compound": True, ...}     ->  trigger_a followed by trigger_b
    """
    if spec is None:
        return []
    if isinstance(spec, int):
        return [spec] if spec in available_codes else []
    if isinstance(spec, dict):
        # --- threshold spec ----------------------------------------------
        lo = spec.get("min", -np.inf)
        hi = spec.get("max", np.inf)
        return [int(c) for c in available_codes if lo <= c <= hi]
    raise TypeError(f"Unsupported trigger spec type: {type(spec)}")


def build_events(
    events_df: pd.DataFrame,
    trigger_map: dict[str, int | dict | None],
    logger,
    tag: str,
    sfreq: float | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Convert trigger DataFrame to MNE events array + event_id dict.

    When a threshold spec ({"min": N}) matches multiple codes, each code
    is registered separately (e.g. "stimulus_52": 52) so MNE can handle
    them individually or merge them with mne.merge_events() downstream.

    Compound trigger specs ({"compound": True, ...}) are resolved via
    mne.event.define_target_events and merged into the final events array.
    """
    available = events_df["trigger_value"].unique()
    event_id: dict[str, int] = {}
    any_defined = False

    for name, spec in trigger_map.items():
        if spec is None:
            continue
        any_defined = True
        matched = _resolve_trigger_spec(spec, available)

        if not matched:
            logger.warning("[%s]  No events for trigger spec %s=%s", tag, name, spec)
            continue
        if len(matched) == 1:
            event_id[name] = matched[0]
        else:
            for code in matched:
                event_id[f"{name}_{code}"] = code
            logger.info(
                "[%s]  %s matched %d codes: %s", tag, name, len(matched), matched
            )

    if not any_defined:
        raise ValueError(
            f"[{tag}]  No trigger codes defined in EPOCH_CONFIGS['triggers']. "
            f"Run with --show-triggers to identify codes, then update core.py."
        )
    if not event_id:
        raise ValueError(
            f"[{tag}]  Trigger specs defined but no matching events found. "
            f"Available codes: {sorted(available)}. Check EPOCH_CONFIGS."
        )

    valid_codes = set(event_id.values())
    df = events_df[events_df["trigger_value"].isin(valid_codes)].copy()

    events = np.zeros((len(df), 3), dtype=np.int32)
    events[:, 0] = df["sample_index"].values
    events[:, 2] = df["trigger_value"].values
    events        = events[events[:, 0].argsort()]

    code_to_name = {v: k for k, v in event_id.items()}
    for code in sorted(event_id.values()):
        n = (events[:, 2] == code).sum()
        logger.info("[%s]  %s (code %d): %d events", tag, code_to_name[code], code, n)
    logger.info("[%s]  Total events selected: %d", tag, len(events))

    return events, event_id


# ---------------------------------------------------------------------------
# Butterfly plot
# ---------------------------------------------------------------------------


def plot_butterfly(
    paths: Paths, label: str, task: str, epoch_config: str, logger,
    trial_filter: str = "all",
) -> None:
    """Load existing epochs and plot a butterfly (one line per channel)
    of the condition average, with spatial colours.

    Parameters
    ----------
    trial_filter : str
        "all"       — plot all epochs (default)
        "perceived" — plot only epochs rated > 0 and not "miss"
    """
    cfg = EPOCH_CONFIGS[epoch_config]
    task_cfg = cfg.get("task_overrides", {}).get(task, {})
    cfg = {**cfg, **task_cfg}
    desc = f"{cfg['desc']}-preproc"
    tag = f"sub-{label} / {task} / {epoch_config}"
    epo_file = paths.epochs(label, task, desc=desc)

    if not _exists(epo_file):
        logger.warning("[%s]  No epochs file found: %s", tag, epo_file)
        return

    logger.info("[%s]  Loading epochs for butterfly plot: %s", tag, epo_file.name)
    epochs = mne.read_epochs(epo_file, preload=True, verbose=False)

    # Apply trial filter
    trial_mask = load_trial_mask(paths, label, task, trial_filter, len(epochs), logger)
    if trial_mask is not None:
        epochs = epochs[trial_mask]
        if len(epochs) == 0:
            logger.warning("[%s]  No epochs remaining after filter — skipping", tag)
            return
        logger.info("[%s]  Butterfly: %d epochs after '%s' filter",
                    tag, len(epochs), trial_filter)

    for condition in epochs.event_id:
        try:
            evoked = epochs[condition].average()
        except Exception as e:
            logger.warning(
                "[%s]  Could not average condition %s: %s", tag, condition, e
            )
            continue

        fig = evoked.plot(spatial_colors=True, show=False)
        fig.suptitle(f"sub-{label} / {task} / {condition}", fontsize=10)
        logger.info("[%s]  Showing butterfly for condition: %s", tag, condition)

    plt.show()



# ---------------------------------------------------------------------------
# Trial count statistics
# ---------------------------------------------------------------------------


def save_trial_stats(paths: Paths, all_stats: list[dict], logger) -> None:
    """Save per-subject trial count statistics and group averages to TSV."""
    if not all_stats:
        return

    import csv as _csv
    from collections import defaultdict

    out_dir = paths.log_dir() / "trial_counts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-subject TSV
    per_sub = out_dir / "epoch_trial_counts.tsv"
    fieldnames = [
        "subject", "task", "epoch_config",
        "n_total", "n_after_reject", "n_after_perceived",
        "n_rejected", "n_nonperceived",
        "pct_rejected", "pct_final",
    ]
    with open(per_sub, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames, delimiter="	")
        w.writeheader()
        w.writerows(all_stats)
    logger.info("Trial count stats saved: %s", per_sub)

    # Group average — one row per task/epoch_config
    groups: dict = defaultdict(list)
    for s in all_stats:
        groups[(s["task"], s["epoch_config"])].append(s)

    group_rows = []
    for (task, cfg), rows in sorted(groups.items()):
        def avg(field, rows=rows):
            vals = [r[field] for r in rows if r.get(field) is not None]
            return round(sum(vals) / len(vals), 1) if vals else None
        group_rows.append({
            "task": task, "epoch_config": cfg, "n_subjects": len(rows),
            "mean_n_total":           avg("n_total"),
            "mean_n_after_reject":    avg("n_after_reject"),
            "mean_n_after_perceived": avg("n_after_perceived"),
            "mean_pct_rejected":      avg("pct_rejected"),
            "mean_pct_final":         avg("pct_final"),
        })

    group_out = out_dir / "epoch_trial_counts_group.tsv"
    group_fields = [
        "task", "epoch_config", "n_subjects",
        "mean_n_total", "mean_n_after_reject", "mean_n_after_perceived",
        "mean_pct_rejected", "mean_pct_final",
    ]
    with open(group_out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=group_fields, delimiter="	")
        w.writeheader()
        w.writerows(group_rows)
    logger.info("Group trial count stats saved: %s", group_out)


# ---------------------------------------------------------------------------
# Main per-subject pipeline
# ---------------------------------------------------------------------------


def epoch_one(
    paths: Paths,
    label: str,
    task: str,
    epoch_config: str,
    reject: bool,
    overwrite: bool,
    logger,
    reject_all_trials: bool = True,
) -> tuple[bool, str | None]:
    """Epoch one subject/task.  Returns (success, qc_warning_or_None)."""
    cfg = EPOCH_CONFIGS[epoch_config]
    task_cfg = cfg.get("task_overrides", {}).get(task, {})
    cfg = {**cfg, **task_cfg}
    desc = f"{cfg['desc']}-preproc"
    tag = f"sub-{label} / {task} / {epoch_config}"
    out_file = paths.epochs(label, task, desc=desc)

    if _exists(out_file) and not overwrite:
        logger.info("[%s]  SKIP (exists)", tag)
        return True, None, None

    prep_file = paths.prep_raw(label, task, desc=cfg["prep_desc"])
    if not _exists(prep_file):
        logger.warning("[%s]  Preprocessed file not found: %s", tag, prep_file.name)
        return False, None, None

    logger.info("[%s]  Loading: %s", tag, prep_file.name)
    raw = mne.io.read_raw_fif(prep_file, preload=True, verbose=False)

    # -- Trigger decoding via 6-channel STI decoder -----------------------
    try:
        _, events_df = get_triggers_from_raw(raw)
    except Exception as e:
        logger.warning("[%s]  Trigger decoding failed: %s", tag, e)
        return False, None, None

    if events_df.empty:
        logger.warning("[%s]  No triggers found after STI decoding", tag)
        return False, None, None

    # -- Build MNE events array -------------------------------------------
    # Use subject- and task-specific trigger codes if defined in
    # SUBJECT_TRIGGERS, otherwise fall back to EPOCH_CONFIGS defaults.
    trigger_map = get_triggers_for(label, task, epoch_config)
    logger.info("[%s]  Trigger map: %s", tag, trigger_map)
    try:
        events, event_id = build_events(
            events_df,
            trigger_map,
            logger,
            tag,
        )
    except ValueError as e:
        logger.warning("%s", e)
        return False, None, None

    # For triggercheck subjects (laser): filter code-4 to confirmed laser
    # bundles only, removing rating-4s that share the same STI channel
    json_bundle_order: list[int] = []  # bundle numbers for confirmed laser epochs
    if task == "laser" and label in TRIGGERCHECK_SUBJECTS:
        events, json_bundle_order = filter_laser_events_by_json(events, events_df, paths, label, logger, tag)
        if len(events) == 0:
            logger.warning("[%s]  No events remain after triggercheck filter", tag)
            return False, None, None

    # -- Epoch ------------------------------------------------------------
    reject_criteria = cfg.get("reject") if reject else None
    flat_criteria = cfg.get("flat") if reject else None

    logger.info(
        "[%s]  Epoching: tmin=%.2f s  tmax=%.2f s  baseline=%s  reject=%s",
        tag,
        cfg["tmin"],
        cfg["tmax"],
        cfg.get("baseline"),
        "enabled" if reject else "disabled",
    )

    try:
        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=cfg["tmin"],
            tmax=cfg["tmax"],
            baseline=cfg.get("baseline", (cfg["tmin"], 0)),
            reject=reject_criteria,
            flat=flat_criteria,
            picks="meg",
            preload=True,
            event_repeated="drop",
            verbose=False,
        )
    except Exception as e:
        logger.warning("[%s]  Epoching failed: %s", tag, e)
        return False, None, None

    n_total = len(events)
    n_kept = len(epochs)
    n_rejected = n_total - n_kept

    logger.info(
        "[%s]  Epochs: %d total  |  %d kept  |  %d rejected (%.1f%%)",
        tag,
        n_total,
        n_kept,
        n_rejected,
        100 * n_rejected / n_total if n_total > 0 else 0,
    )

    if n_kept == 0:
        logger.warning("[%s]  All epochs rejected — check rejection thresholds", tag)
        return False, None, None

    # -- QC: sweep count --------------------------------------------------
    # For triggercheck subjects, read n_expected from the JSON
    # (e.g. 45 for sub-1409 laser) rather than the hardcoded 50.
    # For all other subjects use EPOCH_CONFIGS n_expected (default 50).
    expected = cfg.get("n_expected")
    json_path = (
        paths.deriv / "trigger_check" / sub_id(label)
        / f"{sub_id(label)}_task-{task}_triggercheck.json"
    )
    if json_path.exists():
        try:
            with json_path.open() as _f:
                _tc = json.load(_f)
            # Use n_laser_bundles for laser, n_stim_bundles for others
            _key = "n_laser_bundles" if task == "laser" else "n_stim_bundles"
            if _key in _tc:
                expected = _tc[_key]
        except Exception:
            pass  # fall back to cfg n_expected

    qc_warn = None
    if expected and n_kept < expected:
        qc_warn = (
            f"sub-{label} / {task} / {epoch_config}: "
            f"only {n_kept}/{expected} sweeps kept ({n_rejected} rejected)"
        )
        logger.warning("[%s]  QC FAIL — %s", tag, qc_warn)
    elif expected:
        logger.info("[%s]  QC OK — %d/%d sweeps", tag, n_kept, expected)

    n_after_reject = len(epochs)

    # -- Apply perceived trial filter ------------------------------------
    # Load ratings for all n_total original stimuli, then use
    # epochs.selection (original 0-based event indices) to map
    # surviving post-rejection epochs back to their original index.
    # This correctly handles the mismatch between 50 ratings and
    # fewer post-rejection epochs.
    n_after_perceived = n_after_reject
    if not reject_all_trials:
        if json_bundle_order:
            # Triggercheck subject: ratings are stored per bundle in the JSON.
            # json_bundle_order[i] = bundle number for the ith confirmed epoch
            # (before amplitude rejection). epochs.selection tells us which
            # of those survived rejection. Map each surviving epoch back to
            # its bundle number, then look up intensity_fif for that bundle.
            json_path = (
                paths.deriv / "trigger_check" / sub_id(label)
                / f"{sub_id(label)}_task-laser_triggercheck.json"
            )
            with json_path.open() as _f:
                tc = json.load(_f)
            bundle_rating = {
                t["bundle"]: t.get("intensity_fif")
                for t in tc["trials"]
            }
            # Build a rating list indexed by position in json_bundle_order
            # then use epochs.selection to pick the survivors
            ratings_by_position = [
                bundle_rating.get(b) for b in json_bundle_order
            ]
            keep = []
            for sel_idx in epochs.selection:
                val = ratings_by_position[sel_idx] if sel_idx < len(ratings_by_position) else None
                keep.append(val is not None and val != -1 and float(val) > 0)
            keep = np.array(keep, dtype=bool)
            n_perceived_total = sum(
                1 for v in ratings_by_position
                if v is not None and v != -1 and float(v) > 0
            )
            logger.info(
                "[%s]  Ratings from triggercheck JSON (%d stim trials found)",
                tag, len(json_bundle_order)
            )
            logger.info(
                "[%s]  Trial filter 'perceived': %d / %d trials kept (intensity > 0)",
                tag, n_perceived_total, len(json_bundle_order)
            )
            epochs = epochs[keep]
            n_after_perceived = len(epochs)
            if n_after_perceived == 0:
                logger.warning("[%s]  No perceived trials remain — skipping", tag)
                return False, None, None
        else:
            perceived_mask_full = load_trial_mask(
                paths, label, task, "perceived", n_total, logger
            )
            if perceived_mask_full is not None:
                keep = np.array([
                    perceived_mask_full[i] for i in epochs.selection
                ], dtype=bool)
                epochs = epochs[keep]
                n_after_perceived = len(epochs)
                if n_after_perceived == 0:
                    logger.warning("[%s]  No perceived trials remain — skipping", tag)
                    return False, None, None
            else:
                logger.info("[%s]  No rating source found — keeping all epochs", tag)
    else:
        logger.info("[%s]  --perceived not set: keeping all non-rejected trials", tag)

    # -- Trial count statistics ------------------------------------------
    pct_rejected  = 100 * (n_total - n_after_reject)  / n_total if n_total > 0 else 0
    pct_final     = 100 * n_after_perceived             / n_total if n_total > 0 else 0
    trial_stats = {
        "subject":         label,
        "task":            task,
        "epoch_config":    epoch_config,
        "n_total":          expected if expected else n_total,
        "n_after_reject":   n_after_reject,
        "n_after_perceived":n_after_perceived,
        "n_rejected":      n_total - n_after_reject,
        "n_nonperceived":  n_after_reject - n_after_perceived,
        "pct_rejected":    round(pct_rejected, 1),
        "pct_final":       round(pct_final, 1),
    }
    logger.info(
        "[%s]  Trial counts: total=%d  after_reject=%d  after_perceived=%d  "
        "(%.1f%% final)",
        tag, n_total, n_after_reject, n_after_perceived, pct_final
    )

    # -- Save -------------------------------------------------------------
    out_file.parent.mkdir(parents=True, exist_ok=True)
    epochs.save(out_file, overwrite=True, verbose=False)
    logger.info("[%s]  Saved: %s", tag, out_file.name)
    return True, qc_warn, trial_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Trigger decoding and epoching for the laser-pain MEG study."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--tasks", nargs="+", default=None, choices=TASKS)
    parser.add_argument(
        "--epoch-config",
        default=None,
        choices=list(EPOCH_CONFIGS.keys()),
        help="Run a specific epoch config only. By default all configs are run.",
    )
    parser.add_argument(
        "--show-triggers",
        action="store_true",
        help="Decode and print trigger codes without epoching",
    )
    parser.add_argument(
        "--no-reject",
        action="store_true",
        help="Disable peak-to-peak and flat-signal rejection",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--perceived",
        action="store_true",
        help=(
            "Keep only trials rated > 0 (intensity-based perceived filter). "
            "By default all non-rejected trials are kept regardless of rating. "
            "Use this flag for pain-perception-specific analyses."
        ),
    )
    parser.add_argument(
        "--plot-butterfly",
        action="store_true",
        help="Load existing epochs and plot a butterfly (condition average). "
        "Does not re-epoch — reads from derivatives/epochs/.",
    )
    args = parser.parse_args()

    paths = Paths(args.root)
    logger = setup_logging(paths, "epoch")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks = args.tasks if args.tasks else TASKS

    epoch_configs = (
        [args.epoch_config] if args.epoch_config else list(EPOCH_CONFIGS.keys())
    )

    if args.show_triggers:
        logger.info("-- Trigger inspection mode --")
        for label in subjects:
            for task in tasks:
                show_triggers(paths, label, task, logger)
        logger.info(
            "Fill in trigger codes in EPOCH_CONFIGS['triggers'] in core.py, "
            "then re-run without --show-triggers."
        )
        return

    if args.plot_butterfly:
        logger.info("-- Butterfly plot mode --")
        for label in subjects:
            for task in tasks:
                trial_filter = "perceived" if args.perceived else "all"
                plot_butterfly(paths, label, task, epoch_configs[0], logger, trial_filter=trial_filter)
        return

    logger.info("Subjects      : %s", subjects)
    logger.info("Tasks         : %s", tasks)
    logger.info("Epoch configs : %s", epoch_configs)
    logger.info("Rejection     : %s", "disabled" if args.no_reject else "enabled")
    logger.info("Overwrite     : %s", args.overwrite)

    n_ok = n_skip = n_fail = 0
    qc_warnings: list[str] = []
    all_trial_stats: list[dict] = []

    for label in subjects:
        for task in tasks:
            for epoch_config in epoch_configs:
                logger.info("-" * 50)
                try:
                    ok, warn, stats = epoch_one(
                        paths,
                        label,
                        task,
                        epoch_config=epoch_config,
                        reject=not args.no_reject,
                        overwrite=args.overwrite,
                        logger=logger,
                        reject_all_trials=not args.perceived,
                    )
                    if ok:
                        n_ok += 1
                        if stats:
                            all_trial_stats.append(stats)
                    else:
                        n_skip += 1
                    if warn:
                        qc_warnings.append(warn)
                except Exception as e:
                    logger.error(
                        "[sub-%s / %s]  FAILED: %s", label, task, e, exc_info=True
                    )
                    n_fail += 1

    save_trial_stats(paths, all_trial_stats, logger)

    logger.info("=" * 50)
    logger.info("Done.  OK: %d  |  Skipped: %d  |  Failed: %d", n_ok, n_skip, n_fail)

    if qc_warnings:
        logger.warning("-- QC warnings --")
        for w in qc_warnings:
            logger.warning("  %s", w)
        qc_path = paths.log_dir() / "epoch_qc_sweep_counts.txt"
        qc_path.write_text("\n".join(qc_warnings) + "\n", encoding="utf-8")
        logger.warning("QC summary saved: %s", qc_path)

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    print("script started")
    main()
