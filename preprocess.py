#!/usr/bin/env python3
"""
preprocess.py
-------------
Preprocessing pipeline for MEG/EEG data.

Steps
-----
1. Load raw .fif file
2. Load bad channels from derivatives/bads/<sub>_task-<task>_bads.json
3. Apply bandpass filter (from FILTER_CONFIGS in core.py)
4. Fit Picard ICA on the broadband gradiometer signal   [opt-in: --ica]
5. Auto-detect and mark ECG artefact components
6. Apply ICA
7. Interpolate bad channels
8. Save preprocessed raw to derivatives/prep/

All settings (filter bands, tasks, subject list) are read from core.py.
The script is safe to re-run: existing outputs are skipped unless --overwrite.

System notes (Neuromag-122)
---------------------------
- 122 planar gradiometers, no magnetometers, no on-board EEG in base config.
- REJECT_FOR_ICA threshold is set for gradiometers (fT/cm).
- ICA is fitted on grad channels only (meg="grad").
- EOG detection via find_bads_eog is available if an EOG channel exists;
  otherwise it is skipped gracefully.

Usage
-----
    # All subjects, no ICA (default)
    python preprocess.py --root /Volumes/ExtremePro/laser

    # Single subject with ICA
    python preprocess.py --root /Volumes/ExtremePro/laser --subjects 4382 --ica

    # Overwrite existing outputs
    python preprocess.py --root /Volumes/ExtremePro/laser --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from core import (
    FILTER_CONFIGS,
    TASKS,
    Paths,
    load_subjects,
    setup_logging,
    sub_id,
)

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")

# ---------------------------------------------------------------------------
# ICA settings  (Neuromag-122 gradiometers)
# ---------------------------------------------------------------------------

ICA_METHOD = "picard"
ICA_N_COMPONENTS = 60  # Neuromag-122 has 122 grads; 60 components is
# a stable default (covers >99% variance)
ICA_MAX_ITER = 500
ICA_RANDOM_STATE = 42

# Peak-to-peak amplitude rejection before ICA fit — gradiometers only
REJECT_FOR_ICA = {
    "grad": 4000e-13,  # 4000 fT/cm
}


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Load raw + bads
# ---------------------------------------------------------------------------


def load_raw(paths: Paths, label: str, task: str, logger) -> mne.io.Raw | None:
    raw_file = paths.raw_meg(label, task)
    if not raw_file.exists():
        logger.warning("[sub-%s / %s]  Raw file not found: %s", label, task, raw_file)
        return None

    logger.info("[sub-%s / %s]  Loading raw: %s", label, task, raw_file.name)
    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)

    # Load bad channels written by inspect_raw.py
    bads_file = paths.bads_file(label, task)
    if bads_file.exists():
        bads = json.loads(bads_file.read_text(encoding="utf-8")).get("bads", [])
        if bads:
            raw.info["bads"] = bads
            logger.info(
                "[sub-%s / %s]  Marking %d bad channel(s) from bads.json: %s",
                label,
                task,
                len(bads),
                bads,
            )
        else:
            logger.info("[sub-%s / %s]  bads.json present but empty", label, task)
    else:
        logger.warning(
            "[sub-%s / %s]  No bads.json found — run inspect_raw.py first", label, task
        )

    # Drop EEG channels — not used in this study
    raw.pick_types(meg=True, eeg=False, stim=True)

    return raw


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------


def filter_raw(
    raw: mne.io.Raw, l_freq: float, h_freq: float, logger, tag: str
) -> mne.io.Raw:
    logger.info("[%s]  Bandpass filter: %.1f – %.1f Hz", tag, l_freq, h_freq)
    return raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        verbose=False,
    )


# ---------------------------------------------------------------------------
# ICA  (Neuromag-122: grad only)
# ---------------------------------------------------------------------------


def fit_ica(raw: mne.io.Raw, logger, tag: str) -> mne.preprocessing.ICA:
    logger.info(
        "[%s]  Fitting ICA (method=%s, n_components=%d, Neuromag-122 grads)",
        tag,
        ICA_METHOD,
        ICA_N_COMPONENTS,
    )
    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        method=ICA_METHOD,
        max_iter=ICA_MAX_ITER,
        random_state=ICA_RANDOM_STATE,
    )
    # High-pass at 1 Hz before ICA fit (standard practice)
    raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)

    # Neuromag-122: fit on gradiometers only, exclude bads
    picks_grad = mne.pick_types(raw_for_ica.info, meg="grad", exclude="bads")
    logger.info("[%s]  ICA picks: %d gradiometers (excl. bads)", tag, len(picks_grad))

    ica.fit(raw_for_ica, picks=picks_grad, reject=REJECT_FOR_ICA, verbose=False)
    logger.info("[%s]  ICA fitted: %d components", tag, ica.n_components_)
    return ica


def detect_artefacts(
    ica: mne.preprocessing.ICA, raw: mne.io.Raw, logger, tag: str
) -> mne.preprocessing.ICA:
    """Auto-detect ECG (and EOG if channel present) artefact components."""
    exclude = set()

    # EOG — only if an EOG channel is present in the recording
    eog_chs = mne.pick_types(raw.info, eog=True)
    if len(eog_chs) > 0:
        try:
            eog_idx, _ = ica.find_bads_eog(raw, verbose=False)
            if eog_idx:
                exclude.update(eog_idx)
                logger.info("[%s]  EOG components: %s", tag, eog_idx)
        except Exception as e:
            logger.warning("[%s]  EOG detection failed: %s", tag, e)
    else:
        logger.info("[%s]  No EOG channel — skipping EOG artefact detection", tag)

    # ECG
    try:
        ecg_idx, _ = ica.find_bads_ecg(raw, verbose=False)
        if ecg_idx:
            exclude.update(ecg_idx)
            logger.info("[%s]  ECG components: %s", tag, ecg_idx)
    except Exception as e:
        logger.warning("[%s]  ECG detection failed: %s", tag, e)

    ica.exclude = sorted(exclude)
    logger.info(
        "[%s]  Excluding %d ICA component(s): %s",
        tag,
        len(ica.exclude),
        ica.exclude,
    )
    return ica


# ---------------------------------------------------------------------------
# ICA inspection
# ---------------------------------------------------------------------------


def inspect_ica(paths: Paths, label: str, task: str, logger) -> None:
    """Plot ICA components, sources and overlay for visual inspection."""
    tag = f"sub-{label} / {task}"

    ica_file = paths.ica_file(label, task)
    if not _exists(ica_file):
        logger.warning("[%s]  No ICA file found: %s", tag, ica_file)
        return

    prep_file = paths.prep_raw(label, task, desc="preproc")
    if not _exists(prep_file):
        logger.warning("[%s]  No preprocessed file found: %s", tag, prep_file)
        return

    logger.info("[%s]  Loading ICA: %s", tag, ica_file.name)
    ica = mne.preprocessing.read_ica(ica_file, verbose=False)
    logger.info("[%s]  Loading preprocessed raw: %s", tag, prep_file.name)
    raw = mne.io.read_raw_fif(prep_file, preload=True, verbose=False)

    logger.info("[%s]  Plotting ICA components (topomaps)", tag)
    ica.plot_components()

    logger.info("[%s]  Plotting ICA sources (time series)", tag)
    ica.plot_sources(raw, show_scrollbars=True)

    logger.info("[%s]  Plotting ICA overlay (before vs after)", tag)
    ica.plot_overlay(raw)

    plt.show()


# ---------------------------------------------------------------------------
# Main per-subject pipeline
# ---------------------------------------------------------------------------


def preprocess_one(
    paths: Paths,
    label: str,
    task: str,
    run_ica: bool,
    overwrite: bool,
    logger,
) -> int:
    """Preprocess one subject/task.  Returns number of output files written."""
    tag = f"sub-{label} / {task}"

    raw = load_raw(paths, label, task, logger)
    if raw is None:
        return 0

    # ── ICA: fit once on broadband signal, reuse for all filter bands ────
    ica = None
    if run_ica:
        ica_file = paths.ica_file(label, task)
        if _exists(ica_file) and not overwrite:
            logger.info("[%s]  Loading existing ICA: %s", tag, ica_file.name)
            ica = mne.preprocessing.read_ica(ica_file, verbose=False)
        else:
            ica = fit_ica(raw, logger, tag)
            ica = detect_artefacts(ica, raw, logger, tag)
            ica_file.parent.mkdir(parents=True, exist_ok=True)
            ica.save(ica_file, overwrite=True, verbose=False)
            logger.info("[%s]  ICA saved: %s", tag, ica_file.name)
    else:
        logger.info("[%s]  ICA disabled (run with --ica to enable)", tag)

    # ── Filter + (optionally) apply ICA for each configured band ─────────
    n_written = 0
    for band_name, filt_cfg in FILTER_CONFIGS.items():
        desc = f"{band_name}-preproc" if band_name != "preproc" else "preproc"
        out_file = paths.prep_raw(label, task, desc=desc)

        if _exists(out_file) and not overwrite:
            logger.info("[%s]  SKIP %s (exists)", tag, out_file.name)
            continue

        raw_band = filter_raw(raw, filt_cfg["l_freq"], filt_cfg["h_freq"], logger, tag)
        notch_freqs = filt_cfg.get("notch")
        if notch_freqs:
            logger.info("[%s]  Notch filter: %s Hz", tag, notch_freqs)
            raw_band.notch_filter(notch_freqs, picks="grad", verbose=False)

        if ica is not None:
            logger.info("[%s]  Applying ICA to %s band", tag, band_name)
            ica.apply(raw_band, verbose=False)

        if raw_band.info["bads"]:
            logger.info(
                "[%s]  Interpolating %d bad channel(s): %s",
                tag,
                len(raw_band.info["bads"]),
                raw_band.info["bads"],
            )
            raw_band.interpolate_bads(verbose=False)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        raw_band.save(out_file, overwrite=True, verbose=False)
        logger.info("[%s]  Saved: %s", tag, out_file.name)
        n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing (filter + optional Picard ICA) for Neuromag-122 MEG."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--tasks", nargs="+", default=None, choices=TASKS)
    parser.add_argument(
        "--ica",
        action="store_true",
        help="Fit and apply Picard ICA (off by default; enable when bads are confirmed)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--inspect-ica",
        action="store_true",
        help="Plot ICA components, sources and overlay for visual inspection. "
        "Does not reprocess — reads existing ICA and preprocessed files.",
    )
    args = parser.parse_args()

    paths = Paths(args.root)
    logger = setup_logging(paths, "preprocess")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks = args.tasks if args.tasks else TASKS

    if args.inspect_ica:
        logger.info("-- ICA inspection mode --")
        for label in subjects:
            for task in tasks:
                inspect_ica(paths, label, task, logger)
        return

    logger.info("Subjects : %s", subjects)
    logger.info("Tasks    : %s", tasks)
    logger.info("ICA      : %s", args.ica)
    logger.info("Overwrite: %s", args.overwrite)

    n_ok = n_fail = 0
    for label in subjects:
        for task in tasks:
            logger.info("─" * 50)
            try:
                n = preprocess_one(
                    paths,
                    label,
                    task,
                    run_ica=args.ica,
                    overwrite=args.overwrite,
                    logger=logger,
                )
                n_ok += n
            except Exception as e:
                logger.error("[sub-%s / %s]  FAILED: %s", label, task, e, exc_info=True)
                n_fail += 1

    logger.info("═" * 50)
    logger.info("Done.  Written: %d  |  Failed: %d", n_ok, n_fail)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
