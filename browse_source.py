#!/usr/bin/env python3
"""
browse_source.py
----------------
Interactive dSPM STC viewer for the laser-pain MEG study.

Loads individual subject STCs (written by source.py) and opens an
interactive MNE Brain viewer (PyVista backend).

The Qt application is created BEFORE any PyVista import — this is the
critical fix for the macOS spinning beach ball.

Usage
-----
    # Single task, default time window
    python browse_source.py --subjects 4382 --tasks laser

    # Restrict time window (e.g. N2 component)
    python browse_source.py --subjects 4382 --tasks laser --tmin 0.0 --tmax 0.5

    # All three tasks sequentially (close each window to open next)
    python browse_source.py --subjects 4382 --tasks laser pinprick tactile --sequential

    # Difference: laser - pinprick
    python browse_source.py --subjects 4382 --diff laser pinprick

    # Manual colormap limits
    python browse_source.py --subjects 4382 --tasks laser --clim 2 10

    # Lateral + medial views
    python browse_source.py --subjects 4382 --tasks laser --views lateral medial
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Qt event loop — MUST be created before any PyVista/pyvistaqt import.
# On macOS a script process has no QApplication by default, which causes
# the spinning beach ball.  Create one here, at module level, before mne.
# ---------------------------------------------------------------------------


def _make_qt_app():
    """Return (or create) a QApplication; returns (app, QtWidgets) or (None, None)."""
    # PyQt6 changed the import style — use direct imports instead of importlib
    # to avoid silent failures on attribute resolution.
    try:
        from PyQt6 import QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        return app, QtWidgets
    except Exception:
        pass
    try:
        from PyQt5 import QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        return app, QtWidgets
    except Exception:
        pass
    try:
        from PySide6 import QtWidgets

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        return app, QtWidgets
    except Exception:
        pass
    return None, None


_QT_APP, _QT_WIDGETS = _make_qt_app()

# Safe to import mne / pyvistaqt only after QApplication exists
import mne

from core import (
    TASKS,
    EPOCH_CONFIGS,
    DEFAULT_EPOCH_CONFIG,
    Paths,
    load_subjects,
    setup_logging,
    sub_id,
)

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")
DEFAULT_CONDITION = "stimulus"
DEFAULT_COLORMAP = "hot"
DIFF_COLORMAP = "RdBu_r"


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Load STC
# ---------------------------------------------------------------------------


def load_stc(
    paths: Paths, label: str, task: str, condition: str, logger
) -> mne.SourceEstimate | None:
    stc_dir = paths.stc_dir(label, task)
    stc_stem = stc_dir / f"{sub_id(label)}_task-{task}_cond-{condition}_desc-dSPM"
    lh_file = Path(str(stc_stem) + "-lh.stc")

    if not _exists(lh_file):
        logger.error(
            "[sub-%s / %s / %s]  STC not found: %s", label, task, condition, lh_file
        )
        return None

    logger.info("[sub-%s / %s / %s]  Loading STC", label, task, condition)
    return mne.read_source_estimate(str(stc_stem), subject=sub_id(label))


# ---------------------------------------------------------------------------
# Colormap limits
# ---------------------------------------------------------------------------


def _auto_clim(stc: mne.SourceEstimate, pct_lo: float = 50, pct_hi: float = 99) -> dict:
    """One-sided clim from percentiles of |data|."""
    vals = np.abs(stc.data)
    lo = float(np.percentile(vals, pct_lo))
    hi = float(np.percentile(vals, pct_hi))
    return dict(kind="value", lims=[lo, (lo + hi) / 2, hi])


def _auto_clim_diverging(stc: mne.SourceEstimate, pct: float = 99) -> dict:
    """Symmetric diverging clim for difference STCs."""
    hi = float(np.percentile(np.abs(stc.data), pct))
    return dict(kind="value", pos_lims=[0, hi / 2, hi])


# ---------------------------------------------------------------------------
# Brain viewer
# ---------------------------------------------------------------------------


def morph_to_fsaverage(
    stc: mne.SourceEstimate,
    label: str,
    subjects_dir: str,
    logger,
) -> mne.SourceEstimate:
    """Morph individual subject STC to fsaverage."""
    logger.info("[sub-%s]  Morphing to fsaverage ...", label)
    morph = mne.compute_source_morph(
        stc,
        subject_from=sub_id(label),
        subject_to="fsaverage",
        subjects_dir=subjects_dir,
        verbose=False,
    )
    return morph.apply(stc)


def open_brain(
    stc: mne.SourceEstimate,
    subjects_dir: str,
    title: str,
    views: list[str],
    colormap: str,
    clim: dict,
    tmin: float | None,
    tmax: float | None,
) -> mne.viz.Brain:
    """Open an interactive PyVista Brain window on fsaverage."""
    if tmin is not None or tmax is not None:
        t0 = tmin if tmin is not None else stc.tmin
        t1 = tmax if tmax is not None else stc.times[-1]
        stc = stc.copy().crop(tmin=t0, tmax=t1)

    brain = stc.plot(
        subject="fsaverage",
        subjects_dir=subjects_dir,
        hemi="both",
        views=views,
        colormap=colormap,
        clim=clim,
        time_viewer=True,
        show_traces=True,
        time_label="%.0f ms",
        title=title,
        backend="pyvistaqt",
        size=(1200 * len(views), 600),
        verbose=False,
    )
    return brain


# ---------------------------------------------------------------------------
# Qt event loop helpers
# ---------------------------------------------------------------------------


def _qt_exec():
    """Start the Qt event loop; blocks until all windows are closed."""
    if _QT_APP is None:
        import time

        print("Qt app not found — press Ctrl-C to exit", file=sys.stderr)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        return
    exec_fn = getattr(_QT_APP, "exec", None) or getattr(_QT_APP, "exec_", None)
    if exec_fn:
        exec_fn()


def _qt_exec_until_closed(brain):
    """Block until a single Brain window is closed."""
    try:
        brain.wait_for_close()
    except Exception:
        _qt_exec()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Interactive dSPM STC viewer for the laser-pain MEG study."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        choices=TASKS,
        help="Task(s) to display (default: laser)",
    )
    parser.add_argument(
        "--condition",
        default=DEFAULT_CONDITION,
        help=f"Condition name in event_id (default: {DEFAULT_CONDITION})",
    )
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("TASK_A", "TASK_B"),
        help="Show difference STC: TASK_A minus TASK_B",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Start of time window (s)",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=None,
        help="End of time window (s)",
    )
    parser.add_argument(
        "--clim",
        nargs=2,
        type=float,
        default=None,
        metavar=("LO", "HI"),
        help="Manual colormap limits (default: auto from percentiles)",
    )
    parser.add_argument(
        "--views",
        nargs="+",
        default=["lateral"],
        choices=["lateral", "medial", "dorsal", "ventral", "frontal", "caudal"],
        help="Brain views (default: lateral)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Open one window at a time — close each to proceed",
    )
    args = parser.parse_args()

    paths = Paths(args.root)
    logger = setup_logging(paths, "browse_source")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks = args.tasks if args.tasks else [TASKS[0]]
    subjects_dir = str(paths.freesurfer_dir())

    brains = []  # keep references alive — GC would close windows

    for label in subjects:
        # ── Difference mode ──────────────────────────────────────────────
        if args.diff:
            task_a, task_b = args.diff
            stc_a = load_stc(paths, label, task_a, args.condition, logger)
            stc_b = load_stc(paths, label, task_b, args.condition, logger)
            if stc_a is None or stc_b is None:
                continue

            stc = stc_a - stc_b
            clim = (
                dict(
                    kind="value",
                    pos_lims=[args.clim[0], sum(args.clim) / 2, args.clim[1]],
                )
                if args.clim
                else _auto_clim_diverging(stc)
            )
            title = f"sub-{label}  |  {task_a} − {task_b}  |  {args.condition}"
            logger.info("Opening: %s", title)
            stc = morph_to_fsaverage(stc, label, subjects_dir, logger)
            brain = open_brain(
                stc,
                subjects_dir,
                title,
                args.views,
                DIFF_COLORMAP,
                clim,
                args.tmin,
                args.tmax,
            )
            brains.append(brain)
            if args.sequential:
                _qt_exec_until_closed(brain)

        # ── Single-task mode ─────────────────────────────────────────────
        else:
            for task in tasks:
                stc = load_stc(paths, label, task, args.condition, logger)
                if stc is None:
                    continue

                clim = (
                    dict(
                        kind="value",
                        lims=[args.clim[0], sum(args.clim) / 2, args.clim[1]],
                    )
                    if args.clim
                    else _auto_clim(stc)
                )
                title = f"sub-{label}  |  {task}  |  {args.condition}  |  dSPM"
                logger.info("Opening: %s", title)
                stc = morph_to_fsaverage(stc, label, subjects_dir, logger)
                brain = open_brain(
                    stc,
                    subjects_dir,
                    title,
                    args.views,
                    DEFAULT_COLORMAP,
                    clim,
                    args.tmin,
                    args.tmax,
                )
                brains.append(brain)
                if args.sequential:
                    logger.info("Close window to continue ...")
                    _qt_exec_until_closed(brain)

    if not args.sequential and brains:
        logger.info("All %d window(s) open — close all to exit", len(brains))
        _qt_exec()

    logger.info("Done.")


if __name__ == "__main__":
    main()
