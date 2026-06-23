#!/usr/bin/env python3
"""
source.py
---------
Source reconstruction pipeline for the laser-pain MEG study.

Two modes
---------
1. --setup-bem   (run once per subject before the main pipeline)
   - Copies watershed BEM surfaces from derivatives/freesurfer/ to
     derivatives/source/<sub>/bem/
   - Computes single-shell BEM solution (inner skull only; standard for MEG)
   - Computes source space (oct6, ~4000 vertices per hemisphere)

2. Main mode     (run after preprocess.py + epoch.py)
   - Loads epochs, computes noise covariance from baseline
   - Loads trans, BEM solution and source space
   - Computes forward solution
   - Computes inverse operator (dSPM, loose=0.2, depth=0.8)
   - Applies inverse to evoked (per condition) → saves STC files

Directory layout written
------------------------
derivatives/source/sub-<label>/
    bem/
        sub-<label>-bem-surfaces.fif   ← watershed surfaces (copied)
        sub-<label>-5120-bem-sol.fif   ← single-shell BEM solution
        sub-<label>-src.fif            ← oct6 source space
    task-<task>/meg/
        sub-<label>_task-<task>_fwd.fif
        sub-<label>_task-<task>_cov.fif
        sub-<label>_task-<task>_inv.fif
        stc/
            sub-<label>_task-<task>_cond-<condition>_desc-dSPM-lh.stc
            sub-<label>_task-<task>_cond-<condition>_desc-dSPM-rh.stc

Usage
-----
    # Step 1: BEM + source space (once per subject)
    python source.py --subjects 4382 --setup-bem

    # Step 2: Forward + inverse + dSPM for all tasks
    python source.py --subjects 4382

    # Overwrite existing outputs
    python source.py --subjects 4382 --overwrite

    # All subjects
    python source.py --setup-bem
    python source.py
"""

import argparse
import sys
from pathlib import Path

import mne
import numpy as np

from core import (
    DEFAULT_EPOCH_CONFIG,
    EPOCH_CONFIGS,
    TASKS,
    Paths,
    load_subjects,
    setup_logging,
    sub_id,
)

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")
FREESURFER_SUBJECT = "fsaverage"  # used only as fallback; actual subject dir
# is derived from sub_id(label)

# Source space resolution
SRC_SPACING = "oct6"  # ~4.9 mm spacing, ~4000 vertices/hemisphere

# dSPM inverse parameters
LOOSE = 0.2  # orientation constraint (0=fixed, 1=free)
DEPTH = 0.8  # depth weighting
SNR = 3.0  # assumed SNR for regularisation
LAMBDA2 = 1.0 / SNR**2


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------------
# Step 1a: Copy watershed BEM surfaces
# ---------------------------------------------------------------------------


def _find_fs_bem_file(fs_bem_dir, label, suffix):
    """Locate a FreeSurfer BEM .fif regardless of ico level or sub- prefix.

    Skips macOS dot-file artefacts (._filename) that glob would otherwise
    pick up before the real file.
    """
    for stem in (label, sub_id(label)):
        p = fs_bem_dir / f"{stem}-{suffix}.fif"
        if p.exists() and not p.name.startswith("."):
            return p
    candidates = [
        p
        for p in sorted(fs_bem_dir.glob(f"*-{suffix}.fif"))
        if not p.name.startswith(".")
    ]
    return candidates[0] if candidates else None


def copy_bem_surfaces(paths, label, overwrite, logger):
    """Copy BEM surfaces .fif from FreeSurfer bem/ to derivatives/source/<sub>/bem/."""
    import shutil

    tag = f"sub-{label}"
    dest_file = paths.bem_dir(label) / f"{sub_id(label)}-bem-surfaces.fif"

    if _exists(dest_file) and not overwrite:
        logger.info("[%s]  BEM surfaces already copied, skipping", tag)
        return dest_file

    fs_bem_dir = paths.freesurfer_dir() / sub_id(label) / "bem"
    src_file = _find_fs_bem_file(fs_bem_dir, label, suffix="bem")
    if src_file is None:
        src_file = _find_fs_bem_file(fs_bem_dir, label, suffix="head")
    if src_file is None:
        logger.error("[%s]  No BEM surfaces .fif found in %s", tag, fs_bem_dir)
        return None

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dest_file)
    logger.info(
        "[%s]  BEM surfaces copied: %s -> %s", tag, src_file.name, dest_file.name
    )
    return dest_file


# ---------------------------------------------------------------------------
# Step 1b: Copy BEM solution (already computed by FreeSurfer/MNE)
# ---------------------------------------------------------------------------


def compute_bem_solution(paths, label, overwrite, logger):
    """Compute single-shell BEM solution using MNE from the watershed surfaces.

    Even though a BEM solution may already exist in the FreeSurfer bem/ folder,
    it may have been created by a different tool or MNE version and thus fail
    the FIF file-id check in make_forward_solution.  Recomputing with the
    current MNE version guarantees a readable output.

    Uses ico=5 (20480 triangles) to match the existing watershed surface.
    """
    tag = f"sub-{label}"
    sol_file = paths.bem_sol(label)

    if _exists(sol_file) and not overwrite:
        logger.info("[%s]  BEM solution exists, skipping", tag)
        return True

    logger.info("[%s]  Computing single-shell BEM solution (ico=5) ...", tag)
    try:
        bem_model = mne.make_bem_model(
            subject=sub_id(label),
            ico=5,  # 20480 triangles — matches watershed output
            conductivity=(0.3,),  # single shell: inner skull only
            subjects_dir=str(paths.freesurfer_dir()),
            verbose=False,
        )
        bem_sol = mne.make_bem_solution(bem_model, verbose=False)
        sol_file.parent.mkdir(parents=True, exist_ok=True)
        mne.write_bem_solution(str(sol_file), bem_sol, overwrite=True, verbose=False)
        logger.info("[%s]  BEM solution written: %s", tag, sol_file.name)
        return True
    except Exception as e:
        logger.error("[%s]  BEM solution failed: %s", tag, e)
        return False


# ---------------------------------------------------------------------------
# Step 1c: Compute source space
# ---------------------------------------------------------------------------


def compute_source_space(paths: Paths, label: str, overwrite: bool, logger) -> bool:
    """Compute oct6 cortical source space."""
    tag = f"sub-{label}"
    src_file = paths.src(label)

    if _exists(src_file) and not overwrite:
        logger.info("[%s]  Source space exists, skipping", tag)
        return True

    logger.info("[%s]  Computing source space (%s) ...", tag, SRC_SPACING)
    try:
        src = mne.setup_source_space(
            subject=sub_id(label),
            spacing=SRC_SPACING,
            subjects_dir=str(paths.freesurfer_dir()),
            add_dist=False,
            verbose=False,
        )
        src_file.parent.mkdir(parents=True, exist_ok=True)
        src.save(str(src_file), overwrite=True, verbose=False)
        logger.info(
            "[%s]  Source space written: %s  (%d + %d vertices)",
            tag,
            src_file.name,
            src[0]["nuse"],
            src[1]["nuse"],
        )
        return True
    except Exception as e:
        logger.error("[%s]  Source space computation failed: %s", tag, e)
        return False


# ---------------------------------------------------------------------------
# Step 1: BEM setup entry point (--setup-bem)
# ---------------------------------------------------------------------------


def setup_bem(paths: Paths, label: str, overwrite: bool, logger) -> bool:
    """Run all three BEM setup steps for one subject."""
    surf_ok = copy_bem_surfaces(paths, label, overwrite, logger) is not None
    if not surf_ok:
        return False
    bem_ok = compute_bem_solution(paths, label, overwrite, logger)
    src_ok = compute_source_space(paths, label, overwrite, logger)
    return bem_ok and src_ok


# ---------------------------------------------------------------------------
# Step 2a: Noise covariance from epoch baseline
# ---------------------------------------------------------------------------


def compute_noise_cov(
    paths: Paths, label: str, task: str, epoch_config: str, overwrite: bool, logger
) -> mne.Covariance | None:
    """Compute noise covariance from the pre-stimulus baseline of the epochs."""
    tag = f"sub-{label} / {task}"
    cov_file = paths.noise_cov(label, task)

    if _exists(cov_file) and not overwrite:
        logger.info("[%s]  Loading existing noise covariance", tag)
        return mne.read_cov(str(cov_file), verbose=False)

    cfg = EPOCH_CONFIGS[epoch_config]
    epo_file = paths.epochs(label, task, desc=f"{cfg['desc']}-preproc")

    if not _exists(epo_file):
        logger.warning("[%s]  Epochs file not found: %s", tag, epo_file.name)
        return None

    logger.info("[%s]  Computing noise covariance from baseline ...", tag)
    epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)

    tmin_bl, tmax_bl = cfg.get("baseline", (-0.1, 0))
    try:
        cov = mne.compute_covariance(
            epochs,
            tmin=tmin_bl,
            tmax=tmax_bl,
            method="shrunk",  # robust default; alternative: "empirical"
            rank=None,
            verbose=False,
        )
        cov_file.parent.mkdir(parents=True, exist_ok=True)
        cov.save(str(cov_file), overwrite=True, verbose=False)
        logger.info("[%s]  Noise covariance saved: %s", tag, cov_file.name)
        return cov
    except Exception as e:
        logger.error("[%s]  Noise covariance failed: %s", tag, e)
        return None


# ---------------------------------------------------------------------------
# Step 2b: Forward solution
# ---------------------------------------------------------------------------


def compute_forward(
    paths: Paths, label: str, task: str, overwrite: bool, logger
) -> mne.Forward | None:
    """Compute forward solution using trans, BEM, and source space."""
    tag = f"sub-{label} / {task}"
    fwd_file = paths.fwd(label, task)

    if _exists(fwd_file) and not overwrite:
        logger.info("[%s]  Loading existing forward solution", tag)
        return mne.read_forward_solution(str(fwd_file), verbose=False)

    trans_file = paths.trans(label, task)
    bem_file = paths.bem_sol(label)
    src_file = paths.src(label)
    info_file = paths.prep_raw(label, task, desc="preproc")

    for f, name in [
        (trans_file, "trans"),
        (bem_file, "BEM solution"),
        (src_file, "source space"),
        (info_file, "preprocessed raw"),
    ]:
        if not _exists(f):
            logger.warning("[%s]  %s not found: %s", tag, name, f.name)
            return None

    logger.info("[%s]  Computing forward solution ...", tag)
    try:
        raw_info = mne.io.read_info(str(info_file), verbose=False)
        src = mne.read_source_spaces(str(src_file), verbose=False)
        fwd = mne.make_forward_solution(
            raw_info,
            trans=str(trans_file),
            src=src,
            bem=str(bem_file),
            meg=True,
            eeg=False,
            mindist=5.0,  # mm — exclude sources too close to inner skull
            verbose=False,
        )
        fwd_file.parent.mkdir(parents=True, exist_ok=True)
        mne.write_forward_solution(str(fwd_file), fwd, overwrite=True, verbose=False)
        logger.info(
            "[%s]  Forward solution saved: %s  (%d sources)",
            tag,
            fwd_file.name,
            fwd["nsource"],
        )
        return fwd
    except Exception as e:
        logger.error("[%s]  Forward solution failed: %s", tag, e)
        return None


# ---------------------------------------------------------------------------
# Step 2c: Inverse operator
# ---------------------------------------------------------------------------


def compute_inverse(
    paths: Paths,
    label: str,
    task: str,
    fwd: mne.Forward,
    cov: mne.Covariance,
    overwrite: bool,
    logger,
) -> mne.minimum_norm.InverseOperator | None:
    """Compute MNE inverse operator."""
    tag = f"sub-{label} / {task}"
    inv_file = paths.inv(label, task)

    if _exists(inv_file) and not overwrite:
        logger.info("[%s]  Loading existing inverse operator", tag)
        return mne.minimum_norm.read_inverse_operator(str(inv_file), verbose=False)

    # Need info from preprocessed raw
    info_file = paths.prep_raw(label, task, desc="preproc")
    if not _exists(info_file):
        logger.warning("[%s]  Preprocessed raw not found for inverse", tag)
        return None

    logger.info(
        "[%s]  Computing inverse operator (loose=%.1f, depth=%.1f) ...",
        tag,
        LOOSE,
        DEPTH,
    )
    try:
        raw_info = mne.io.read_info(str(info_file), verbose=False)
        inv = mne.minimum_norm.make_inverse_operator(
            raw_info,
            fwd,
            cov,
            loose=LOOSE,
            depth=DEPTH,
            verbose=False,
        )
        mne.minimum_norm.write_inverse_operator(
            str(inv_file), inv, overwrite=True, verbose=False
        )
        logger.info("[%s]  Inverse operator saved: %s", tag, inv_file.name)
        return inv
    except Exception as e:
        logger.error("[%s]  Inverse operator failed: %s", tag, e)
        return None


# ---------------------------------------------------------------------------
# Step 2d: Apply inverse → dSPM STCs per condition
# ---------------------------------------------------------------------------


def apply_inverse_epochs(
    paths: Paths,
    label: str,
    task: str,
    inv: mne.minimum_norm.InverseOperator,
    epoch_config: str,
    overwrite: bool,
    logger,
) -> int:
    """Apply dSPM inverse to each condition's evoked and save STC files.

    Returns the number of conditions successfully written.
    """
    tag = f"sub-{label} / {task}"
    cfg = EPOCH_CONFIGS[epoch_config]
    epo_file = paths.epochs(label, task, desc=f"{cfg['desc']}-preproc")
    stc_dir = paths.stc_dir(label, task)

    if not _exists(epo_file):
        logger.warning("[%s]  Epochs not found: %s", tag, epo_file.name)
        return 0

    epochs = mne.read_epochs(str(epo_file), preload=True, verbose=False)
    conditions = list(epochs.event_id.keys())
    logger.info(
        "[%s]  Applying dSPM inverse to %d condition(s): %s",
        tag,
        len(conditions),
        conditions,
    )

    stc_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    for cond in conditions:
        stc_stem = stc_dir / f"{sub_id(label)}_task-{task}_cond-{cond}_desc-dSPM"
        # STC files are written as <stem>-lh.stc and <stem>-rh.stc by MNE
        lh_file = Path(str(stc_stem) + "-lh.stc")
        if _exists(lh_file) and not overwrite:
            logger.info("[%s]  SKIP STC %s (exists)", tag, cond)
            n_written += 1
            continue

        try:
            evoked = epochs[cond].average()
            stc = mne.minimum_norm.apply_inverse(
                evoked,
                inv,
                lambda2=LAMBDA2,
                method="dSPM",
                pick_ori=None,
                verbose=False,
            )
            stc.save(str(stc_stem), overwrite=True, verbose=False)
            logger.info(
                "[%s]  STC saved: %s  (%.1f – %.1f s, %d vertices)",
                tag,
                stc_stem.name,
                stc.tmin,
                stc.tmin + stc.tstep * stc.data.shape[1],
                stc.data.shape[0],
            )
            n_written += 1
        except Exception as e:
            logger.error("[%s]  dSPM for condition %s failed: %s", tag, cond, e)

    return n_written


# ---------------------------------------------------------------------------
# Main per-subject / per-task pipeline
# ---------------------------------------------------------------------------


def source_one(
    paths: Paths,
    label: str,
    task: str,
    epoch_config: str,
    overwrite: bool,
    logger,
) -> bool:
    """Run full source pipeline for one subject/task.  Returns True on success."""
    tag = f"sub-{label} / {task}"
    logger.info("[%s]  ── Source reconstruction ──", tag)

    cov = compute_noise_cov(paths, label, task, epoch_config, overwrite, logger)
    if cov is None:
        return False

    fwd = compute_forward(paths, label, task, overwrite, logger)
    if fwd is None:
        return False

    inv = compute_inverse(paths, label, task, fwd, cov, overwrite, logger)
    if inv is None:
        return False

    n = apply_inverse_epochs(paths, label, task, inv, epoch_config, overwrite, logger)
    return n > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Source reconstruction for the laser-pain MEG study."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--tasks", nargs="+", default=None, choices=TASKS)
    parser.add_argument(
        "--epoch-config",
        default=DEFAULT_EPOCH_CONFIG,
        choices=list(EPOCH_CONFIGS.keys()),
    )
    parser.add_argument(
        "--setup-bem",
        action="store_true",
        help=(
            "Copy watershed BEM surfaces, compute BEM solution and source "
            "space. Run once per subject before the main pipeline."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = Paths(args.root)
    logger = setup_logging(paths, "source")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks = args.tasks if args.tasks else TASKS

    logger.info("Subjects     : %s", subjects)
    logger.info("Tasks        : %s", tasks)
    logger.info("Epoch config : %s", args.epoch_config)
    logger.info("Overwrite    : %s", args.overwrite)

    # ── BEM setup mode ───────────────────────────────────────────────────
    if args.setup_bem:
        logger.info("── BEM setup mode ───────────────────────────────")
        n_ok = n_fail = 0
        for label in subjects:
            ok = setup_bem(paths, label, args.overwrite, logger)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
        logger.info("BEM setup done.  OK: %d  |  Failed: %d", n_ok, n_fail)
        if n_fail:
            sys.exit(1)
        return

    # ── Main source pipeline ─────────────────────────────────────────────
    n_ok = n_fail = 0
    for label in subjects:
        for task in tasks:
            logger.info("─" * 50)
            try:
                ok = source_one(
                    paths,
                    label,
                    task,
                    epoch_config=args.epoch_config,
                    overwrite=args.overwrite,
                    logger=logger,
                )
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                logger.error("[sub-%s / %s]  FAILED: %s", label, task, e, exc_info=True)
                n_fail += 1

    logger.info("═" * 50)
    logger.info("Done.  OK: %d  |  Failed: %d", n_ok, n_fail)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
