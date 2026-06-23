#!/usr/bin/env python3
"""
create_bids_structure.py
------------------------
Creates a BIDS-compliant directory structure for an MEG/EEG project.

All project-specific settings (tasks, epoch configs, atlases) are read from
core.py in the same directory.  To adapt for a new project, edit core.py only
— this script requires no changes.

What is created
---------------
rawdata/
    dataset_description.json
    participants.json / participants.tsv
    task-<task>_events.json          (one per task, with placeholder fields)
    sub-XXXX/meg/                    (placeholder .fif + sidecar files)

derivatives/
    freesurfer/sub-XXXX/
    trans/                           (coregistration transforms)
    bads/                            (bad channel lists)
    prep/                            (preprocessed continuous data)
    epochs/                          (epoched data)
    source/sub-XXXX/                 (BEM, src, fwd, inv, STC, ROI)
    contrasts/group/                 (grand-average STCs + ROI stats)
    stats/permtest/                  (cluster permutation test results)
    connectivity/                    (WPLI, PSI)
    logs/                            (pipeline logs + QC plots)

code/
    core.py, preprocess.py, epoch.py, source.py, contrast.py,
    visualize.py, permtest.py, plot_clusters.py, batch.py  (stubs)

Usage
-----
    python create_bids_structure.py
    python create_bids_structure.py --root /path/to/project
    python create_bids_structure.py --root /path/to/project --subjects P01 P02
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Import project configuration from core.py
# ---------------------------------------------------------------------------

# Allow running from any directory by adding code/ to sys.path
_code_dir = Path(__file__).parent
if str(_code_dir) not in sys.path:
    sys.path.insert(0, str(_code_dir))

try:
    from core import (
        TASKS,
        EPOCH_CONFIGS,
        ATLAS_CONFIGS,
        PROJECT_NAME,
    )
except ImportError as e:
    sys.exit(
        f"Cannot import core.py: {e}\n"
        "Make sure core.py is in the same directory as this script."
    )

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ROOT = Path("./bids_project")
DEFAULT_SUBJECTS = ["P01"]  # bare labels — no "sub-" prefix

# ROI names derived from all atlases in core.py
_ALL_ROI_NAMES = sorted({roi for cfg in ATLAS_CONFIGS.values() for roi in cfg["rois"]})

# Contrast names: all task pairs (A-B) from TASKS
_CONTRASTS = [f"{a}-{b}" for i, a in enumerate(TASKS) for b in TASKS[i + 1 :]]

# ---------------------------------------------------------------------------
# Subject ID helper (mirrors core.py — no import dependency needed here)
# ---------------------------------------------------------------------------


def sub_id(label: str) -> str:
    """Return BIDS-prefixed subject ID.  Idempotent."""
    return label if label.startswith("sub-") else f"sub-{label}"


# ---------------------------------------------------------------------------
# BIDS metadata templates
# ---------------------------------------------------------------------------


def _dataset_description() -> dict:
    return {
        "Name": PROJECT_NAME,
        "BIDSVersion": "1.9.0",
        "DatasetType": "raw",
        "Authors": [""],
        "License": "",
        "ReferencesAndLinks": [],
        "DatasetDOI": "",
    }


def _participants_json() -> dict:
    return {
        "participant_id": {
            "Description": "Unique participant identifier, e.g. sub-P01"
        },
        "age": {"Description": "Age in years", "Units": "years"},
        "sex": {
            "Description": "Biological sex",
            "Levels": {"M": "Male", "F": "Female"},
        },
        "handedness": {
            "Description": "Handedness",
            "Levels": {"R": "Right", "L": "Left", "A": "Ambidextrous"},
        },
    }


def _events_json(task: str) -> dict:
    """Generic events sidecar — fill in actual trigger codes and levels."""
    return {
        "onset": {"Description": "Onset of event in seconds"},
        "duration": {"Description": "Duration of event in seconds"},
        "trial_type": {
            "Description": "Type of trial",
            "Levels": {"stimulus": "Placeholder — replace with actual levels"},
        },
        "stim_id": {"Description": "Stimulus identifier"},
        # Add response and accuracy fields if needed:
        # "response":   {"Description": "Participant response"},
        # "correct":    {"Description": "Response accuracy (1=correct, 0=incorrect)"},
    }


# ---------------------------------------------------------------------------
# File system helpers
# ---------------------------------------------------------------------------


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def touch(path: Path) -> None:
    """Create an empty placeholder file (size 0)."""
    path.touch(exist_ok=True)


# ---------------------------------------------------------------------------
# rawdata/
# ---------------------------------------------------------------------------


def create_rawdata(root: Path, subjects: list[str]) -> None:
    raw = root / "rawdata"
    mkdir(raw)

    # BIDS top-level metadata
    write_json(raw / "dataset_description.json", _dataset_description())
    write_json(raw / "participants.json", _participants_json())

    # participants.tsv — single source of truth for subject list
    header = "participant_id\tage\tsex\thandedness\n"
    rows = [f"{sub_id(s)}\tn/a\tn/a\tn/a" for s in subjects]
    write_text(raw / "participants.tsv", header + "\n".join(rows) + "\n")

    # Task-level events sidecars (shared across subjects)
    for task in TASKS:
        write_json(raw / f"task-{task}_events.json", _events_json(task))

    # Per-subject raw data placeholders
    for s in subjects:
        meg_dir = raw / sub_id(s) / "meg"
        mkdir(meg_dir)
        for task in TASKS:
            base = f"{sub_id(s)}_task-{task}"
            touch(meg_dir / f"{base}_meg.fif")  # raw MEG recording
            touch(meg_dir / f"{base}_meg.json")  # acquisition sidecar
            touch(meg_dir / f"{base}_channels.tsv")  # channel metadata
            touch(meg_dir / f"{base}_events.tsv")  # events


# ---------------------------------------------------------------------------
# derivatives/
# ---------------------------------------------------------------------------


def create_derivatives(root: Path, subjects: list[str]) -> None:
    deriv = root / "derivatives"

    for s in subjects:
        sid = sub_id(s)

        # FreeSurfer reconstruction
        mkdir(deriv / "freesurfer" / sid)

        # Coregistration transforms (one per task — covers multi-session case)
        trans_dir = deriv / "trans"
        mkdir(trans_dir)
        for task in TASKS:
            touch(trans_dir / f"{sid}_task-{task}-trans.fif")

        # Bad channels
        bads_dir = deriv / "bads"
        mkdir(bads_dir)
        for task in TASKS:
            write_json(
                bads_dir / f"{sid}_task-{task}_bads.json",
                {"bad_channels": [], "notes": ""},
            )

        # Preprocessed continuous data
        prep_dir = deriv / "prep"
        mkdir(prep_dir)
        for task in TASKS:
            touch(prep_dir / f"{sid}_task-{task}_desc-preproc_meg.fif")

        # Epochs
        epo_dir = deriv / "epochs"
        mkdir(epo_dir)
        for task in TASKS:
            for cfg_name in EPOCH_CONFIGS:
                touch(
                    epo_dir
                    / f"{sid}_task-{task}_desc-{EPOCH_CONFIGS[cfg_name]['desc']}-preproc_epo.fif"
                )

        # Source analysis
        src_base = deriv / "source" / sid
        bem_dir = src_base / "bem"
        mkdir(bem_dir)
        touch(bem_dir / f"{sid}-bem-sol.fif")
        touch(src_base / f"{sid}-src.fif")

        for task in TASKS:
            meg_dir = src_base / f"task-{task}" / "meg"
            mkdir(meg_dir)
            touch(meg_dir / f"{sid}_task-{task}_fwd.fif")
            touch(meg_dir / f"{sid}_task-{task}_cov.fif")
            touch(meg_dir / f"{sid}_task-{task}_inv.fif")
            mkdir(meg_dir / "stc")

            roi_dir = meg_dir / "roi"
            mkdir(roi_dir)
            for roi in _ALL_ROI_NAMES:
                for cfg_name in EPOCH_CONFIGS:
                    touch(roi_dir / f"{sid}_task-{task}_roi-{roi}_{cfg_name}_ave.h5")
            touch(roi_dir / f"{sid}_task-{task}_roi-all_epo.h5")

        # Connectivity
        for task in TASKS:
            mkdir(deriv / "connectivity" / sid / f"task-{task}")

    # Group-level: grand-average STCs, ROI stats, permutation tests
    group_dir = deriv / "contrasts" / "group"
    mkdir(group_dir / "roi_grandavg")
    mkdir(group_dir / "roi_stats")
    mkdir(deriv / "stats" / "permtest")

    # Logs + QC plots
    mkdir(deriv / "logs" / "plots" / "group")


# ---------------------------------------------------------------------------
# code/  — stub scripts + core.py copy
# ---------------------------------------------------------------------------

_SCRIPT_STUBS = {
    "preprocess.py": "# Bandpass filtering + ICA\n",
    "epoch.py": "# Trigger decoding + epoching\n",
    "source.py": "# Forward model, dSPM inverse, ROI extraction\n",
    "contrast.py": "# Grand averages, contrasts, group ROI statistics\n",
    "visualize.py": "# Interactive brain viewer (PyVista/MNE)\n",
    "permtest.py": "# Spatio-temporal cluster permutation tests\n",
    "plot_clusters.py": "# Brain viewer for permtest results\n",
    "batch.py": "# Full pipeline orchestration\n",
}


def create_code(root: Path) -> None:
    code = root / "code"
    mkdir(code)

    for fname, stub in _SCRIPT_STUBS.items():
        fpath = code / fname
        if not fpath.exists():
            write_text(fpath, f"#!/usr/bin/env python3\n{stub}")

    # Copy this script itself into code/
    self_path = Path(__file__)
    dest = code / self_path.name
    if not dest.exists():
        dest.write_bytes(self_path.read_bytes())

    # Remind user to copy core.py
    core_dest = code / "core.py"
    if not core_dest.exists():
        note = (
            "# core.py — copy core_template.py here and fill in your project settings\n"
            "# See: https://github.com/<your-org>/meg-bids-template\n"
        )
        write_text(core_dest, note)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            f"Create a BIDS directory structure for project '{PROJECT_NAME}'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python create_bids_structure.py
              python create_bids_structure.py --root /data/myproject
              python create_bids_structure.py --root /data/myproject --subjects P01 P02 P03
        """),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Project root directory (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=DEFAULT_SUBJECTS,
        metavar="LABEL",
        help=f"Bare subject labels without 'sub-' prefix (default: {DEFAULT_SUBJECTS})",
    )
    args = parser.parse_args()

    root = args.root
    subjects = args.subjects

    print(f"Project        : {PROJECT_NAME}")
    print(f"Root           : {root}")
    print(f"Subjects       : {', '.join(sub_id(s) for s in subjects)}")
    print(f"Tasks          : {', '.join(TASKS)}")
    print(f"Epoch configs  : {', '.join(EPOCH_CONFIGS)}")
    print(f"Atlases        : {', '.join(ATLAS_CONFIGS)}")
    print(f"ROIs           : {', '.join(_ALL_ROI_NAMES)}")
    print()

    root.mkdir(parents=True, exist_ok=True)
    create_rawdata(root, subjects)
    create_derivatives(root, subjects)
    create_code(root)

    print("Done.")
    print()
    print("Next steps:")
    print("  1. Edit code/core.py — set TASKS, EPOCH_CONFIGS, ATLAS_CONFIGS")
    print("  2. Copy raw .fif files into rawdata/<sub>/meg/")
    print("  3. Fill in trigger codes in rawdata/task-*_events.json")
    print("  4. Add participant metadata in rawdata/participants.tsv")
    print("  5. Place trans.fif files in derivatives/trans/")


if __name__ == "__main__":
    main()
