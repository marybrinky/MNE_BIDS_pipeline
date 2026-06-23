#!/usr/bin/env python3
"""
inspect_raw.py
--------------
First-pass quality control for raw MEG/EEG .fif files.

Prints a summary report for each subject/task and optionally opens
the MNE interactive raw browser for visual inspection.

What is checked
---------------
- File exists and can be opened
- Recording duration and sampling rate
- Channel types and count
- Trigger codes and event counts (vs. expected from core.py)
- Existing bad channels in the file and/or JSON sidecar
- Signal range (min/max, std) per channel type — flags outliers
- PowerLine noise estimate (50/60 Hz)

After inspection / browsing
---------------------------
- Bad channels are written to derivatives/bads/<sub>_task-<task>_bads.json
  (via paths.bads_file).
- If a file already exists, old and new bads are merged, so manual edits are
  not overwritten.

Usage
-----
    python code/inspect_raw.py --root /path/to/project
    python code/inspect_raw.py --root /path/to/project --subjects 4382 --browse
    python code/inspect_raw.py --root /path/to/project --tasks laser --browse
    python code/inspect_raw.py --root /path/to/project --report
"""

import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np

from core import (
    TASKS,
    EPOCH_CONFIGS,
    DEFAULT_EPOCH_CONFIG,
    Paths,
    load_subjects,
    setup_logging,
)

DEFAULT_ROOT = Path("./bids_project")

# Expected powerline frequency (Hz) — change to 60 for US data
POWERLINE_FREQ = 50


def configure_browser_backend() -> str:
    """Try to use the Qt browser, fall back to matplotlib."""
    try:
        mne.viz.set_browser_backend("qt")
        return "qt"
    except Exception:
        try:
            mne.viz.set_browser_backend("matplotlib")
            return "matplotlib"
        except Exception:
            return "unknown"


# ---------------------------------------------------------------------------
# Bads persistence
# ---------------------------------------------------------------------------


def _normalize_bads(bads) -> list[str]:
    """Return a sorted unique list of channel names."""
    if bads is None:
        return []
    return sorted({str(ch) for ch in bads if str(ch).strip()})


def read_bads_json(paths: Paths, label: str, task: str, logger) -> list[str]:
    """Read bad channels from derivatives/bads/<sub>_task-<task>_bads.json."""
    bads_file = paths.bads_file(label, task)

    if not bads_file.exists():
        return []

    try:
        with open(bads_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        logger.warning(
            "[sub-%s / %s]  Could not read bads JSON %s: %s",
            label,
            task,
            bads_file,
            exc,
        )
        return []

    if isinstance(payload, dict):
        bads = payload.get("bads", [])
    elif isinstance(payload, list):
        # tolerate legacy format: plain list in JSON
        bads = payload
    else:
        logger.warning(
            "[sub-%s / %s]  Invalid bads JSON format in %s",
            label,
            task,
            bads_file,
        )
        return []

    bads = _normalize_bads(bads)
    if bads:
        logger.info(
            "[sub-%s / %s]  Loaded %d bad channel(s) from JSON: %s",
            label,
            task,
            len(bads),
            bads,
        )
    return bads


def write_bads_json(paths: Paths, label: str, task: str, bads: list[str], logger) -> Path | None:
    """Write merged bad channels to derivatives/bads/<sub>_task-<task>_bads.json."""
    bads_file = paths.bads_file(label, task)
    bads_file.parent.mkdir(parents=True, exist_ok=True)

    old_bads = read_bads_json(paths, label, task, logger)
    merged = _normalize_bads(list(old_bads) + list(bads))

    payload = {
        "subject": str(label),
        "task": str(task),
        "bads": merged,
    }

    try:
        with open(bads_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception as exc:
        logger.warning(
            "[sub-%s / %s]  Could not write bads JSON %s: %s",
            label,
            task,
            bads_file,
            exc,
        )
        return None

    logger.info(
        "[sub-%s / %s]  Saved %d bad channel(s) to %s",
        label,
        task,
        len(merged),
        bads_file,
    )
    return bads_file


def merge_bads(raw, json_bads: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Merge bads from the FIF info and the JSON file."""
    bads_in_file = _normalize_bads(raw.info.get("bads", []))
    bads_in_json = _normalize_bads(json_bads)
    merged = _normalize_bads(bads_in_file + bads_in_json)
    raw.info["bads"] = merged
    return bads_in_file, bads_in_json, merged


# ---------------------------------------------------------------------------
# Single-file inspection
# ---------------------------------------------------------------------------


def inspect_raw(
    paths: Paths,
    label: str,
    task: str,
    epoch_config: str,
    logger,
) -> dict:
    """Load and inspect one raw file. Returns a summary dict."""
    tag = f"sub-{label} / {task}"
    raw_file = paths.raw_meg(label, task)

    summary = {
        "subject": label,
        "task": task,
        "file": str(raw_file),
        "ok": False,
        "warnings": [],
    }

    if not raw_file.exists():
        summary["warnings"].append("FILE NOT FOUND")
        logger.warning("[%s]  ✗  File not found: %s", tag, raw_file)
        return summary

    try:
        raw = mne.io.read_raw_fif(raw_file, preload=False, verbose=False)
    except Exception as e:
        summary["warnings"].append(f"Cannot open file: {e}")
        logger.warning("[%s]  ✗  Cannot open: %s", tag, e)
        return summary

    bads_in_json = read_bads_json(paths, label, task, logger)
    bads_in_file, _, merged_bads = merge_bads(raw, bads_in_json)

    info = raw.info
    duration_s = raw.times[-1]
    sfreq = info["sfreq"]
    n_channels = len(info["chs"])
    ch_types = raw.get_channel_types()
    type_counts = {}
    for ct in ch_types:
        type_counts[ct] = type_counts.get(ct, 0) + 1

    summary.update(
        {
            "duration_s": round(duration_s, 1),
            "sfreq": sfreq,
            "n_channels": n_channels,
            "ch_types": type_counts,
            "bads_in_file": bads_in_file,
            "bads_in_json": bads_in_json,
            "bads_merged": merged_bads,
        }
    )

    logger.info(
        "[%s]  Duration: %.1f s  |  sfreq: %.0f Hz  |  Channels: %d",
        tag,
        duration_s,
        sfreq,
        n_channels,
    )
    logger.info("[%s]  Channel types: %s", tag, type_counts)

    if bads_in_file:
        logger.warning("[%s]  Bad channels in FIF file: %s", tag, bads_in_file)
    if bads_in_json:
        logger.warning("[%s]  Bad channels in JSON: %s", tag, bads_in_json)
    if merged_bads:
        summary["warnings"].append(f"Known bad channels: {merged_bads}")

    # Trigger codes
    try:
        events = mne.find_events(raw, verbose=False)
        unique, counts = np.unique(events[:, 2], return_counts=True)
        event_summary = {int(k): int(v) for k, v in zip(unique, counts)}
        summary["events"] = event_summary
        logger.info("[%s]  Events: %s", tag, event_summary)

        cfg = EPOCH_CONFIGS[epoch_config]
        event_id = cfg.get("event_id", {})
        expected_n = cfg.get("n_expected", None)

        for name, code in event_id.items():
            n_found = event_summary.get(code, 0)
            if n_found == 0:
                msg = f"Trigger code {code} ({name}) NOT FOUND in data"
                summary["warnings"].append(msg)
                logger.warning("[%s]  ✗  %s", tag, msg)
            else:
                logger.info("[%s]  ✓  Trigger %d (%s): %d events", tag, code, name, n_found)
                if expected_n and n_found < 0.8 * expected_n:
                    msg = (
                        f"Only {n_found}/{expected_n} sweeps "
                        f"({100 * n_found / expected_n:.0f}%) for trigger {code}"
                    )
                    summary["warnings"].append(msg)
                    logger.warning("[%s]  ⚠  %s", tag, msg)

    except Exception as e:
        summary["warnings"].append(f"Event detection failed: {e}")
        logger.warning("[%s]  Event detection failed: %s", tag, e)

    # Signal quality
    try:
        raw.load_data(verbose=False)

        for ch_type in ("grad", "mag", "eeg"):
            picks = mne.pick_types(
                info,
                meg=(ch_type in ("grad", "mag")),
                eeg=(ch_type == "eeg"),
                exclude="bads",
            )
            if len(picks) == 0:
                continue

            data = raw.get_data(picks=picks)
            data_std = np.std(data, axis=1)
            data_max = np.max(np.abs(data), axis=1)

            median_std = np.median(data_std)
            noisy = np.where(data_std > 5 * median_std)[0]
            flat = np.where(data_std < 0.01 * median_std)[0]

            if len(noisy):
                noisy_names = [info["ch_names"][picks[i]] for i in noisy]
                msg = f"Noisy {ch_type} channels (std >5× median): {noisy_names}"
                summary["warnings"].append(msg)
                logger.warning("[%s]  ⚠  %s", tag, msg)

            if len(flat):
                flat_names = [info["ch_names"][picks[i]] for i in flat]
                msg = f"Flat {ch_type} channels (std <1%% median): {flat_names}"
                summary["warnings"].append(msg)
                logger.warning("[%s]  ⚠  %s", tag, msg)

            unit = "pT/cm" if ch_type == "grad" else "pT" if ch_type == "mag" else "µV"
            scale = 1e13 if ch_type == "grad" else 1e12 if ch_type == "mag" else 1e6
            logger.info(
                "[%s]  %s: median std=%.2f %s, max=%.2f %s  (%d noisy, %d flat)",
                tag,
                ch_type,
                median_std * scale,
                unit,
                np.max(data_max) * scale,
                unit,
                len(noisy),
                len(flat),
            )

        spectrum = raw.compute_psd(fmin=45, fmax=55, verbose=False)
        freqs = spectrum.freqs
        pl_idx = np.argmin(np.abs(freqs - POWERLINE_FREQ))
        pl_power = spectrum.get_data()[:, pl_idx].mean()
        logger.info("[%s]  PowerLine (%d Hz) mean power: %.2e", tag, POWERLINE_FREQ, pl_power)

    except Exception as e:
        logger.warning("[%s]  Signal quality check failed: %s", tag, e)

    summary["ok"] = len(summary["warnings"]) == 0

    if summary["ok"]:
        logger.info("[%s]  ✓  No issues found", tag)
    else:
        logger.warning("[%s]  ⚠  %d warning(s):", tag, len(summary["warnings"]))
        for w in summary["warnings"]:
            logger.warning("[%s]    - %s", tag, w)

    return summary


# ---------------------------------------------------------------------------
# Interactive browser
# ---------------------------------------------------------------------------


def browse_raw(paths: Paths, label: str, task: str, logger) -> None:
    """Open MNE interactive raw browser for one subject/task."""
    import importlib

    for mod_name in (
        "PyQt6.QtWidgets",
        "PySide6.QtWidgets",
        "PyQt5.QtWidgets",
        "PySide2.QtWidgets",
    ):
        try:
            Qt = importlib.import_module(mod_name)
            app = Qt.QApplication.instance()
            if app is None:
                Qt.QApplication(sys.argv)
            break
        except Exception:
            continue

    raw_file = paths.raw_meg(label, task)
    if not raw_file.exists():
        logger.error("[sub-%s / %s]  File not found for browsing", label, task)
        return

    logger.info("[sub-%s / %s]  Opening raw browser ...", label, task)
    raw = mne.io.read_raw_fif(raw_file, preload=True, verbose=False)

    bads_in_json = read_bads_json(paths, label, task, logger)
    bads_in_file, _, merged_bads = merge_bads(raw, bads_in_json)
    if merged_bads:
        logger.info(
            "[sub-%s / %s]  Starting browser with %d known bad channel(s): %s",
            label,
            task,
            len(merged_bads),
            merged_bads,
        )

    try:
        events = mne.find_events(raw, verbose=False)
        annotations = mne.annotations_from_events(
            events,
            sfreq=raw.info["sfreq"],
            verbose=False,
        )
        raw.set_annotations(annotations)
    except Exception:
        pass

    raw.plot(
        duration=10,
        n_channels=30,
        scalings="auto",
        title=f"sub-{label} / {task} — raw inspection",
        show=True,
        block=True,
        verbose=False,
    )

    final_bads = _normalize_bads(raw.info.get("bads", []))
    if final_bads != merged_bads:
        logger.info(
            "[sub-%s / %s]  Browser updated bad channels: old=%s new=%s",
            label,
            task,
            merged_bads,
            final_bads,
        )
    write_bads_json(paths, label, task, final_bads, logger)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="First-pass QC inspection of raw MEG/EEG .fif files."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--tasks", nargs="+", default=None, choices=TASKS)
    parser.add_argument(
        "--epoch-config",
        default=DEFAULT_EPOCH_CONFIG,
        choices=list(EPOCH_CONFIGS.keys()),
        help="Epoch config used to check expected trigger codes and sweep count",
    )
    parser.add_argument(
        "--browse",
        action="store_true",
        help="Open interactive raw browser (one window per subject/task)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Save summary report to derivatives/logs/inspect_report.txt",
    )
    args = parser.parse_args()

    browser_backend = configure_browser_backend()

    paths = Paths(args.root)
    logger = setup_logging(paths, "inspect")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks = args.tasks if args.tasks else TASKS

    logger.info("Browser backend : %s", browser_backend)
    logger.info("Subjects        : %s", subjects)
    logger.info("Tasks           : %s", tasks)

    all_summaries = []

    for label in subjects:
        for task in tasks:
            logger.info("─" * 50)
            summary = inspect_raw(paths, label, task, args.epoch_config, logger)
            all_summaries.append(summary)

            if args.browse and "FILE NOT FOUND" not in summary["warnings"]:
                browse_raw(paths, label, task, logger)

    logger.info("═" * 50)
    logger.info("INSPECTION SUMMARY")
    logger.info("═" * 50)
    logger.info("%-10s  %-12s  %-6s  %s", "Subject", "Task", "OK", "Warnings")
    logger.info("─" * 50)

    n_ok = 0
    n_warn = 0
    for s in all_summaries:
        status = "✓" if s["ok"] else "⚠"
        if s["ok"]:
            n_ok += 1
        else:
            n_warn += 1
        logger.info(
            "%-10s  %-12s  %-6s  %s",
            s["subject"],
            s["task"],
            status,
            "; ".join(s["warnings"])[:80] if s["warnings"] else "—",
        )

    logger.info("─" * 50)
    logger.info("OK: %d  |  Warnings: %d  |  Total: %d", n_ok, n_warn, len(all_summaries))
    logger.info("═" * 50)

    if args.report:
        report_path = paths.log_dir() / "inspect_report.txt"
        lines = ["INSPECTION REPORT\n", "=" * 50 + "\n"]
        for s in all_summaries:
            lines.append(f"\nsub-{s['subject']} / {s['task']}\n")
            lines.append(f"  File        : {s['file']}\n")
            lines.append(f"  Duration    : {s.get('duration_s', '?')} s\n")
            lines.append(f"  Events      : {s.get('events', {})}\n")
            lines.append(f"  Bads (FIF)  : {s.get('bads_in_file', [])}\n")
            lines.append(f"  Bads (JSON) : {s.get('bads_in_json', [])}\n")
            lines.append(f"  Bads (all)  : {s.get('bads_merged', [])}\n")
            lines.append(f"  Status      : {'OK' if s['ok'] else 'WARNINGS'}\n")
            for w in s["warnings"]:
                lines.append(f"  ⚠  {w}\n")
        report_path.write_text("".join(lines), encoding="utf-8")
        logger.info("Report saved: %s", report_path)

    if n_warn:
        sys.exit(1)


if __name__ == "__main__":
    main()
