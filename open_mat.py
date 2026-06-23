#!/usr/bin/env python3
"""
open_mat.py
-----------
Browse, rename and edit the behavioural rating mat files for the
laser-pain MEG study.

What it does
------------
1. Finds all .mat files under rawdata/sub-{label}/beh/
2. Shows the fif recording timestamps so you can match mat files to tasks
3. Lets you rename each mat file to the correct BIDS name
   (sub-{label}_task-{task}_ratings.mat)
4. Prints the full trial list (intensity + quality) for each file
5. Lets you edit individual trials if needed

This script does NOT do any trigger decoding or rating matching.
Rating matching for standard subjects (pinprick, tactile, and laser
subjects with code 32) is handled by match_ratings.py.
For laser triggercheck subjects (compound trigger group) the ratings
live in the triggercheck JSON under derivatives/trigger_check/.

Usage
-----
    python code/open_mat.py --root $MEGROOT
    python code/open_mat.py --root $MEGROOT --subjects 1409 3691
    python code/open_mat.py --root $MEGROOT --tasks laser
"""

import argparse
import re
import sys
from pathlib import Path

import mne
import numpy as np
import scipy.io

from core import TASKS, Paths, load_subjects, setup_logging, sub_id

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_fif_timestamps(paths: Paths, label: str, tasks: list[str]) -> dict[str, str]:
    """Return local (Berlin) recording time for each task fif file."""
    from datetime import timezone, timedelta
    import datetime as dt_module

    def utc_to_berlin(utc_dt):
        try:
            import zoneinfo
            return utc_dt.astimezone(zoneinfo.ZoneInfo("Europe/Berlin"))
        except Exception:
            pass
        try:
            import pytz
            return utc_dt.astimezone(pytz.timezone("Europe/Berlin"))
        except Exception:
            pass
        year = utc_dt.year
        m31  = dt_module.datetime(year, 3, 31, 1, 0, tzinfo=timezone.utc)
        cest = m31 - timedelta(days=m31.weekday() + 1)
        o31  = dt_module.datetime(year, 10, 31, 1, 0, tzinfo=timezone.utc)
        cend = o31 - timedelta(days=o31.weekday() + 1)
        return utc_dt + timedelta(hours=2 if cest <= utc_dt < cend else 1)

    ts = {}
    for task in tasks:
        fif = paths.raw_meg(label, task)
        if not fif.exists():
            ts[task] = "fif not found"
            continue
        try:
            raw      = mne.io.read_raw_fif(str(fif), preload=False, verbose=False)
            dt       = raw.info["meas_date"]
            ts[task] = utc_to_berlin(dt).strftime("%Y%m%d_%Hh%M") if dt else "no timestamp"
        except Exception as e:
            ts[task] = f"error: {e}"
    return ts


def _find_mat_files(paths: Paths, subjects: list[str]) -> list[tuple[str, Path]]:
    found = []
    for label in subjects:
        beh = paths.raw / sub_id(label) / "beh"
        if beh.exists():
            for f in sorted(beh.glob("*.mat")):
                found.append((label, f))
    return found


def _print_mat_summary(fpath: Path) -> None:
    try:
        mat   = scipy.io.loadmat(str(fpath))
        r     = mat["response"][0, 0]
        resps = r["responses"]
        n     = resps.shape[1]
        has_q = resps.shape[0] > 1
        print(f"\n{'─'*60}")
        print(f"File: {fpath.name}   Trials: {n}   Quality: {has_q}")
        print(f"{'─'*60}")
        for i in range(n):
            inten = str(resps[0, i].flat[0]).strip()
            qual  = str(resps[1, i].flat[0]).strip() if has_q else ""
            print(f"  {i+1:<5} {inten:<12} {qual}")
        print(f"{'─'*60}")
    except Exception as e:
        print(f"  ERROR reading mat: {e}")


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def open_mat_interactive(paths: Paths, subjects: list[str],
                         tasks: list[str], logger) -> None:
    mat_files = _find_mat_files(paths, subjects)
    if not mat_files:
        print("\nNo mat files found in rawdata/sub-{label}/beh/")
        return

    print("\nReading fif timestamps ...")
    seen   = list(dict.fromkeys(lb for lb, _ in mat_files))
    fif_ts = {lb: _get_fif_timestamps(paths, lb, tasks) for lb in seen}

    print(f"\nFound {len(mat_files)} mat file(s):")
    for i, (lb, fp) in enumerate(mat_files):
        expected = [f"{sub_id(lb)}_task-{t}_ratings.mat" for t in tasks]
        status   = "✓" if fp.name in expected else "⚠  needs renaming"
        print(f"  [{i+1}] sub-{lb}  {fp.name}  {status}")

    print("\nFIF recording timestamps:")
    print(f"  {'Subject':<12} {'Task':<12} {'Local time'}")
    print(f"  {'─'*44}")
    for lb in seen:
        for t in tasks:
            print(f"  {f'sub-{lb}':<12} {t:<12} {fif_ts[lb].get(t, '?')}")
    print()

    idx = 0
    while idx < len(mat_files):
        lb, fp = mat_files[idx]
        _print_mat_summary(fp)
        expected = [f"{sub_id(lb)}_task-{t}_ratings.mat" for t in tasks]

        # Offer rename if filename doesn't match BIDS convention
        if fp.name not in expected:
            print(f"\nsub-{lb} | {fp.name} | needs task assignment")
            for ti, t in enumerate(tasks):
                print(f"  [{ti+1}] {t:<12} recorded: {fif_ts[lb].get(t, '?')}")
            ts_m = re.search(r"(\d{8}_\d{2}h\d{2})", fp.stem)
            print(f"  Mat timestamp: {ts_m.group(1) if ts_m else 'not found'}")
            print("  [s] skip   [q] quit")
            ch = input("Choice: ").strip().lower()
            if ch == "q":
                break
            if ch == "s":
                idx += 1
                continue
            if ch.isdigit() and 1 <= int(ch) <= len(tasks):
                task     = tasks[int(ch) - 1]
                new_name = f"{sub_id(lb)}_task-{task}_ratings.mat"
                new_path = fp.parent / new_name
                new_path.parent.mkdir(parents=True, exist_ok=True)
                if new_path.exists():
                    if input(f"{new_name} exists. Overwrite? [y/n]: "
                             ).strip().lower() != "y":
                        idx += 1
                        continue
                fp.rename(new_path)
                print(f"✓ Renamed to {new_name}")
                mat_files[idx] = (lb, new_path)
                fp = new_path

        # Edit / next / quit
        print("\n[e] edit trial   [n] next   [q] quit")
        while True:
            a = input("Action: ").strip().lower()
            if a == "q":
                print("Done.")
                return
            if a == "n":
                break
            if a == "e":
                try:
                    tn    = int(input("Trial #: ").strip())
                    mat   = scipy.io.loadmat(str(fp))
                    r     = mat["response"][0, 0]
                    resps = r["responses"]
                    if not (1 <= tn <= resps.shape[1]):
                        print(f"Must be 1–{resps.shape[1]}")
                        continue
                    i = tn - 1
                    print(f"Current intensity: {resps[0, i].flat[0]}")
                    v = input("New value (Enter = keep): ").strip()
                    if v:
                        resps[0, i] = np.array([[v]], dtype=object)
                    if resps.shape[0] > 1:
                        print(f"Current quality: {resps[1, i].flat[0]}")
                        v = input("New value (Enter = keep): ").strip()
                        if v:
                            resps[1, i] = np.array([[v]], dtype=object)
                    mat["response"][0, 0]["responses"] = resps
                    scipy.io.savemat(str(fp), mat)
                    print("✓ Saved")
                    _print_mat_summary(fp)
                except Exception as e:
                    print(f"Error: {e}")

        idx += 1

    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Browse, rename and edit mat rating files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--root",     type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--tasks",    nargs="+", default=None, choices=TASKS)
    args = parser.parse_args()

    paths    = Paths(args.root)
    logger   = setup_logging(paths, "open_mat")
    subjects = args.subjects if args.subjects else load_subjects(paths)
    tasks    = args.tasks    if args.tasks    else TASKS

    open_mat_interactive(paths, subjects, tasks, logger)


if __name__ == "__main__":
    main()
