#!/usr/bin/env python3
"""
task_order.py
--------------
Reads the measurement timestamp (raw.info['meas_date']) from each
subject's three raw fif files (laser, pinprick, tactile) to determine
the actual order tasks were recorded in. Useful for checking
counterbalancing/randomization across the cohort.

Outputs:
  - derivatives/logs/task_order.tsv   — one row per subject:
        subject, laser_time, pinprick_time, tactile_time, order
    where 'order' is e.g. "laser > pinprick > tactile"
  - A summary count of how many subjects started with each task,
    and the full distribution of all 6 possible orderings.

Usage
-----
    python code/task_order.py --root $MEGROOT
    python code/task_order.py --root $MEGROOT --subjects 1409 3691
"""

import argparse
import csv
from collections import Counter
from itertools import permutations
from pathlib import Path

import mne

from core import Paths, TASKS, load_subjects, sub_id


def get_meas_time(fif_path: Path):
    """Return the measurement datetime for a raw fif, or None if missing."""
    if not fif_path.exists():
        return None
    try:
        raw = mne.io.read_raw_fif(str(fif_path), preload=False, verbose=False)
        meas_date = raw.info.get("meas_date")
        return meas_date
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Determine task recording order per subject from raw "
                     "fif timestamps."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    args = parser.parse_args()

    paths = Paths(args.root)
    subjects = args.subjects if args.subjects else load_subjects(paths)

    out_dir = paths.log_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / "task_order.tsv"

    rows = []
    for label in subjects:
        times = {}
        for task in TASKS:
            fif_path = (
                paths.raw / sub_id(label) / "meg"
                / f"{sub_id(label)}_task-{task}_meg.fif"
            )
            times[task] = get_meas_time(fif_path)

        valid = {t: ts for t, ts in times.items() if ts is not None}
        if len(valid) < 2:
            order_str = "incomplete"
            order_tasks = []
        else:
            order_tasks = sorted(valid, key=lambda t: valid[t])
            order_str = " > ".join(order_tasks)

        row = {
            "subject": label,
            "laser_time":    times["laser"].strftime("%Y-%m-%d %H:%M") if times.get("laser") else "-",
            "pinprick_time": times["pinprick"].strftime("%Y-%m-%d %H:%M") if times.get("pinprick") else "-",
            "tactile_time":  times["tactile"].strftime("%Y-%m-%d %H:%M") if times.get("tactile") else "-",
            "order": order_str,
            "first_task": order_tasks[0] if order_tasks else "-",
        }
        rows.append(row)

    fieldnames = ["subject", "laser_time", "pinprick_time", "tactile_time",
                  "order", "first_task"]
    with open(out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {out_tsv}\n")

    out_pos_tsv = out_dir / "task_order_position_summary.tsv"

    # --- Summary: position breakdown (1st/2nd/3rd) per task, counts + % ---
    complete_rows = [r for r in rows if r["order"] != "incomplete"]
    n_complete = len(complete_rows)

    position_counts = {task: [0, 0, 0] for task in TASKS}  # [1st, 2nd, 3rd]
    for r in complete_rows:
        order_tasks = r["order"].split(" > ")
        for pos, task in enumerate(order_tasks):
            position_counts[task][pos] += 1

    print("Task position breakdown (n = %d complete subjects):\n" % n_complete)
    header = f"  {'Task':10s} {'1st':>14s} {'2nd':>14s} {'3rd':>14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    pos_rows = []
    for task in TASKS:
        counts = position_counts[task]
        pcts = [100 * c / n_complete if n_complete else 0 for c in counts]
        row_str = f"  {task:10s} " + " ".join(
            f"{c:3d} ({p:5.1f}%)".rjust(14) for c, p in zip(counts, pcts)
        )
        print(row_str)
        pos_rows.append({
            "task": task,
            "n_1st": counts[0], "pct_1st": round(pcts[0], 1),
            "n_2nd": counts[1], "pct_2nd": round(pcts[1], 1),
            "n_3rd": counts[2], "pct_3rd": round(pcts[2], 1),
        })

    with open(out_pos_tsv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["task", "n_1st", "pct_1st", "n_2nd", "pct_2nd",
                           "n_3rd", "pct_3rd"],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(pos_rows)
    print(f"\nSaved: {out_pos_tsv}")

    # --- Summary: full ordering distribution ---
    order_counts = Counter(r["order"] for r in complete_rows)
    all_orderings = [" > ".join(p) for p in permutations(TASKS)]
    print("\nFull order distribution (all 6 possible orderings):")
    for ordering in all_orderings:
        n = order_counts.get(ordering, 0)
        pct = 100 * n / n_complete if n_complete else 0
        print(f"  {ordering:35s}: {n:3d}  ({pct:5.1f}%)")

    n_incomplete = sum(1 for r in rows if r["order"] == "incomplete")
    if n_incomplete:
        print(f"\n{n_incomplete} subject(s) with incomplete/missing timestamps:")
        for r in rows:
            if r["order"] == "incomplete":
                print(f"  sub-{r['subject']}")

    print(f"\nTotal subjects: {len(rows)}")


if __name__ == "__main__":
    main()
