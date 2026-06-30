#!/usr/bin/env python3
"""
trial_counts.py
-----------------
Computes true total / after-reject / perceived / not-perceived trial
counts per subject and task, reading ratings DIRECTLY from the same
sources as epoch.py (triggercheck JSON first, behavioural mat file
fallback) rather than relying on epoch_trial_counts.tsv.

This matters because epoch.py now defaults to keeping ALL non-rejected
trials (--perceived is opt-in). If epoch.py was last run without
--perceived, its own trial_counts TSV will show n_after_perceived ==
n_after_reject for every subject (no filter was applied), which is NOT
the same as "everyone was perceived". This script reads the actual
ratings independently of which mode epoch.py was last run in, so the
perceived/not-perceived breakdown is always correct.

How it works
------------
For each subject/task:
  1. Load the full rating list (n_total trials) from JSON or mat,
     same priority/parsing as epoch.py and plot_ratings.py.
  2. Load the saved epoch .fif file and read epochs.selection — the
     0-based indices of the ORIGINAL trials that survived amplitude
     rejection (epoch.py's rejection step, independent of perceived
     filtering).
  3. For each surviving index, look up its rating:
       rating > 0   -> perceived
       rating == 0  -> not perceived
       rating is None (miss) -> excluded from both
  4. n_total, n_after_reject, n_perceived, n_nonperceived, and
     percentages are computed directly from this — never depends on
     whether epoch.py was run with --perceived.

Outputs
-------
    derivatives/logs/trial_counts/trial_counts.tsv
    derivatives/logs/trial_counts/trial_counts.html   (with --html)
    LaTeX table printed to stdout                     (with --latex)

Usage
-----
    python code/trial_counts.py --root $MEGROOT
    python code/trial_counts.py --root $MEGROOT --html --latex
"""

import argparse
import csv
from pathlib import Path

import mne

from core import Paths, TASKS, EPOCH_CONFIGS, DEFAULT_EPOCH_CONFIG, load_subjects, sub_id
from ratings_io import load_ratings


# ---------------------------------------------------------------------------
# Per-subject/task computation
# ---------------------------------------------------------------------------

def compute_counts(paths: Paths, label: str, task: str, epoch_config: str):
    ratings = load_ratings(paths, label, task)
    n_total = len(ratings)
    if n_total == 0:
        return None

    cfg = EPOCH_CONFIGS.get(epoch_config, {})
    desc = cfg.get("desc", epoch_config)
    epo_file = (
        paths.deriv / "epochs" / sub_id(label) / "meg"
        / f"{sub_id(label)}_task-{task}_desc-{desc}-preproc_epo.fif"
    )
    if not epo_file.exists():
        return {
            "subject": label, "task": task,
            "n_total": n_total, "n_after_reject": "-",
            "n_perceived": "-", "n_nonperceived": "-", "n_miss": "-",
            "pct_rejected": "-", "pct_perceived": "-", "pct_nonperceived": "-",
        }

    epochs = mne.read_epochs(str(epo_file), preload=False, verbose=False)
    n_after_reject = len(epochs)
    selection = epochs.selection  # original 0-based indices that survived

    n_perceived = n_nonperceived = n_miss = 0
    for i in selection:
        if i >= n_total:
            continue
        val = ratings[i]
        # Misses can come back from ratings_io as None, but also as the raw
        # mat-file marker ('miss' string, or ['miss'] nested array, as seen
        # in sub-5026 pinprick) if that marker isn't normalized upstream.
        # Treat anything that isn't a plain number as a miss rather than
        # letting it silently fall through to "not perceived".
        if not isinstance(val, (int, float)):
            n_miss += 1
        elif val > 0:
            n_perceived += 1
        else:
            n_nonperceived += 1

    pct_rejected = 100 * (n_total - n_after_reject) / n_total if n_total else 0
    pct_perceived = 100 * n_perceived / n_total if n_total else 0
    pct_nonperceived = 100 * n_nonperceived / n_total if n_total else 0

    return {
        "subject": label, "task": task,
        "n_total": n_total, "n_after_reject": n_after_reject,
        "n_perceived": n_perceived, "n_nonperceived": n_nonperceived,
        "n_miss": n_miss,
        "pct_rejected": round(pct_rejected, 1),
        "pct_perceived": round(pct_perceived, 1),
        "pct_nonperceived": round(pct_nonperceived, 1),
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Trial Counts</title>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif;
         background: #1e1e1e; color: #ddd; padding: 24px; }
  .timestamp { color: #888; font-size: 12px; margin-bottom: 16px; }
  table { border-collapse: collapse; font-size: 13px; }
  th { background: #2d2d2d; color: #fff; padding: 6px 10px; text-align: left;
       border: 1px solid #444; position: sticky; top: 0; }
  td { padding: 5px 10px; border: 1px solid #333; text-align: center; }
  td.subject { text-align: left; font-weight: 600; background: #262626; }
  tr.shade td { background: #252b33; }
  tr.shade td.subject { background: #2c333c; }
  .n { color: #9ecbff; }
  .summary td { font-weight: 700; border-top: 2px solid #555; }
</style></head><body>
"""
_HTML_TAIL = "</body></html>\n"


def write_html(rows: list[dict], summary_rows: list[dict], out_path: Path) -> None:
    from datetime import datetime
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = [_HTML_HEAD]
    html.append(f'<div class="timestamp">Generated: {generated}</div>')
    html.append("<table><tr>"
                 "<th>Nr</th><th>Subject</th><th>Task</th>"
                 "<th>n_total</th><th>n_after_reject</th>"
                 "<th>n_perceived</th><th>n_nonperceived</th><th>n_miss</th>"
                 "<th>% rejected</th><th>% perceived</th><th>% nonperceived</th>"
                 "</tr>")
    prev_subject = None
    subj_counter = 0
    shade = False
    for r in rows:
        is_new_subject = r["subject"] != prev_subject
        if is_new_subject:
            subj_counter += 1
            shade = (subj_counter % 2 == 0)
            prev_subject = r["subject"]
        nr_cell = subj_counter if is_new_subject else ""
        tr_class = ' class="shade"' if shade else ""
        html.append(
            f"<tr{tr_class}>"
            f'<td>{nr_cell}</td>'
            f'<td class="subject">{r["subject"]}</td>'
            f'<td>{r["task"]}</td>'
            f'<td class="n">{r["n_total"]}</td>'
            f'<td class="n">{r["n_after_reject"]}</td>'
            f'<td class="n">{r["n_perceived"]}</td>'
            f'<td class="n">{r["n_nonperceived"]}</td>'
            f'<td class="n">{r["n_miss"]}</td>'
            f'<td class="n">{r["pct_rejected"]}</td>'
            f'<td class="n">{r["pct_perceived"]}</td>'
            f'<td class="n">{r["pct_nonperceived"]}</td>'
            "</tr>"
        )
    for r in summary_rows:
        html.append(
            '<tr class="summary">'
            f'<td></td><td class="subject">{r["subject"]}</td><td>{r["task"]}</td>'
            f'<td class="n">{r["n_total"]}</td>'
            f'<td class="n">{r["n_after_reject"]}</td>'
            f'<td class="n">{r["n_perceived"]}</td>'
            f'<td class="n">{r["n_nonperceived"]}</td>'
            f'<td class="n">{r["n_miss"]}</td>'
            f'<td class="n">{r["pct_rejected"]}</td>'
            f'<td class="n">{r["pct_perceived"]}</td>'
            f'<td class="n">{r["pct_nonperceived"]}</td>'
            "</tr>"
        )
    html.append("</table>")
    html.append(_HTML_TAIL)
    out_path.write_text("\n".join(html), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute true total/after-reject/perceived/not-perceived "
                     "trial counts, reading ratings directly (independent of "
                     "epoch.py's --perceived flag)."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--epoch-config", default=DEFAULT_EPOCH_CONFIG)
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    paths = Paths(args.root)
    subjects = args.subjects if args.subjects else load_subjects(paths)

    out_dir = paths.log_dir() / "trial_counts"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label in subjects:
        for task in TASKS:
            r = compute_counts(paths, label, task, args.epoch_config)
            if r:
                rows.append(r)

    # --- Summary rows ---
    def summarize(task_filter, label_str):
        relevant = [r for r in rows
                    if (task_filter is None or r["task"] == task_filter)
                    and isinstance(r["n_total"], int)]
        if not relevant:
            return None
        n_total = sum(r["n_total"] for r in relevant)
        n_after_reject = sum(r["n_after_reject"] for r in relevant
                              if isinstance(r["n_after_reject"], int))
        n_perceived = sum(r["n_perceived"] for r in relevant
                           if isinstance(r["n_perceived"], int))
        n_nonperceived = sum(r["n_nonperceived"] for r in relevant
                              if isinstance(r["n_nonperceived"], int))
        n_miss = sum(r["n_miss"] for r in relevant
                     if isinstance(r["n_miss"], int))
        return {
            "subject": label_str, "task": "(all)" if task_filter is None else task_filter,
            "n_total": n_total, "n_after_reject": n_after_reject,
            "n_perceived": n_perceived, "n_nonperceived": n_nonperceived,
            "n_miss": n_miss,
            "pct_rejected": round(100 * (n_total - n_after_reject) / n_total, 1) if n_total else 0,
            "pct_perceived": round(100 * n_perceived / n_total, 1) if n_total else 0,
            "pct_nonperceived": round(100 * n_nonperceived / n_total, 1) if n_total else 0,
        }

    summary_rows = []
    for task in TASKS:
        s = summarize(task, f"SUM ({task})")
        if s:
            summary_rows.append(s)
    grand = summarize(None, "SUM (all tasks)")
    if grand:
        summary_rows.append(grand)

    # --- Save TSV ---
    out_tsv = out_dir / "trial_counts.tsv"
    fieldnames = ["subject", "task", "n_total", "n_after_reject",
                  "n_perceived", "n_nonperceived", "n_miss",
                  "pct_rejected", "pct_perceived", "pct_nonperceived"]
    with open(out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows + summary_rows)
    print(f"Saved: {out_tsv}")
    print(f"Subjects x tasks: {len(rows)}")

    if args.html:
        out_html = out_dir / "trial_counts.html"
        write_html(rows, summary_rows, out_html)
        print(f"Saved: {out_html}")

    if args.latex:
        print()
        print(r"% Requires \usepackage[table]{xcolor} in the preamble for \rowcolor")
        print(r"\begin{longtable}{l|l|l|c|c|c|c|c|c|c|c}")
        print(r"Nr & Subject & Task & n\_total & n\_after\_reject & "
              r"n\_perceived & n\_nonperceived & n\_miss & "
              r"\%rejected & \%perceived & \%nonperceived \\ \hline")
        prev_subject = None
        subj_counter = 0
        shade = False
        for r in rows:
            is_new_subject = r["subject"] != prev_subject
            if is_new_subject:
                subj_counter += 1
                shade = (subj_counter % 2 == 0)
                prev_subject = r["subject"]
            nr_cell = subj_counter if is_new_subject else ""
            rowcolor = r"\rowcolor[gray]{0.92} " if shade else ""
            print(
                f"{rowcolor}{nr_cell} & {r['subject']} & {r['task']} & {r['n_total']} & "
                f"{r['n_after_reject']} & {r['n_perceived']} & "
                f"{r['n_nonperceived']} & {r['n_miss']} & "
                f"{r['pct_rejected']} & {r['pct_perceived']} & "
                f"{r['pct_nonperceived']} " + r"\\"
            )
        for r in summary_rows:
            print(
                f" & {r['subject']} & {r['task']} & {r['n_total']} & "
                f"{r['n_after_reject']} & {r['n_perceived']} & "
                f"{r['n_nonperceived']} & {r['n_miss']} & "
                f"{r['pct_rejected']} & {r['pct_perceived']} & "
                f"{r['pct_nonperceived']} " + r"\\"
            )
        print(r"\end{longtable}")


if __name__ == "__main__":
    main()
