#!/usr/bin/env python3
"""
pipeline_status.py
-------------------
Auto-generates a per-subject pipeline progress overview, replacing manual
LaTeX table editing. Scans derivatives/ and rawdata/ for expected output
files and ticks off each step automatically.

Columns (in pipeline order)
----------------------------
    info              age/sex filled in participants.tsv
    mat               behavioural ratings mat file exists
    trigcheck_json    triggercheck JSON exists (n/a if never needed)
    bads              bad-channel json exists (any task)
    prep              preprocessed fif exists (any task)
    epoch             epoch fif exists for all 3 tasks
    mri               'individual' / 'scaled-fsaverage' / 'fsaverage' / 'no'
    coreg             coregistration trans file exists (any task)
    bem               BEM solution exists under derivatives/source/.../bem/
    source            inverse operator exists (any task)
    parcellation      HCPMMP1 annot files exist
    wpli / pac / psi  connectivity result files exist

Trial counts (separate table, authoritative source)
------------------------------------------------------
Reads derivatives/logs/trial_counts/trial_counts.tsv, produced by
trial_counts.py — which reads ratings directly (JSON/mat) and is correct
regardless of whether epoch.py was last run with --perceived. Run
trial_counts.py before this script for the trial-count table to be
populated.

Adding a new subject to rawdata/ + participants.tsv automatically makes
them appear on the next run — no manual table editing needed.

Usage
-----
    python code/trial_counts.py --root $MEGROOT          # run first
    python code/pipeline_status.py --root $MEGROOT --html --latex
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

from core import Paths, TASKS, ATLAS_CONFIGS, DEFAULT_ATLAS, load_subjects, sub_id


def _mtime_str(path: Path) -> str:
    """Return 'YYYY-MM-DD HH:MM' last-modified timestamp, or '' if missing."""
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def _newest_mtime(paths_list: list[Path]) -> str:
    existing = [p for p in paths_list if p.exists()]
    if not existing:
        return ""
    newest = max(existing, key=lambda p: p.stat().st_mtime)
    return _mtime_str(newest)


def check_info(paths: Paths, label: str) -> bool:
    """True if age/sex are filled in (not 'n/a') in participants.tsv."""
    tsv = paths.raw / "participants.tsv"
    if not tsv.exists():
        return False
    with open(tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("participant_id", "").replace("sub-", "") == label:
                age = str(row.get("age", "")).strip().lower()
                sex = str(row.get("sex", "")).strip().lower()
                return age not in ("", "n/a") and sex not in ("", "n/a")
    return False


def check_mat(paths: Paths, label: str) -> bool:
    """True if at least one behavioural ratings mat file exists."""
    beh_dir = paths.raw / sub_id(label) / "beh"
    if not beh_dir.exists():
        return False
    return any(beh_dir.glob(f"{sub_id(label)}_task-*_ratings.mat"))


def check_parcellation(paths: Paths, label: str, atlas_key: str = DEFAULT_ATLAS) -> bool:
    parc = ATLAS_CONFIGS[atlas_key]["parc"]
    label_dir = paths.freesurfer_dir() / sub_id(label) / "label"
    lh = label_dir / f"lh.{parc}.annot"
    rh = label_dir / f"rh.{parc}.annot"
    return lh.exists() and rh.exists()


def check_wpli(paths: Paths, label: str) -> bool:
    conn_dir = paths.deriv / "connectivity" / sub_id(label)
    if not conn_dir.exists():
        return False
    return any(conn_dir.glob("task-*/*_wpli_painmatrix*.h5"))


def check_pac(paths: Paths, label: str) -> bool:
    conn_dir = paths.deriv / "connectivity" / sub_id(label)
    if not conn_dir.exists():
        return False
    return any(conn_dir.glob("task-*/*_pac_painmatrix*.h5"))


def check_psi(paths: Paths, label: str) -> bool:
    conn_dir = paths.deriv / "connectivity" / sub_id(label)
    if not conn_dir.exists():
        return False
    return any(conn_dir.glob("task-*/*_psi_painmatrix*.h5"))


def check_bads(paths: Paths, label: str) -> bool:
    return any(
        (paths.deriv / "bads" / f"{sub_id(label)}_task-{t}_bads.json").exists()
        for t in TASKS
    )


def check_trigcheck_json(paths: Paths, label: str) -> bool:
    return (paths.deriv / "trigger_check" / sub_id(label)).exists()


def check_prep(paths: Paths, label: str) -> bool:
    return any(
        paths.prep_raw(label, t).exists() for t in TASKS
    )


def check_epoch(paths: Paths, label: str) -> bool:
    """Checks for the actual saved epoch filename pattern, nested under
    derivatives/epochs/sub-{label}/meg/ — not directly in derivatives/epochs/."""
    epochs_dir = paths.deriv / "epochs" / sub_id(label) / "meg"
    if not epochs_dir.exists():
        return False
    return all(
        any(epochs_dir.glob(f"{sub_id(label)}_task-{t}_desc-*-preproc_epo.fif"))
        for t in TASKS
    )


def check_mri(paths: Paths, label: str) -> str:
    """Returns one of:
      'individual'       — subject has their own recon-all reconstruction
      'scaled-fsaverage'  — fsaverage template scaled to the subject's head
                             via mne.scale_mri() (no real individual MRI was
                             acquired); detected via the presence of
                             "MRI scaling parameters.cfg", which mne.scale_mri
                             writes and a true recon-all run never creates
      'fsaverage'         — falling back to the unscaled template entirely
      'no'                — neither exists
    """
    sub_dir = paths.freesurfer_dir() / sub_id(label)
    scaling_cfg = sub_dir / "MRI scaling parameters.cfg"
    surf_dir = sub_dir / "surf"
    has_surfaces = (surf_dir / "lh.white").exists() and (surf_dir / "rh.white").exists()

    if scaling_cfg.exists():
        return "scaled-fsaverage"
    if has_surfaces:
        return "individual"
    fsavg_dir = paths.freesurfer_dir() / "fsaverage" / "surf"
    if fsavg_dir.exists():
        return "fsaverage"
    return "no"


def check_coreg(paths: Paths, label: str) -> bool:
    return any(paths.trans(label, t).exists() for t in TASKS)


def check_bem(paths: Paths, label: str) -> bool:
    """Checks for a BEM solution file. The solution lives under
    derivatives/source/sub-{label}/bem/ (computed by source.py), not
    under derivatives/freesurfer/ -- that folder only has the raw
    watershed surfaces and unsolved BEM geometry."""
    bem_dir = paths.source_dir(label) / "bem"
    if not bem_dir.exists():
        return False
    return any(bem_dir.glob("*-bem-sol.fif"))


def check_source(paths: Paths, label: str) -> bool:
    src_dir = paths.source_dir(label)
    if not src_dir.exists():
        return False
    return any(src_dir.glob("task-*/meg/*_inv.fif"))


def load_trial_counts(paths: Paths) -> dict:
    """Returns {(label, task): {field: value, ...}} from trial_counts.tsv,
    produced by trial_counts.py. This is the authoritative source for
    perceived/not-perceived breakdowns — it reads ratings directly and is
    correct regardless of whether epoch.py was last run with --perceived.
    (epoch_trial_counts.tsv from epoch.py only reflects whatever filter
    was active during that specific run, so it is not used here.)"""
    tsv = paths.log_dir() / "trial_counts" / "trial_counts.tsv"
    if not tsv.exists():
        return {}
    out = {}
    with open(tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row["subject"].replace("sub-", "")
            if label.startswith("SUM"):
                continue   # skip summary rows already baked into the TSV
            task = row["task"]
            out[(label, task)] = row
    return out


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------

_HTML_HEAD = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pipeline Status</title>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif;
         background: #1e1e1e; color: #ddd; padding: 24px; }
  h1 { font-size: 20px; margin-bottom: 4px; }
  .timestamp { color: #888; font-size: 12px; margin-bottom: 24px; }
  table { border-collapse: collapse; margin-bottom: 40px; font-size: 13px; }
  th { background: #2d2d2d; color: #fff; padding: 6px 10px; text-align: left;
       position: sticky; top: 0; border: 1px solid #444; }
  td { padding: 5px 10px; border: 1px solid #333; text-align: center; }
  td.subject { text-align: left; font-weight: 600; background: #262626; }
  .yes  { background: #1b3a1b; color: #7CFC7C; }
  .na   { background: #2a2a35; color: #9090a8; }
  .no   { background: #3a1b1b; color: #FF8C8C; }
  .fsavg { background: #3a3010; color: #e0c060; }
  .date { color: #999; font-size: 11px; white-space: nowrap; }
  .date.stale { color: #e0a030; font-weight: 600; }
  .n    { color: #9ecbff; }
  tr:hover td { filter: brightness(1.3); }
</style>
</head>
<body>
"""

_HTML_TAIL = "</body></html>\n"


def _cell(value: str) -> str:
    cls = "yes" if value == "yes" else "no"
    return f'<td class="{cls}">{value}</td>'


def _na_cell(value: str) -> str:
    """For columns where 'not applicable' is a valid, non-actionable state
    (e.g. trigcheck_json — only used for subjects with trigger issues)."""
    if value == "yes":
        return '<td class="yes">yes</td>'
    return '<td class="na">n/a</td>'


def _date_cell(value: str, ref_date: str = "") -> str:
    """Highlight a date cell as 'stale' if it's older than a reference date
    (e.g. epoch_date older than prep_date means epoch.py needs rerunning)."""
    if not value:
        return '<td class="date">-</td>'
    cls = "date"
    if ref_date and value < ref_date:
        cls += " stale"
    return f'<td class="{cls}">{value}</td>'


def _mri_cell(value: str) -> str:
    if value == "individual":
        return '<td class="yes">individual</td>'
    if value == "scaled-fsaverage":
        return '<td class="fsavg">scaled fsavg</td>'
    if value == "fsaverage":
        return '<td class="fsavg">fsaverage</td>'
    return '<td class="no">no</td>'


def write_html_dashboard(rows: list[dict], trial_rows: list[dict], out_path: Path) -> None:
    from datetime import datetime
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = [_HTML_HEAD]
    html.append(f'<div class="timestamp">Generated: {generated}  |  '
                f'{len(rows)} subjects</div>')

    # --- Single merged table, columns in actual pipeline order ---
    # info / mat / trigcheck_json are basics, checked first; then the
    # processing steps in the order the corresponding scripts run:
    # bads -> prep -> epoch -> mri -> coreg -> bem -> source ->
    # parcellation -> wpli -> pac -> psi
    html.append("<table><tr>"
                 "<th>Nr</th><th>Subject</th>"
                 "<th>Info</th><th>.mat</th><th>TrigJSON</th>"
                 "<th>Bads</th>"
                 "<th>Prep</th><th>Prep date</th>"
                 "<th>Epoch</th><th>Epoch date</th>"
                 "<th>MRI</th><th>Coreg</th><th>BEM</th>"
                 "<th>Source</th><th>Source date</th>"
                 "<th>Parcellation</th>"
                 "<th>WPLI</th><th>WPLI date</th>"
                 "<th>PAC</th><th>PAC date</th>"
                 "<th>PSI</th><th>PSI date</th>"
                 "</tr>")
    for row in rows:
        html.append(
            "<tr>"
            f'<td>{row["idx"]}</td>'
            f'<td class="subject">{row["subject"]}</td>'
            f'{_cell(row["info"])}'
            f'{_cell(row["mat"])}'
            f'{_na_cell(row["trigcheck_json"])}'
            f'{_cell(row["bads"])}'
            f'{_cell(row["prep"])}'
            f'{_date_cell(row["prep_date"])}'
            f'{_cell(row["epoch"])}'
            f'{_date_cell(row["epoch_date"], row["prep_date"])}'
            f'{_mri_cell(row["mri"])}'
            f'{_cell(row["coreg"])}'
            f'{_cell(row["bem"])}'
            f'{_cell(row["source"])}'
            f'{_date_cell(row["source_date"], row["epoch_date"])}'
            f'{_cell(row["parcellation"])}'
            f'{_cell(row["wpli"])}'
            f'{_date_cell(row["wpli_date"], row["source_date"])}'
            f'{_cell(row["pac"])}'
            f'{_date_cell(row["pac_date"], row["source_date"])}'
            f'{_cell(row["psi"])}'
            f'{_date_cell(row["psi_date"], row["source_date"])}'
            "</tr>"
        )
    html.append("</table>")

    html.append(
        '<div class="timestamp">Orange dates = older than an upstream '
        'step\'s output — likely needs to be rerun. '
        '"fsaverage" in the MRI column means no individual reconstruction '
        'was available and the template brain was used instead.</div>'
    )

    # --- Trial counts table, moved to the bottom ---
    html.append("<h2>Trial Counts</h2>")
    html.append("<table><tr>"
                 "<th>Nr</th><th>Subject</th><th>Task</th>"
                 "<th>n_total</th><th>n_after_reject</th>"
                 "<th>n_perceived</th><th>n_nonperceived</th><th>n_miss</th>"
                 "<th>% rejected</th><th>% perceived</th>"
                 "</tr>")
    for tr in trial_rows:
        is_summary = str(tr["subject"]).startswith("SUM")
        row_style = ' style="font-weight:700;border-top:2px solid #555;"' if is_summary else ""
        html.append(
            f"<tr{row_style}>"
            f'<td>{tr["idx"]}</td>'
            f'<td class="subject">{tr["subject"]}</td>'
            f'<td>{tr["task"]}</td>'
            f'<td class="n">{tr["n_total"]}</td>'
            f'<td class="n">{tr["n_after_reject"]}</td>'
            f'<td class="n">{tr["n_perceived"]}</td>'
            f'<td class="n">{tr["n_nonperceived"]}</td>'
            f'<td class="n">{tr["n_miss"]}</td>'
            f'<td class="n">{tr["pct_rejected"]}</td>'
            f'<td class="n">{tr["pct_perceived"]}</td>'
            "</tr>"
        )
    html.append("</table>")
    html.append(_HTML_TAIL)

    out_path.write_text("\n".join(html), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-generate pipeline progress TSV per subject."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--latex", action="store_true",
                         help="Also print a LaTeX longtable to stdout")
    parser.add_argument("--html", action="store_true",
                         help="Also write a colour-coded HTML dashboard")
    args = parser.parse_args()

    paths = Paths(args.root)
    subjects = load_subjects(paths)
    trial_counts = load_trial_counts(paths)

    out_dir = paths.log_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = out_dir / "pipeline_status.tsv"

    fieldnames = [
        "idx", "subject", "info", "bads", "trigcheck_json", "prep", "prep_date",
        "epoch", "epoch_date", "mri", "coreg", "bem",
        "source", "source_date",
        "parcellation", "wpli", "wpli_date", "pac", "pac_date",
        "psi", "psi_date", "mat",
    ]
    rows = []
    trial_rows = []
    for idx, label in enumerate(subjects, start=1):
        prep_files  = [paths.prep_raw(label, t) for t in TASKS]
        epochs_dir  = paths.deriv / "epochs" / sub_id(label) / "meg"
        epoch_files = (
            list(epochs_dir.glob(f"{sub_id(label)}_task-*_desc-*-preproc_epo.fif"))
            if epochs_dir.exists() else []
        )
        src_dir     = paths.source_dir(label)
        src_files   = list(src_dir.glob("task-*/meg/*_inv.fif")) if src_dir.exists() else []
        conn_dir    = paths.deriv / "connectivity" / sub_id(label)
        wpli_files  = list(conn_dir.glob("task-*/*_wpli_painmatrix*.h5")) if conn_dir.exists() else []
        pac_files   = list(conn_dir.glob("task-*/*_pac_painmatrix*.h5")) if conn_dir.exists() else []
        psi_files   = list(conn_dir.glob("task-*/*_psi_painmatrix*.h5")) if conn_dir.exists() else []

        row = {
            "idx":            idx,
            "subject":        label,
            "info":           "yes" if check_info(paths, label) else "no",
            "bads":           "yes" if check_bads(paths, label) else "no",
            "trigcheck_json": "yes" if check_trigcheck_json(paths, label) else "n/a",
            "prep":           "yes" if check_prep(paths, label) else "no",
            "prep_date":      _newest_mtime(prep_files),
            "epoch":          "yes" if check_epoch(paths, label) else "no",
            "epoch_date":     _newest_mtime(epoch_files),
            "mri":            check_mri(paths, label),
            "coreg":          "yes" if check_coreg(paths, label) else "no",
            "bem":            "yes" if check_bem(paths, label) else "no",
            "source":         "yes" if check_source(paths, label) else "no",
            "source_date":    _newest_mtime(src_files),
            "parcellation":   "yes" if check_parcellation(paths, label) else "no",
            "wpli":           "yes" if check_wpli(paths, label) else "no",
            "wpli_date":      _newest_mtime(wpli_files),
            "pac":            "yes" if check_pac(paths, label) else "no",
            "pac_date":       _newest_mtime(pac_files),
            "psi":            "yes" if check_psi(paths, label) else "no",
            "psi_date":       _newest_mtime(psi_files),
            "mat":            "yes" if check_mat(paths, label) else "no",
        }
        rows.append(row)

        # Separate trial-count table — full breakdown per task, mirrors
        # epoch_trial_counts.tsv directly so numbers are never duplicated
        # or compressed/ambiguous.
        for task in TASKS:
            tc = trial_counts.get((label, task))
            trial_rows.append({
                "idx":            idx,
                "subject":        label,
                "task":           task,
                "n_total":        tc.get("n_total", "-") if tc else "-",
                "n_after_reject": tc.get("n_after_reject", "-") if tc else "-",
                "n_perceived":    tc.get("n_perceived", "-") if tc else "-",
                "n_nonperceived": tc.get("n_nonperceived", "-") if tc else "-",
                "n_miss":         tc.get("n_miss", "-") if tc else "-",
                "pct_rejected":   tc.get("pct_rejected", "-") if tc else "-",
                "pct_perceived":  tc.get("pct_perceived", "-") if tc else "-",
            })

    # --- Summary rows: totals + percentages across all subjects ---
    def _to_int(v):
        try:
            return int(v)
        except (ValueError, TypeError):
            return None

    def _summary_row(task_filter, label_str):
        relevant = [tr for tr in trial_rows
                    if task_filter is None or tr["task"] == task_filter]
        totals = {"n_total": 0, "n_after_reject": 0, "n_perceived": 0,
                   "n_nonperceived": 0, "n_miss": 0}
        any_valid = False
        for tr in relevant:
            for key in totals:
                v = _to_int(tr.get(key))
                if v is not None:
                    totals[key] += v
                    any_valid = True
        if not any_valid:
            return None
        n_total = totals["n_total"]
        pct_rejected  = 100 * (n_total - totals["n_after_reject"]) / n_total if n_total else 0
        pct_perceived = 100 * totals["n_perceived"] / n_total if n_total else 0
        return {
            "idx": "", "subject": label_str, "task": "(all)" if task_filter is None else task_filter,
            "n_total":        totals["n_total"],
            "n_after_reject": totals["n_after_reject"],
            "n_perceived":    totals["n_perceived"],
            "n_nonperceived": totals["n_nonperceived"],
            "n_miss":         totals["n_miss"],
            "pct_rejected":   round(pct_rejected, 1),
            "pct_perceived":  round(pct_perceived, 1),
        }

    summary_rows = []
    for task in TASKS:
        r = _summary_row(task, f"SUM ({task})")
        if r:
            summary_rows.append(r)
    grand_total = _summary_row(None, "SUM (all tasks)")
    if grand_total:
        summary_rows.append(grand_total)

    with open(out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    print(f"Saved: {out_tsv}")
    print(f"Subjects: {len(rows)}")

    if args.html:
        out_html = out_dir / "pipeline_status.html"
        write_html_dashboard(rows, trial_rows + summary_rows, out_html)
        print(f"Saved: {out_html}")

    if args.latex:
        def mark(v):
            return r"\checkmark" if v == "yes" else "-"

        def mark_na(v):
            return r"\checkmark" if v == "yes" else "n/a"

        def mark_mri(v):
            if v == "individual":
                return r"\checkmark"
            if v == "scaled-fsaverage":
                return "scaled"
            if v == "fsaverage":
                return "fsavg"
            return "-"

        print()
        print("% --- Pipeline overview (merged, pipeline order) ---")
        print(r"\begin{longtable}{l|l|c|c|c|c|c|c|c|c|c|c|c|c|c}")
        print(r"Nr & Subject & Info & .mat & TrigJSON & Bads & Prep & Epoch & "
              r"MRI & Coreg & BEM & Source & Parcellation & WPLI & PAC & PSI \\ \hline")
        for row in rows:
            print(
                f"{row['idx']} & {row['subject']} & {mark(row['info'])} & {mark(row['mat'])} & "
                f"{mark_na(row['trigcheck_json'])} & {mark(row['bads'])} & "
                f"{mark(row['prep'])} & {mark(row['epoch'])} & "
                f"{mark_mri(row['mri'])} & {mark(row['coreg'])} & "
                f"{mark(row['bem'])} & {mark(row['source'])} & "
                f"{mark(row['parcellation'])} & {mark(row['wpli'])} & "
                f"{mark(row['pac'])} & {mark(row['psi'])} " + r"\\"
            )
        print(r"\end{longtable}")

        print()
        print("% --- Trial counts ---")
        print(r"\begin{longtable}{l|l|l|c|c|c|c|c|c|c}")
        print(r"Nr & Subject & Task & n\_total & n\_after\_reject & "
              r"n\_perceived & n\_nonperceived & n\_miss & "
              r"\%rejected & \%perceived \\ \hline")
        for tr in trial_rows + summary_rows:
            print(
                f"{tr['idx']} & {tr['subject']} & {tr['task']} & {tr['n_total']} & "
                f"{tr['n_after_reject']} & {tr['n_perceived']} & "
                f"{tr['n_nonperceived']} & {tr['n_miss']} & "
                f"{tr['pct_rejected']} & {tr['pct_perceived']} " + r"\\"
            )
        print(r"\end{longtable}")


if __name__ == "__main__":
    main()
