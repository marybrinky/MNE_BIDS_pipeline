# Neural Signatures of Pain — MEG Pipeline (12 June 2026)

**MD Thesis project** | Marie Brinkmann | MEG Laboratory, Sektion Biomagnetismus,
Neurologische Klinik, Universitätsklinikum Heidelberg, Germany

Supervisor: PD Dr André Rupp ([@ruppomat](https://github.com/ruppomat))

> This pipeline is adapted from the original MEG analysis framework by
> PD Dr André Rupp. The original structure, trigger decoding, and BIDS layout
> were designed by him. Adaptations for this project include a custom trigger
> verification workflow for hardware-related trigger ambiguities, per-subject
> JSON sidecar files for stimulus onset correction, and extensions for
> mechanical pain stimuli (pinprick, tactile).
> This readme and code was written with help of Claude AI by Anthropic.

---

## Study Overview

This project investigates the **neural signatures of laser-evoked and
mechanical pain** in healthy volunteers using whole-head MEG. The aim is
to identify functional connectivity patterns within the cortical pain matrix
that distinguish pain quality and intensity across three stimulation
modalities: CO₂ laser (radiant heat pain), pinprick, and tactile control.

The primary analysis is **weighted phase-lag index (WPLI)** computed from
source-reconstructed MEG data, with regions of interest drawn from the
Destrieux and Desikan–Killiany atlases.

**22 healthy volunteers** | **Neuromag TRIUX** (Heidelberg) |
**3 tasks × 50 stimuli each**

---

## Repository Structure

```
code/                        ← all analysis scripts (this repo)
├── core.py                  ← shared constants, Paths, trigger decoding
├── create_bids_structure.py ← scaffold a new BIDS project folder
├── preprocess.py            ← bandpass filtering + ICA artefact rejection
├── epoch.py                 ← trigger decoding, epoching, trial rejection
├── source.py                ← BEM, forward model, dSPM inverse, ROI extraction
├── wpli.py                  ← weighted phase-lag index (WPLI)
├── plot_wpli.py             ← WPLI visualisation
├── pac.py                   ← phase–amplitude coupling
├── pac_group.py             ← group-level PAC
├── plot_pac.py              ← PAC visualisation
├── match_ratings.py         ← legacy: match fif-decoded ratings to mat file
├── open_mat.py              ← interactive mat file browser / renamer
├── inspect_raw.py           ← raw data quality inspection
├── inspect_triggers.py      ← trigger sequence viewer
├── browse_source.py         ← interactive source estimate viewer
└── environment.yml          ← conda environment specification
```

The project data (not included here) follows BIDS convention:

```
project_root/
├── rawdata/                      ← raw .fif files per subject/task
│   └── sub-{label}/beh/          ← behavioural mat files (ratings)
├── derivatives/
│   ├── prep/                     ← preprocessed continuous data
│   ├── epochs/                   ← epoched data, perceived trials only
│   ├── source/                   ← forward models, inverse operators, STCs, ROIs
│   ├── connectivity/             ← WPLI results
│   ├── trigger_check/            ← per-subject trigger verification JSON sidecars
│   ├── freesurfer/               ← FreeSurfer recon-all output
│   └── logs/
│       └── trial_counts/         ← per-subject and group trial count statistics
└── code/                         ← this repository
```

---

## Installation

```bash
git clone https://github.com/<your-username>/neural-signatures-of-pain.git
cd neural-signatures-of-pain

conda env create -f environment.yml
conda activate mne110
```

**Requirements:** Python 3.11, MNE-Python 1.10, NumPy, SciPy, Pandas,
Matplotlib. All pinned in `environment.yml`.

---

## Usage

All scripts share a common interface:

```bash
# Preprocess all subjects
python code/preprocess.py --root $MEGROOT

# Preprocess specific subjects and tasks
python code/preprocess.py --root $MEGROOT --subjects 1409 3691 --tasks laser

# Epoch — perceived trials only (default)
python code/epoch.py --root $MEGROOT --overwrite

# Epoch specific subjects
python code/epoch.py --root $MEGROOT --subjects 1409 --tasks laser --overwrite

# Epoch all trials including non-perceived (opt-in)
python code/epoch.py --root $MEGROOT --overwrite --all-trials

# Source reconstruction
python code/source.py --root $MEGROOT --subjects 1409

# Connectivity (WPLI)
python code/wpli.py --root $MEGROOT --subjects 1409 --tasks laser

# Inspect raw triggers
python code/inspect_triggers.py --root $MEGROOT --subjects 1409
```

Set the project root once:

```bash
export MEGROOT=/path/to/your/project
```

---

## Epoching and Trial Selection

`epoch.py` performs trigger decoding, amplitude-based rejection, and
perceived-trial filtering in one step. **Perceived trials (rating > 0) are
the default** — only trials where the subject reported a sensation are saved
to disk. All downstream analyses (WPLI, source, PAC) automatically operate
on perceived trials only. Use `--all-trials` to include non-perceived trials
for specific comparisons.

### Rating source priority

For each subject and task, ratings are read in this order:

1. **Triggercheck JSON** (`derivatives/trigger_check/sub-{label}/`) — used
   when present; contains corrected trial indexing and `intensity_fif` per
   bundle
2. **Behavioural mat file** (`rawdata/sub-{label}/beh/`) — fallback for all
   other subjects

### Trial count statistics

After each run, two TSV files are saved to `derivatives/logs/trial_counts/`:

- `epoch_trial_counts.tsv` — per subject/task: `n_total`, `n_after_reject`,
  `n_after_perceived`, `n_rejected`, `n_nonperceived`, `pct_rejected`,
  `pct_final`
- `epoch_trial_counts_group.tsv` — group averages per task

---

## Trigger Decoding

The Heidelberg Neuromag/TRIUX encodes stimuli across 6 binary STI channels
(STI 001–006). Each channel contributes one bit to a composite trigger code.
This pipeline decodes all channels simultaneously and combines simultaneous
activations via bitwise OR, replicating the behaviour of the MATLAB
`loadmeg.m` function (Riedel, Heidelberg).

### Sub-combination fix (`core.py`)

When multiple STI channels fire within 1 sample of each other, the decoder
marks all contributing channel samples as consumed before processing lower-
order combinations. This prevents a hardware pulse on e.g. STI 001 + STI 004
(composite code 9) from also generating spurious single-channel events
(codes 1 and 8). This replicates MATLAB's `TRIGmaxDT = 1` sample grouping
exactly.

### Compound trigger subjects

For a subset of subjects the laser stimulus was accidentally routed through
STI channel 3 (bit 2 = code 4), which was shared with the response buttons.
This caused bit-overlap artefacts where the laser appeared as codes 3, 5, 6,
or 7 instead of 4. These subjects are handled via per-subject
**trigger verification JSON sidecars**:

```
derivatives/trigger_check/sub-{label}/sub-{label}_task-laser_triggercheck.json
```

Each sidecar stores all bundles in recording order, each with the exact
trigger codes observed in the fif (`triggers` array), a laser/non-laser flag,
and the intensity rating (`intensity_fif`). The pipeline matches this JSON
raster directly against the fif trigger stream in a single forward pass:
the full sequence of expected trigger codes is concatenated and scanned
against the fif, skipping dummy placeholders (code 123). This approach is
robust to extra noise triggers in the fif and ensures each epoch is
unambiguously linked to its correct bundle and rating.

Two subjects (4365, 5026) also have JSON sidecars for pinprick and tactile
tasks where triggers were lost or ambiguous.

---

## Adapting for a New Project

1. Edit `core.py` — set `PROJECT_NAME`, `TASKS`, `EPOCH_CONFIGS`,
   `ATLAS_CONFIGS`, and `SUBJECT_TRIGGERS` (trigger codes per subject)
2. Run `create_bids_structure.py` to scaffold the folder layout
3. Copy raw `.fif` files into `rawdata/sub-<label>/meg/`
4. Add FreeSurfer surfaces to `derivatives/freesurfer/sub-<label>/`
5. Place coregistration transforms in `derivatives/trans/`

The pipeline is designed to require changes **only in `core.py`** for most
adaptations — all other scripts read their configuration from there.

---

## Citation and Acknowledgements

If you use this pipeline, please acknowledge:

- The original pipeline framework by **PD Dr André Rupp**
  ([@ruppomat](https://github.com/ruppomat)),
  MEG Laboratory, Universitätsklinikum Heidelberg
- **MNE-Python**: Gramfort et al. (2013), *Frontiers in Neuroscience*
- **FreeSurfer**: Fischl et al. (2002), *Neuron*

---

## Contact

Questions, bug reports, and suggestions are very welcome — please open a
[GitHub Issue](../../issues) or reach out directly.

**Marie Brinkmann**
Promotionsstudentin, MEG Labor — Sektion Biomagnetismus
Neurologische Klinik, Universitätsklinikum Heidelberg
[@marybrinky](https://github.com/marybrinky)
