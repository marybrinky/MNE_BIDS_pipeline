#!/usr/bin/env python3
"""
inspect_triggers.py
-------------------
Read a fif file and print ALL trigger values in the order they appear.
No interpretation. No grouping. Just the raw sequence.

Displayed as blocks of 40 values (4 rows x 10 columns) so you can
visually scan for missing or wrong triggers.

Usage
-----
    python code/inspect_triggers.py --fif $MEGROOT/rawdata/sub-3691/meg/sub-3691_task-laser_meg.fif
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import mne
import numpy as np

# ---------------------------------------------------------------------------
# Trigger decoder  (exact core.py algorithm)
# ---------------------------------------------------------------------------

STI_CHANNELS  = [f"STI 00{i}" for i in range(1, 7)]
BIT_WEIGHTS   = np.array([2**i for i in range(6)], dtype=np.int32)
TOLERANCE_SMP = 1
MIN_DURATION  = 0.002
SHORTEST_EVT  = 1


def _events_per_channel(raw):
    result = []
    for ch in STI_CHANNELS:
        if ch not in raw.ch_names:
            result.append(np.array([], dtype=np.int32))
            continue
        try:
            evs = mne.find_events(
                raw, stim_channel=ch,
                min_duration=MIN_DURATION,
                shortest_event=SHORTEST_EVT,
                verbose=False,
            )[:, 0].astype(np.int32)
        except Exception:
            evs = np.array([], dtype=np.int32)
        if len(evs) > 1:
            too_close = np.where(np.diff(evs) <= 1)[0]
            if len(too_close):
                evs = np.delete(evs, too_close)
        result.append(evs)
    return result


def _expand_with_tolerance(samples, tol=TOLERANCE_SMP):
    if len(samples) == 0:
        return np.array([], dtype=np.int32)
    offsets  = np.arange(-tol, tol + 1, dtype=np.int32)
    expanded = (samples[:, np.newaxis] + offsets).ravel()
    return np.unique(expanded).astype(np.int32)


def _remove_duplicates(samples, tol=TOLERANCE_SMP):
    if len(samples) == 0:
        return samples
    keep = np.ones(len(samples), dtype=bool)
    for i in range(1, len(samples)):
        if samples[i] - samples[i - 1] <= tol:
            keep[i] = False
    return samples[keep]


def _canonical_sample(candidates, channel_samples):
    hits = []
    lo, hi = candidates.min(), candidates.max()
    for ch_smp in channel_samples:
        hits.extend(ch_smp[(ch_smp >= lo) & (ch_smp <= hi)].tolist())
    return int(np.median(hits)) if hits else int(candidates[0])


def decode_triggers(raw):
    """Return (times_ms, codes) using exactly the core.py algorithm."""
    if not raw.preload:
        raw.load_data(verbose=False)
    sfreq     = raw.info["sfreq"]
    ch_events = _events_per_channel(raw)
    ch_exp    = [_expand_with_tolerance(e) for e in ch_events]

    found_samples = set()
    event_list    = []
    active        = [i for i, e in enumerate(ch_events) if len(e) > 0]

    for n_bits in range(len(active), 0, -1):
        for combo in combinations(active, n_bits):
            intersection = ch_exp[combo[0]].copy()
            for i in combo[1:]:
                intersection = np.intersect1d(intersection, ch_exp[i])
            if len(intersection) == 0:
                continue
            intersection = np.array(
                [s for s in intersection if s not in found_samples],
                dtype=np.int32,
            )
            if len(intersection) == 0:
                continue
            intersection = _remove_duplicates(np.sort(intersection))
            code = int(sum(BIT_WEIGHTS[i] for i in combo))
            for smp in intersection:
                if smp not in found_samples:
                    window = np.arange(smp - TOLERANCE_SMP,
                                       smp + TOLERANCE_SMP + 1, dtype=np.int32)
                    canon  = _canonical_sample(
                        window, [ch_events[i] for i in combo])
                    event_list.append((canon, code))
                    for s in range(smp - TOLERANCE_SMP,
                                   smp + TOLERANCE_SMP + 1):
                        found_samples.add(s)

    if not event_list:
        return np.array([]), np.array([], dtype=int)

    arr      = np.array(sorted(event_list), dtype=np.int32)
    times_ms = (arr[:, 0] / sfreq) * 1000.0
    codes    = arr[:, 1].astype(int)
    return times_ms, codes


# ---------------------------------------------------------------------------
# Print all triggers as 4-row blocks of 10
# ---------------------------------------------------------------------------

def print_all_triggers(times_ms, codes, cols=10):
    """Print every trigger in order, reading DOWN each column.

    Consecutive triggers are stacked vertically in the same column.
    Each block is 4 rows x cols columns = cols*4 triggers.
    The index on the left shows the absolute index of the first
    trigger in that row.

    Example with cols=10, triggers 0..39:
        col:     1    2    3  ...  10
        row 1:   0    4    8  ...  36     <- triggers 0,4,8,...
        row 2:   1    5    9  ...  37     <- triggers 1,5,9,...
        row 3:   2    6   10  ...  38     <- triggers 2,6,10,...
        row 4:   3    7   11  ...  39     <- triggers 3,7,11,...
    """
    N          = len(codes)
    cw         = 5   # column width
    lw         = 8   # index label width
    block_size = cols * 4

    for block_start in range(0, N, block_size):
        block_codes = codes[block_start:min(block_start + block_size, N)]
        # Pad to full block so indexing is clean
        padded = list(block_codes) + [None] * (block_size - len(block_codes))

        print()
        for row in range(4):
            # The triggers in this row are at positions row, row+4, row+8, ...
            # i.e. block_start + row + col*4  for col in 0..cols-1
            abs_idx = block_start + row   # absolute index of first value in row
            line = f"  {abs_idx:>{lw}}  "
            for col in range(cols):
                pos = col * 4 + row
                val = padded[pos]
                if val is None:
                    line += f"{'':>{cw}}"
                else:
                    line += f"{val:>{cw}}"
            print(line)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Print all fif trigger values as 4-row blocks of 10.",
    )
    parser.add_argument("--fif", type=Path, required=True,
                        help="Path to raw fif file")
    parser.add_argument("--cols", type=int, default=10,
                        help="Values per row (default 10)")
    args = parser.parse_args()

    if not args.fif.exists():
        print(f"ERROR: {args.fif} not found")
        sys.exit(1)

    print(f"\n  Loading {args.fif.name} ...")
    raw = mne.io.read_raw_fif(str(args.fif), preload=True, verbose=False)
    print(f"  sfreq = {raw.info['sfreq']} Hz")

    times_ms, codes = decode_triggers(raw)
    print(f"  {len(codes)} trigger events found\n")

    print("  All unique codes:")
    uq, cnt = np.unique(codes, return_counts=True)
    for c, n in zip(uq, cnt):
        bits = [i for i in range(6) if c & (1 << i)]
        print(f"    code {c:3d}  (bits {bits})  ->  {n} events")

    print(f"\n  ── Trigger sequence (index on left, {args.cols} per row, 4 rows per block) ──")
    print_all_triggers(times_ms, codes, cols=args.cols)


if __name__ == "__main__":
    main()
