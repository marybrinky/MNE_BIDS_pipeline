#!/usr/bin/env python3
"""
setup_parcellation.py
----------------------
Step 0 for all connectivity analyses (wpli.py, pac.py, psi.py).

Morphs the HCPMMP1 atlas from fsaverage onto each subject's own cortical
surface, producing lh.HCPMMP1.annot / rh.HCPMMP1.annot in each subject's
FreeSurfer label/ directory. Run once per subject before any of the
three connectivity scripts.

Usage
-----
    python code/setup_parcellation.py --root $MEGROOT --subjects 4382
    python code/setup_parcellation.py --root $MEGROOT             # all subjects
    python code/setup_parcellation.py --root $MEGROOT --overwrite
"""

import argparse
import sys
from pathlib import Path

from core import (
    ATLAS_CONFIGS,
    DEFAULT_ATLAS,
    Paths,
    load_subjects,
    setup_logging,
)
from connectivity_common import setup_parcellation

DEFAULT_ROOT = Path("/Volumes/ExtremePro/laser")


def main():
    parser = argparse.ArgumentParser(
        description="Morph HCPMMP1 parcellation from fsaverage to each "
                     "subject. Required once before wpli.py / pac.py / psi.py."
    )
    parser.add_argument("--root",      type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--subjects",  nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--atlas",     default=DEFAULT_ATLAS,
                         choices=list(ATLAS_CONFIGS.keys()))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths    = Paths(args.root)
    logger   = setup_logging(paths, "setup_parcellation")
    subjects = args.subjects if args.subjects else load_subjects(paths)

    logger.info("Subjects : %s", subjects)
    logger.info("Atlas    : %s", args.atlas)
    logger.info("Overwrite: %s", args.overwrite)

    n_ok = n_fail = 0
    for label in subjects:
        ok = setup_parcellation(paths, label, atlas_key=args.atlas,
                                 overwrite=args.overwrite, logger=logger)
        if ok: n_ok += 1
        else:  n_fail += 1

    logger.info("Done.  OK: %d  |  Failed: %d", n_ok, n_fail)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
