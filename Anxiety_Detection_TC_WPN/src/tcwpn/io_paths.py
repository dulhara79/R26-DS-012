"""
io_paths.py — Tolerant MIMIC file resolution.

Kaggle datasets do not agree on how MIMIC is laid out. The same logical file
turns up as any of:

    hosp/patients.csv.gz
    hosp/patients.csv
    patients.csv.gz
    PATIENTS.csv.gz
    PATIENTS.csv/PATIENTS.csv        (a directory named like a file)

Rather than making every script guess, `resolve()` searches a base directory for
a file matching a set of candidate names, case-insensitively, at any depth, and
returns the first hit sorted for determinism. pandas reads gzip transparently
from the extension, so no decompression step is needed.

This module touches paths only. It never reads a note, never sees a label, and
therefore cannot influence the benchmark.
"""

from __future__ import annotations

from pathlib import Path


class MimicFileNotFound(FileNotFoundError):
    pass


def resolve(base, *candidates, max_depth: int = 4) -> Path:
    """
    Find the first file under `base` whose name matches one of `candidates`
    (case-insensitive). Raises MimicFileNotFound with a useful listing.

    >>> resolve(base, "patients.csv.gz", "patients.csv")
    """
    base = Path(base)
    if not base.exists():
        raise MimicFileNotFound(f"base path does not exist: {base}")

    wanted = {c.lower() for c in candidates}

    # Fast path: exact relative hits first, in the order given.
    for c in candidates:
        p = base / c
        if p.is_file():
            return p

    hits = []
    for depth in range(1, max_depth + 1):
        pattern = "/".join(["*"] * depth)
        for p in base.glob(pattern):
            if p.is_file() and p.name.lower() in wanted:
                hits.append(p)
        if hits:
            break

    if not hits:
        listing = sorted(str(p.relative_to(base)) for p in base.glob("*"))[:25]
        raise MimicFileNotFound(
            f"none of {sorted(wanted)} found under {base}.\n"
            f"Top level contains: {listing}"
        )
    return sorted(hits)[0]


def has(base, *candidates, max_depth: int = 4) -> bool:
    try:
        resolve(base, *candidates, max_depth=max_depth)
        return True
    except MimicFileNotFound:
        return False


# Canonical name sets, used by the scripts so the candidate lists stay in one place.
MIMIC4 = {
    "patients": ("hosp/patients.csv.gz", "hosp/patients.csv",
                 "patients.csv.gz", "patients.csv"),
    "admissions": ("hosp/admissions.csv.gz", "hosp/admissions.csv",
                   "admissions.csv.gz", "admissions.csv"),
    "diagnoses": ("hosp/diagnoses_icd.csv.gz", "hosp/diagnoses_icd.csv",
                  "diagnoses_icd.csv.gz", "diagnoses_icd.csv"),
    "discharge": ("note/discharge.csv.gz", "note/discharge.csv",
                  "discharge.csv.gz", "discharge.csv"),
}

MIMIC3 = {
    "patients": ("PATIENTS.csv.gz", "PATIENTS.csv",
                 "patients.csv.gz", "patients.csv"),
    "admissions": ("ADMISSIONS.csv.gz", "ADMISSIONS.csv",
                   "admissions.csv.gz", "admissions.csv"),
    "diagnoses": ("DIAGNOSES_ICD.csv.gz", "DIAGNOSES_ICD.csv",
                  "diagnoses_icd.csv.gz", "diagnoses_icd.csv"),
    "noteevents": ("NOTEEVENTS.csv.gz", "NOTEEVENTS.csv",
                   "noteevents.csv.gz", "noteevents.csv"),
}
