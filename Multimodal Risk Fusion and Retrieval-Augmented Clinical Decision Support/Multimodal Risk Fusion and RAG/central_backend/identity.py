"""
Identity — the piece that makes patient separation real.

The problem this solves: the clinician app knows the patient as an MRN; the
patient app knows them as an app_user_id; the chest strap streams under yet
another device id. Nothing joins automatically. If you get this wrong, one
patient's physiology fuses with another patient's notes, which is the single
worst failure this system could have.

The solution, following the enrolment flow in the sequence diagram:

    1. Clinician enters the MRN in the clinician app.
    2. Backend stores ONLY sha256(MRN + pepper) and mints a random subject_id.
    3. Backend returns a short pairing code.
    4. Clinician reads the code to the patient.
    5. Patient types it into the patient app, which posts it with its app_user_id.
    6. Backend attaches app_user_id as a SECOND alias for the SAME subject_id.

From then on every modality writes against subject_id, and the raw MRN never
leaves the clinician's screen.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import secrets
import uuid

# Pepper: a secret that is NOT stored in the database. Without it, a stolen
# database cannot be brute-forced back to MRNs (an MRN has low entropy — a
# plain unsalted hash of it would fall to a dictionary attack in seconds).
MRN_PEPPER = os.getenv("MRN_PEPPER", "")

PAIRING_CODE_TTL_MINUTES = int(os.getenv("PAIRING_CODE_TTL_MINUTES", "30"))

# Unambiguous alphabet: no O/0, no I/1/L. The code is read aloud in a noisy ward.
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


class PepperNotConfigured(RuntimeError):
    pass


def hash_mrn(mrn: str) -> str:
    """sha256 HMAC of a normalised MRN under the pepper.

    HMAC rather than a plain hash+concat, because HMAC is the construction
    designed for keyed hashing and is not vulnerable to length-extension.
    """
    if not MRN_PEPPER:
        raise PepperNotConfigured(
            "MRN_PEPPER is not set. Refusing to hash patient identifiers with an "
            "empty key — set MRN_PEPPER in the environment before enrolling anyone."
        )
    normalised = mrn.strip().upper()
    if not normalised:
        raise ValueError("MRN is empty")
    return hmac.new(MRN_PEPPER.encode(), normalised.encode(), hashlib.sha256).hexdigest()


def new_subject_id() -> str:
    """Random, unguessable, carries no information about the patient."""
    return str(uuid.uuid4())


def new_pairing_code() -> str:
    """Format XXXX-XXXX, e.g. 4F2K-8Q1M. ~5.0e11 possibilities.

    Uses secrets, not random: this is a credential, and `random` is predictable
    from observed output.
    """
    raw = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(8))
    return f"{raw[:4]}-{raw[4:]}"


def pairing_expiry(now: dt.datetime | None = None) -> dt.datetime:
    now = now or dt.datetime.now(dt.timezone.utc)
    return now + dt.timedelta(minutes=PAIRING_CODE_TTL_MINUTES)


def is_expired(expires_at: dt.datetime, now: dt.datetime | None = None) -> bool:
    now = now or dt.datetime.now(dt.timezone.utc)
    if expires_at.tzinfo is None:                     # SQLite loses tzinfo on round-trip
        expires_at = expires_at.replace(tzinfo=dt.timezone.utc)
    return now > expires_at
