"""
support_bank.py — the labelled reference notes that TC-WPN builds prototypes from.

WHY THIS FILE EXISTS
====================
TC-WPN (called `c3_clinical_nlp` in this backend) is a PROTOTYPICAL NETWORK. It
stores no decision boundary. On every request it embeds a set of labelled
reference notes, builds one weighted centroid per class from them, and
classifies the query note by cosine distance to those centroids. The centroids
are built per request and discarded.

So the support set is not context, not metadata, and not extra evidence. It IS
the classifier's decision boundary, materialised at call time. With zero support
notes there is no boundary and nothing to compute — TC-WPN returns
422 MISSING_SUPPORT_SET rather than a score.

WHO SUPPLIES IT
===============
    Clinician  --[ subject_id, note_text ]-->  THIS BACKEND  --[ note + support_set ]-->  TC-WPN

The clinician sends only the note. She never sees this bank, never picks from
it, and there is no screen that asks a psychiatrist to choose reference notes.
`ClinicalNote.support_set` on the ingestion endpoint stays in the schema for
tests and for manual override, but in normal operation it is empty and this
module fills it.

Before this module existed, an empty support_set was forwarded verbatim. The
old TC-WPN Space then silently substituted two hardcoded demo notes and said
nothing, so every note in the system was scored against the same unadapted pair.
That is the failure this file prevents.

THE SAME NOTES EVERY TIME
=========================
This is the part that trips people up: the bank does NOT vary per patient. The
same ~5 notes are sent for P001, P002 and P500. It changes only when a new
`bank_version` is deliberately published.

Two properties follow, and both are enforced below:

  1. DETERMINISTIC SELECTION. The same query note scored against a different
     support set gives a different number — that is inherent to the model, not a
     bug. If selection were random per call, a clinician reopening yesterday's
     note would see a different score and would stop trusting the number.
     ORDER BY note_id, never ORDER BY RANDOM().

  2. VERSION PINNING. `bank_version` is echoed by TC-WPN as
     `support_set_version` and lands in ModalityReading.detail. Without it a
     stored reading cannot be reproduced, which is an audit failure an ethics
     committee will find.

PATIENT EXCLUSION
=================
src/tcwpn/sampler.py guarantees support ∩ query patients = ∅ in every training
episode, and collate.py raises on overlap. NOTHING RAISES AT SERVING TIME. If a
note belonging to the queried subject ends up in the support set, the reading is
leaked and no error will say so. `select_support_set` filters on
`source_subject_id`; it is a no-op for authored notes (which have none) and
becomes load-bearing the day real NHSL notes are added. It is written now so it
cannot be forgotten then.

K
=
The frozen TC-WPN benchmark is K=5 and only K=5 (AUROC 0.7377, SD 0.0031, five
seeds). Serving at another K is not wrong but is outside what was measured, so
the default here is 3 anxiety + 2 control. Note also that K support notes plus
the query means K+1 BERT passes on a 2-core CPU Space — raising K raises latency
roughly linearly.

MIMIC-IV
========
See ON_MIMIC below before adding any MIMIC-derived note to this file.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from typing import Optional

from sqlalchemy import select

from db_models import AuditLog, SupportBankNote

# Published bank. Bump this string when the notes change; never edit notes in
# place under an existing version, or historical readings silently change
# meaning.
SEED_BANK_VERSION = "synthetic-v1"

DEFAULT_K_ANXIETY = 3
DEFAULT_K_CONTROL = 2


class SupportBankUnavailable(RuntimeError):
    """Raised when no usable support set can be assembled.

    Deliberately fatal for the request rather than fallback-to-anything. A
    prediction built on a support set we cannot describe is worse than no
    prediction: the caller records status='error' and the clinician timeline
    shows a gap, which is honest.
    """


# =============================================================================
# ON_MIMIC — read before adding MIMIC-derived notes
# =============================================================================
ON_MIMIC = """
MIMIC-IV note text must NOT be committed to this repository.

This repo is public (it can be cloned without authentication). The PhysioNet
credentialed-user Data Use Agreement restricts sharing MIMIC content with people
who have not completed the training and signed the DUA themselves. Pasting
discharge summaries into a public GitHub repo distributes them to everyone, and
that is not a grey area — it puts the individual's credentialing and the
institution's PhysioNet access at risk.

If a MIMIC-derived bank is wanted, load it at runtime from a path that never
enters git:

    SUPPORT_BANK_SEED_FILE=/secure/local/path/mimic_bank.json
    SUPPORT_BANK_VERSION=mimic-train-v1

`seed_support_bank()` reads that file when the variable is set. The path belongs
on credentialed infrastructure only, and .gitignore must cover it.

If you do build one, draw notes from the TRAIN split only. Using val or test
notes as support means demonstrating on the same data the reported AUROC was
measured from.

For the demo, the authored notes below avoid the question entirely.
""".strip()


# =============================================================================
# SEED NOTES — authored exemplars, not patient records
# =============================================================================
# These are ILLUSTRATIVE notes written to resemble the register of a clinical
# note. They describe no real person and are derived from no patient record, so
# they carry no DUA or ethics constraint and can be committed, demonstrated and
# printed in an appendix.
#
# They are NOT clinically validated. Dr. Suraweera should review and replace
# them; when she does, publish the result as a new bank_version (e.g.
# "nhsl-reviewed-v1") rather than editing these in place.
#
# One honest limitation to state in the paper: these are out of distribution
# relative to the MIMIC-IV discharge summaries TC-WPN was meta-trained on, so
# the score distribution will differ from the benchmark. That is a property of
# the demo, not of the model.
#
# days_before_index = (that note's patient's index time - note charttime), in
# days, >= 0. It drives w_i^T = exp(-0.5 * days / 365). For authored notes the
# value is a chosen spread, not a measurement. Note that Phase 4 found the
# temporal mechanism has no detectable effect (tcwpn_full vs aux_only: +0.0006,
# p=0.886), so these values are not doing quiet work behind the score.

SEED_NOTES: list[dict] = [
    # ── anxiety ──────────────────────────────────────────────────────────────
    {
        "note_id": "syn-anx-01",
        "label": "anxiety",
        "days_before_index": 0.0,
        "note_text": (
            "Psychiatry review. 34-year-old female describing excessive worry across "
            "multiple domains including work performance, family health and finances, "
            "present most days for approximately ten months. Reports the worry is "
            "difficult to control once it starts. Associated restlessness, muscle "
            "tension in neck and shoulders, early insomnia with sleep onset latency of "
            "one to two hours, and reduced concentration at work. Denies panic attacks. "
            "No current suicidal ideation. GAD-7 score 15. Commenced sertraline 50mg "
            "daily with plan to titrate. Referred for cognitive behavioural therapy. "
            "Impression: generalised anxiety disorder."
        ),
    },
    {
        "note_id": "syn-anx-02",
        "label": "anxiety",
        "days_before_index": 45.0,
        "note_text": (
            "Outpatient follow-up. 27-year-old male with recurrent unexpected episodes "
            "of intense fear peaking within minutes, accompanied by palpitations, chest "
            "tightness, dyspnoea, sweating and a fear of losing control. Episodes "
            "occurring two to three times per week over the past six months. Has begun "
            "avoiding public transport and crowded shops for fear of a further episode. "
            "Cardiac workup previously unremarkable including ECG and troponin. "
            "Escitalopram 10mg daily continued. Breathing retraining discussed. "
            "Impression: panic disorder with early agoraphobic avoidance."
        ),
    },
    {
        "note_id": "syn-anx-03",
        "label": "anxiety",
        "days_before_index": 120.0,
        "note_text": (
            "Nursing assessment, medical ward. Patient appears visibly distressed and "
            "restless overnight, repeatedly asking staff for reassurance about test "
            "results. Reports racing thoughts and inability to settle. Observed pacing "
            "the corridor at 03:00. Declined night sedation. Documented history of "
            "anxiety with previous psychology input. Reports similar episodes during "
            "prior admissions. Psychiatry liaison referral placed. Reassurance and "
            "orientation provided; patient settled towards morning."
        ),
    },
    {
        "note_id": "syn-anx-04",
        "label": "anxiety",
        "days_before_index": 210.0,
        "note_text": (
            "Discharge summary, psychiatric admission. 41-year-old male admitted "
            "following escalating anxiety symptoms with functional decline over three "
            "months. On admission described persistent apprehension, hypervigilance, "
            "irritability and marked sleep disturbance. Unable to attend work for six "
            "weeks prior to admission. Trial of venlafaxine XR titrated to 150mg daily "
            "with partial response. Engaged with ward psychology sessions. GAD-7 "
            "reduced from 18 on admission to 11 at discharge. Discharged with community "
            "mental health team follow-up in two weeks."
        ),
    },
    {
        "note_id": "syn-anx-05",
        "label": "anxiety",
        "days_before_index": 365.0,
        "note_text": (
            "Social work note. Patient discussed ongoing difficulty leaving the house "
            "for appointments and describes intense apprehension in social settings, "
            "particularly where she may be observed or judged. Reports this has limited "
            "her employment options for several years. Avoids family gatherings. "
            "Describes physical symptoms of blushing, tremor and nausea prior to social "
            "contact. Currently under psychiatric follow-up for anxiety. Discussed "
            "graded exposure and referral to a supported employment programme."
        ),
    },

    # ── control ──────────────────────────────────────────────────────────────
    {
        "note_id": "syn-ctl-01",
        "label": "control",
        "days_before_index": 10.0,
        "note_text": (
            "Discharge summary. Patient admitted for elective laparoscopic "
            "cholecystectomy following recurrent right upper quadrant pain after fatty "
            "meals. Ultrasound confirmed cholelithiasis without duct dilatation. "
            "Procedure completed without complication. Tolerating diet post-operatively. "
            "Pain controlled with simple analgesia. No psychiatric history. Discharged "
            "day one post-operative with surgical outpatient follow-up in six weeks."
        ),
    },
    {
        "note_id": "syn-ctl-02",
        "label": "control",
        "days_before_index": 60.0,
        "note_text": (
            "Routine outpatient review, endocrinology. 58-year-old male with type 2 "
            "diabetes mellitus of eight years duration. HbA1c 7.2 percent, improved from "
            "8.1 percent at last review. Reports good adherence to metformin 1g twice "
            "daily. Weight stable. No hypoglycaemic episodes. Feet examined, sensation "
            "intact, no ulceration. Retinal screening up to date. Mood described as "
            "stable, no complaints. Continue current regimen; review in six months."
        ),
    },
    {
        "note_id": "syn-ctl-03",
        "label": "control",
        "days_before_index": 150.0,
        "note_text": (
            "Orthopaedic clinic note. 45-year-old female reviewed six weeks following "
            "open reduction and internal fixation of a distal radius fracture sustained "
            "in a fall. Radiographs demonstrate satisfactory alignment with early callus "
            "formation. Wound healed. Range of motion improving with physiotherapy. "
            "Grip strength returning. Patient reports she is coping well and has "
            "returned to light duties at work. Continue physiotherapy; review in eight "
            "weeks."
        ),
    },
    {
        "note_id": "syn-ctl-04",
        "label": "control",
        "days_before_index": 240.0,
        "note_text": (
            "Nursing note, surgical ward. Observations stable throughout the shift. "
            "Temperature 36.8, blood pressure 124/78, heart rate 72, oxygen saturation "
            "98 percent on room air. Patient mobilising independently to the bathroom. "
            "Diet and fluids taken well. Wound dressing dry and intact. Slept well "
            "overnight without analgesia. No concerns raised by patient or family. "
            "Awaiting consultant review for discharge planning."
        ),
    },
    {
        "note_id": "syn-ctl-05",
        "label": "control",
        "days_before_index": 400.0,
        "note_text": (
            "Respiratory outpatient letter. Patient reviewed for follow-up of "
            "community-acquired pneumonia treated three months previously. Symptoms "
            "fully resolved. Repeat chest radiograph shows complete radiological "
            "clearance. Spirometry within normal limits for age and height. No ongoing "
            "cough or breathlessness. Exercise tolerance returned to baseline. No "
            "further respiratory follow-up required; discharged back to primary care."
        ),
    },
]


# =============================================================================
# SEEDING
# =============================================================================
def _load_seed_notes() -> tuple[str, list[dict]]:
    """Return (bank_version, notes).

    Reads SUPPORT_BANK_SEED_FILE when set, so a MIMIC-derived or NHSL bank can
    be supplied from outside the repository. See ON_MIMIC. Falls back to the
    authored notes above.

    Expected JSON shape:
        {"bank_version": "...",
         "notes": [{"note_id","label","note_text","days_before_index",
                    "source_subject_id"?, "provenance"?}, ...]}
    """
    path = os.getenv("SUPPORT_BANK_SEED_FILE")
    if not path:
        return os.getenv("SUPPORT_BANK_VERSION", SEED_BANK_VERSION), SEED_NOTES

    with open(path, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    version = (os.getenv("SUPPORT_BANK_VERSION")
               or blob.get("bank_version")
               or "external-v1")
    notes = blob.get("notes") or []
    if not notes:
        raise SupportBankUnavailable(f"{path} contains no notes")
    return version, notes


def seed_support_bank(db, *, force: bool = False) -> int:
    """Idempotent. Inserts the seed bank if that version is not already present.

    Safe to call on every startup — the existence check keeps it cheap and stops
    a restart from duplicating rows. `force=True` deletes and reinserts the
    version, which is for development only: doing it in deployment silently
    changes the meaning of every reading already stamped with that version.

    Returns the number of rows inserted (0 when already seeded).
    """
    version, notes = _load_seed_notes()

    existing = db.scalar(
        select(SupportBankNote).where(SupportBankNote.bank_version == version).limit(1))
    if existing and not force:
        return 0
    if existing and force:
        for row in db.scalars(
                select(SupportBankNote).where(SupportBankNote.bank_version == version)):
            db.delete(row)
        db.flush()

    default_prov = ("authored exemplar, not a patient record — pending clinical "
                    "review by Dr. C. Suraweera")
    inserted = 0
    for n in notes:
        if n.get("label") not in ("anxiety", "control"):
            raise SupportBankUnavailable(
                f"note {n.get('note_id')!r} has label {n.get('label')!r}; "
                "must be 'anxiety' or 'control'")
        db.add(SupportBankNote(
            bank_version=version,
            note_id=str(n["note_id"]),
            label=n["label"],
            note_text=n["note_text"],
            days_before_index=float(n.get("days_before_index", 0.0)),
            source_subject_id=n.get("source_subject_id"),
            provenance=n.get("provenance", default_prov),
            active=True,
        ))
        inserted += 1

    db.add(AuditLog(subject_id=None, event="support_bank.seeded", actor="system",
                    detail={"bank_version": version, "notes": inserted,
                            "source": os.getenv("SUPPORT_BANK_SEED_FILE", "builtin")}))
    db.commit()
    return inserted


# =============================================================================
# SELECTION
# =============================================================================
def select_support_set(db, *, subject_id: str, bank_version: str,
                       k_anxiety: int = DEFAULT_K_ANXIETY,
                       k_control: int = DEFAULT_K_CONTROL) -> list[dict]:
    """Build the support_set payload for one TC-WPN call.

    Deterministic: ORDER BY note_id. The same subject on the same bank version
    gets byte-identical notes every time, so re-scoring a note reproduces the
    score exactly.

    Excludes notes whose `source_subject_id` matches the queried subject. A
    no-op for authored notes; load-bearing once real notes are added.

    Raises SupportBankUnavailable rather than returning a short or one-sided
    set — TC-WPN needs at least one of each class and there is no sensible
    partial answer.
    """
    def _take(label: str, k: int) -> list[SupportBankNote]:
        stmt = (select(SupportBankNote)
                .where(SupportBankNote.bank_version == bank_version,
                       SupportBankNote.label == label,
                       SupportBankNote.active.is_(True))
                .order_by(SupportBankNote.note_id))
        rows = list(db.scalars(stmt))
        rows = [r for r in rows
                if not (r.source_subject_id and r.source_subject_id == subject_id)]
        return rows[:k]

    anx = _take("anxiety", k_anxiety)
    ctl = _take("control", k_control)

    if not anx or not ctl:
        raise SupportBankUnavailable(
            f"bank '{bank_version}' yields {len(anx)} anxiety and {len(ctl)} control "
            f"notes after excluding subject {subject_id}; TC-WPN requires at least "
            "one of each. Seed the bank or publish a usable version."
        )

    return [{
        "id": r.note_id,
        "text": r.note_text,
        "label": r.label,
        # The field TC-WPN actually wants. Supplying it gives
        # temporal_axis='backend_supplied'; omitting it makes the service fall
        # back to note_date arithmetic and report 'approximated'.
        "days_before_index": float(r.days_before_index),
    } for r in anx + ctl]


def describe_bank(db, bank_version: str) -> dict:
    """Small summary for /health and for the clinician app's settings screen."""
    rows = list(db.scalars(
        select(SupportBankNote).where(SupportBankNote.bank_version == bank_version,
                                      SupportBankNote.active.is_(True))))
    return {
        "bank_version": bank_version,
        "n_anxiety": sum(1 for r in rows if r.label == "anxiety"),
        "n_control": sum(1 for r in rows if r.label == "control"),
        "k_sent_per_request": DEFAULT_K_ANXIETY + DEFAULT_K_CONTROL,
        "evaluated_k": 5,
        "provenance": sorted({r.provenance for r in rows})[:3],
        "clinically_reviewed": not bank_version.startswith("synthetic"),
    }
