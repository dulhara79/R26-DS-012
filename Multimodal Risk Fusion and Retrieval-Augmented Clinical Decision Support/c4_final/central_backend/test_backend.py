"""
Central Backend validation suite.

Runs the whole sequence diagram against an in-memory database with stubbed
component Spaces, then asserts the properties that actually matter clinically.

    python test_backend.py

The most important test in this file is PATIENT SEPARATION (section 7). Everything
else is plumbing; that one is the safety property.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import tempfile

# Configure BEFORE importing the app — module-level config is read at import time.
_tmpdb = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdb.name}"
os.environ["MRN_PEPPER"] = "test-pepper-not-for-production"
os.environ["FUSION_MODE"] = "inprocess"
os.environ["FUSION_SERVICE_DIR"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "fusion_service"))

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import identity  # noqa: E402
import modality_clients as mc  # noqa: E402
from main import app  # noqa: E402

client = TestClient(app)

passed, failed = 0, 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")


def section(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


# ── stub the component Spaces so tests are deterministic and offline ─────────
_C1_SCORE = [41.8]
_C1_STATUS = ["ok"]      # ok | warming_up | error
_C3_SCORE = [0.68]
_C3_SUPPORT_K = [12]     # 0 -> no_support_set
_C4_SCORE = [0.55]


def stub_c1(subject_id, window=None, client=None):
    if _C1_STATUS[0] == "error":
        return mc.ComponentResult(status="error", note="timeout 30s (Space waking?)")
    if _C1_STATUS[0] == "warming_up":
        return mc.ComponentResult(status="warming_up", raw_score=None, confidence=0.6,
                                  coverage=0.3, model_version="c1-lstmae-v1.2.0",
                                  note="warming_up (300/1200 windows)")
    return mc.ComponentResult(raw_score=_C1_SCORE[0], status="ok", confidence=0.74,
                              coverage=0.81, model_version="c1-lstmae-v1.2.0")


def stub_c3(note_text, note_type="progress", anxiety_support=None,
            control_support=None, subject_external_id=None, client=None):
    if not note_text or not note_text.strip():
        return mc.ComponentResult(status="error", note="no clinical note supplied")
    if _C3_SUPPORT_K[0] == 0:
        return mc.ComponentResult(status="no_support_set", raw_score=None, confidence=0.5,
                                  coverage=1.0, model_version="TC-WPN-v1.0",
                                  note="no_support_set (support_k=0)")
    # mirrors the REAL C3 response: entropy present, so confidence is derived
    # from it rather than from C3's own score-shaped `confidence` field
    return mc.ComponentResult(raw_score=_C3_SCORE[0], status="ok",
                              confidence=mc.confidence_from_entropy(0.6331) or 0.5,
                              coverage=1.0, model_version="TC-WPN-v1.0")


def stub_c2(subject_external_id, payload=None, client=None):
    """Mirrors the REAL C2 Vercel response verbatim — including the trap: an
    experimental behavioral_vulnerability_score sitting next to score=null."""
    return mc.ComponentResult(
        raw_score=None, status="not_validated", confidence=0.0,
        coverage=0.9642857142857143, model_version="M2_mobile_screen_location_v1",
        note="fusion_eligible=false — excluded from composite.",
        detail={"subject_id": subject_external_id, "modality": "c2_behavioral",
                "score": None, "status": "not_validated", "fusion_eligible": False,
                "behavioral_vulnerability_score": 0.025467511026434304,
                "validation_status": "experimental"})


def stub_c4(subject_id, demographics, client=None):
    return mc.ComponentResult(raw_score=_C4_SCORE[0], status="ok", confidence=0.61,
                              coverage=1.0, model_version="dcar-v1.0")


mc.call_c1, mc.call_c2, mc.call_c3, mc.call_c4 = stub_c1, stub_c2, stub_c3, stub_c4
import main  # noqa: E402
main.mc.call_c1, main.mc.call_c2, main.mc.call_c3, main.mc.call_c4 = (
    stub_c1, stub_c2, stub_c3, stub_c4)


# ═════════════════════════════════════════════════════════════════════════════
section("1 · Health and configuration")
h = client.get("/health").json()
print(f"  fusion_mode        : {h['fusion_mode']}")
print(f"  mrn_pepper_set     : {h['mrn_pepper_set']}")
print(f"  excluded modalities: {h['gate']['excluded']}")
check("service starts", h["status"] == "ok")
check("MRN pepper configured", h["mrn_pepper_set"] is True)
check("c2 excluded by rule", h["gate"]["excluded"] == ["c2_behavioral"])


# ═════════════════════════════════════════════════════════════════════════════
section("2 · Enrolment and pairing (steps 1-9)")
r = client.post("/v1/subjects", json={"mrn": "NHSL-2026-0142", "enrolled_by": "dr.perera"})
check("enrolment returns 200", r.status_code == 200, r.text)
enrol = r.json()
P1 = enrol["subject_id"]
code1 = enrol["pairing_code"]
print(f"  subject_id   : {P1}")
print(f"  pairing_code : {code1}")
check("subject_id is a UUID (no MRN inside)", len(P1) == 36 and "NHSL" not in P1)
check("pairing code format XXXX-XXXX", len(code1) == 9 and code1[4] == "-")

r = client.post("/v1/subjects/pair", json={"pairing_code": code1, "app_user_id": "phone-aaa"})
check("pairing succeeds", r.status_code == 200, r.text)
check("pairing resolves to same subject", r.json()["subject_id"] == P1)

r = client.post("/v1/subjects/pair", json={"pairing_code": code1, "app_user_id": "phone-zzz"})
check("code cannot be reused", r.status_code == 409, f"got {r.status_code}")

r = client.get("/v1/subjects/resolve", params={"app_user_id": "phone-aaa"})
check("resolve by app_user_id works", r.json().get("subject_id") == P1)
r = client.get("/v1/subjects/resolve", params={"mrn": "NHSL-2026-0142"})
check("resolve by MRN works", r.json().get("subject_id") == P1)
r = client.get("/v1/subjects/resolve", params={"mrn": "nhsl-2026-0142  "})
check("MRN normalised (case/whitespace)", r.json().get("subject_id") == P1)


# ═════════════════════════════════════════════════════════════════════════════
section("3 · The MRN is never stored in plaintext")
import sqlite3  # noqa: E402
con = sqlite3.connect(_tmpdb.name)
blob = ""
for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
    for row in con.execute(f"SELECT * FROM {table}"):
        blob += str(row)
con.close()
check("raw MRN absent from entire database", "NHSL-2026-0142" not in blob)
check("MRN hash present instead", identity.hash_mrn("NHSL-2026-0142")[:16] in blob)
print(f"  scanned {len(blob)} chars of database content for 'NHSL-2026-0142' — not found")


# ═════════════════════════════════════════════════════════════════════════════
section("4 · Day-one guard: demographics only must NOT produce a tier")
r = client.post("/v1/ingest/contextual", json={
    "app_user_id": "phone-aaa", "gender": "female", "age": 21,
    "edu": "bachelor's degree", "smoke": "never smokes", "drink": "never drinks",
    "gad7_items": [2, 2, 1, 2, 1, 1, 2]})
check("contextual ingest 200", r.status_code == 200, r.text)
check("GAD-7 total recomputed server-side", r.json()["gad7_total"] == 11)

r = client.post("/v1/fusion/run", json={"subject_id": P1, "trigger": "test"})
res = r.json()
print(f"  tier   : {res['tier']}")
print(f"  reason : {res['reason']}")
check("no tier from demographics alone", res["tier"] is None)
check("band is GREY not GREEN", res["band"] == "GREY")
check("a reason is recorded", bool(res["reason"]))

r = client.post("/v1/ingest/contextual", json={
    "app_user_id": "phone-aaa", "gad7_items": [0, 1, 2]})
check("malformed GAD-7 rejected", r.status_code == 422, f"got {r.status_code}")


# ═════════════════════════════════════════════════════════════════════════════
section("5 · Behavioural is stored but excluded from the composite")
r = client.post("/v1/ingest/behavioural", json={
    "app_user_id": "phone-aaa", "observations": {"steps": 4200, "screen_min": 310}})
check("behavioural ingest 200", r.status_code == 200, r.text)
check("marked not_validated", r.json()["status"] == "not_validated")
check("flagged as excluded", r.json()["excluded_from_composite"] is True)

r = client.post("/v1/fusion/run", json={"subject_id": P1})
check("still no tier (behavioural does not count)", r.json()["tier"] is None)
print(f"  gate rejected: {r.json()['gate']['rejected'].get('c2_behavioral')}")


# ═════════════════════════════════════════════════════════════════════════════
section("6 · Full fusion once real modalities arrive")
r = client.post("/v1/ingest/physiological", json={"app_user_id": "phone-aaa"})
check("physiological ingest 200", r.status_code == 200, r.text)
check("physio score stored raw (0-100 scale)", r.json()["score"] == 41.8)

r = client.post("/v1/clinical-notes", json={
    "subject_id": P1, "note_text": "Patient reports persistent worry, poor sleep, "
    "and restlessness over the past two weeks.", "note_type": "progress"})
check("clinical note ingest 200", r.status_code == 200, r.text)

r = client.post("/v1/fusion/run", json={"subject_id": P1, "trigger": "note"})
res = r.json()
print(f"  composite : {res['composite']}")
print(f"  tier      : {res['tier']}  band {res['band']}")
print(f"  weights   : { {k: v for k, v in (res['weights'] or {}).items() if v} }")
print(f"  gate      : usable={res['gate']['usable_modalities']}")
check("fusion now produces a tier", res["tier"] in ("Low", "Medium", "High"), str(res))
check("composite in [0,1]", 0.0 <= (res["composite"] or -1) <= 1.0)
check("weights sum to 1", abs(sum((res["weights"] or {}).values()) - 1.0) < 1e-3)
check("behavioural weight is exactly 0", (res["weights"] or {}).get("c2_behavioral", 0) == 0.0)
check("three modalities used", res["modalities_used"] == 3)


# ═════════════════════════════════════════════════════════════════════════════
section("7 · PATIENT SEPARATION — the safety property")
r = client.post("/v1/subjects", json={"mrn": "NHSL-2026-0999"})
P2 = r.json()["subject_id"]
code2 = r.json()["pairing_code"]
client.post("/v1/subjects/pair", json={"pairing_code": code2, "app_user_id": "phone-bbb"})

_C1_SCORE[0], _C3_SCORE[0], _C4_SCORE[0] = 5.0, 0.12, 0.05   # deliberately low
client.post("/v1/ingest/contextual", json={"app_user_id": "phone-bbb", "gender": "male",
                                           "age": 40, "edu": "master's degree"})
client.post("/v1/ingest/physiological", json={"app_user_id": "phone-bbb"})
client.post("/v1/clinical-notes", json={"subject_id": P2,
                                        "note_text": "Patient stable, no acute concerns."})
r2 = client.post("/v1/fusion/run", json={"subject_id": P2}).json()

t1 = client.get(f"/v1/doctor/patients/{P1}/timeline").json()
t2 = client.get(f"/v1/doctor/patients/{P2}/timeline").json()

print(f"  P001 composite {t1['composite']}   P002 composite {t2['composite']}")
print(f"  P001 physio    {t1['modalities']['c1_physiological']['score']}   "
      f"P002 physio    {t2['modalities']['c1_physiological']['score']}")
check("two distinct subject_ids", P1 != P2)
check("composites differ", t1["composite"] != t2["composite"])
check("P001 physio score is P001's", t1["modalities"]["c1_physiological"]["score"] == 41.8)
check("P002 physio score is P002's", t2["modalities"]["c1_physiological"]["score"] == 5.0)
check("P001 contextual is P001's", t1["modalities"]["c4_demographic"]["score"] == 0.55)
check("P002 contextual is P002's", t2["modalities"]["c4_demographic"]["score"] == 0.05)

r = client.post("/v1/subjects/pair", json={"pairing_code": code2, "app_user_id": "phone-aaa"})
check("phone cannot be re-paired to a second patient", r.status_code in (409, 410),
      f"got {r.status_code}")


# ═════════════════════════════════════════════════════════════════════════════
section("8 · Staleness — tightened freshness windows (service contract §5)")
from sqlalchemy import select  # noqa: E402

from db_models import ModalityReading, SessionLocal  # noqa: E402

import gate  # noqa: E402

check("C1 window matches frozen contract (15 min)", gate.MAX_AGE_MINUTES["c1_physiological"] == 15)
check("C2 window matches frozen contract (7 days)",
      gate.MAX_AGE_MINUTES["c2_behavioral"] == 7 * 24 * 60)
check("C3 window matches frozen contract (90 days)",
      gate.MAX_AGE_MINUTES["c3_clinical_nlp"] == 90 * 24 * 60)

db = SessionLocal()
row = db.scalar(select(ModalityReading)
                .where(ModalityReading.subject_id == P1,
                       ModalityReading.modality == "c1_physiological")
                .order_by(ModalityReading.id.desc()))
row.captured_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=20)  # just past 15 min
db.commit()
db.close()

r = client.post("/v1/fusion/run", json={"subject_id": P1, "trigger": "staleness-test"}).json()
print(f"  gate rejected c1: {r['gate']['rejected'].get('c1_physiological')}")
check("20-min-old physiology rejected under the 15-min window",
      "stale" in str(r["gate"]["rejected"].get("c1_physiological", "")))
check("fusion still succeeds on remaining modalities", r["tier"] is not None)
check("now only two modalities", r["modalities_used"] == 2)

# put a fresh reading back for later sections
client.post("/v1/ingest/physiological", json={"app_user_id": "phone-aaa"})


# ═════════════════════════════════════════════════════════════════════════════
section("8b · Effective-weight floor — the mechanism, not just today's config")
# Direct unit test of gate.effective_weight / the floor logic, independent of
# whatever MAX_AGE_MINUTES happens to be set to. This is the regression test
# for the bug the merge fixed: a 6h MAX_AGE against a 30-min half-life let a
# 3h-old reading count as "fresh" while contributing ~1.6% of the weight.
eff_fresh = gate.effective_weight("c1_physiological", age_minutes=1)
eff_stale = gate.effective_weight("c1_physiological", age_minutes=180)   # 3 hours
eff_prior = gate.effective_weight("c4_demographic", age_minutes=999999)  # never decays
print(f"  c1 @ 1 min   : {eff_fresh:.3f}")
print(f"  c1 @ 180 min : {eff_stale:.3f}  (floor is {gate.EFFECTIVE_WEIGHT_FLOOR})")
print(f"  c4 (prior)   : {eff_prior:.3f}")
check("fresh reading near full weight", eff_fresh > 0.9)
check("3h-old c1 reading falls below the floor", eff_stale < gate.EFFECTIVE_WEIGHT_FLOOR)
check("a prior never decays", eff_prior == 1.0)

# gate.py deliberately duplicates fusion.py's HALF_LIFE_MIN (no runtime coupling
# between the two services) — but a silent drift between them would reintroduce
# a version of the exact bug this floor exists to catch. Prove they still match.
_fusion_dir = os.environ["FUSION_SERVICE_DIR"]
if _fusion_dir not in sys.path:
    sys.path.insert(0, _fusion_dir)
import fusion as fusion_maths  # noqa: E402
check("gate half-lives match fusion.py's half-lives",
      gate.HALF_LIFE_MINUTES == fusion_maths.HALF_LIFE_MIN,
      f"gate={gate.HALF_LIFE_MINUTES} fusion={fusion_maths.HALF_LIFE_MIN}")

# Simulate the bug directly: temporarily loosen MAX_AGE the way the old code
# had it, and confirm the floor STILL rejects a 3h-old reading even though the
# age cutoff alone would now let it through.
_old_max_age = gate.MAX_AGE_MINUTES["c1_physiological"]
gate.MAX_AGE_MINUTES["c1_physiological"] = 6 * 60   # the old, too-loose cutoff
try:
    readings = {"c1_physiological": {"raw_score": 0.6, "status": "ok",
                                     "captured_at": dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=3)},
                "c3_clinical_nlp": {"raw_score": 0.68, "status": "ok",
                                    "captured_at": dt.datetime.now(dt.timezone.utc)}}
    decision = gate.evaluate(readings)
    check("floor catches a 3h-old reading even under a loosened 6h cutoff",
          "c1_physiological" not in decision.usable,
          f"usable={list(decision.usable)}")
finally:
    gate.MAX_AGE_MINUTES["c1_physiological"] = _old_max_age


# ═════════════════════════════════════════════════════════════════════════════
section("9 · New status vocabulary — warming_up, no_support_set")
_C1_STATUS[0] = "warming_up"
r = client.post("/v1/ingest/physiological", json={"app_user_id": "phone-aaa"}).json()
print(f"  C1 ingest status: {r['status']}  note: {r['note']}")
check("C1 baseline-not-ready stored as warming_up", r["status"] == "warming_up")
check("no score invented during warm-up", r["score"] is None)

_C3_SUPPORT_K[0] = 0
r = client.post("/v1/clinical-notes", json={
    "subject_id": P1, "note_text": "Second note, empty support set for this test."}).json()
print(f"  C3 ingest status: {r['status']}  note: {r['note']}")
check("C3 with K=0 stored as no_support_set", r["status"] == "no_support_set")
check("no score invented with no support set", r["score"] is None)

r = client.post("/v1/fusion/run", json={"subject_id": P1}).json()
print(f"  gate rejected: {r['gate']['rejected']}")
check("warming_up C1 excluded from composite", "c1_physiological" not in r["gate"]["usable_modalities"])
check("no_support_set C3 excluded from composite", "c3_clinical_nlp" not in r["gate"]["usable_modalities"])

_C1_STATUS[0], _C3_SUPPORT_K[0] = "ok", 12
client.post("/v1/ingest/physiological", json={"app_user_id": "phone-aaa"})
client.post("/v1/clinical-notes", json={"subject_id": P1, "note_text": "Back to a normal note."})


# ═════════════════════════════════════════════════════════════════════════════
section("10 · Component failure is 'missing', never zero")
_C1_STATUS[0] = "error"
r = client.post("/v1/ingest/physiological", json={"app_user_id": "phone-aaa"}).json()
print(f"  ingest status: {r['status']}  note: {r['note']}")
check("failure stored with status=error", r["status"] == "error")
check("no score invented", r["score"] is None)

r = client.post("/v1/fusion/run", json={"subject_id": P1}).json()
check("errored modality excluded from composite",
      "c1_physiological" not in r["gate"]["usable_modalities"])
check("tier still produced from the rest", r["tier"] is not None)
_C1_STATUS[0] = "ok"
client.post("/v1/ingest/physiological", json={"app_user_id": "phone-aaa"})


# ═════════════════════════════════════════════════════════════════════════════
section("11 · Egress — two views, one source of truth")
pat = client.get(f"/v1/patients/{P1}/risk").json()
doc = client.get(f"/v1/doctor/patients/{P1}/timeline").json()
print(f"  patient view keys : {sorted(pat)}")
print(f"  clinician view keys: {sorted(doc)}")
check("patient sees composite + band", "composite" in pat and "band" in pat)
check("patient does NOT see per-modality scores", "modalities" not in pat)
check("patient does NOT see weights", "weights" not in pat)
check("clinician sees per-modality", "modalities" in doc)
check("clinician sees freshness", "age_minutes" in doc["modalities"]["c4_demographic"])
check("clinician sees the gate decision", doc.get("gate") is not None)
check("clinician sees a trend history", len(doc.get("trend", [])) >= 3)
check("same composite in both views", pat["composite"] == doc["composite"])

note_blob = str(doc)
check("raw clinical note text never leaves via egress",
      "persistent worry" not in note_blob and "persistent worry" not in str(pat))


# ═════════════════════════════════════════════════════════════════════════════
section("12 · Audit trail")
from db_models import AuditLog  # noqa: E402

db = SessionLocal()
events = db.scalars(select(AuditLog).where(AuditLog.subject_id == P1)).all()
kinds = sorted({e.event for e in events})
db.close()
print(f"  events for P001: {kinds}")
check("enrolment audited", "enrol.created" in kinds)
check("pairing audited", "enrol.paired" in kinds)
check("ingestion audited", any(k.startswith("ingest.") for k in kinds))
check("fusion audited", any(k.startswith("fusion.") for k in kinds))
check("egress audited", any(k.startswith("egress.") for k in kinds))


# ═════════════════════════════════════════════════════════════════════════════
section("13 · Rejections and edge cases")
check("unknown subject 404",
      client.post("/v1/fusion/run", json={"subject_id": "does-not-exist"}).status_code == 404)
check("unknown pairing code 404",
      client.post("/v1/subjects/pair",
                  json={"pairing_code": "ZZZZ-ZZZZ", "app_user_id": "x"}).status_code == 404)
check("unknown app_user_id 404",
      client.post("/v1/ingest/physiological",
                  json={"app_user_id": "never-paired"}).status_code == 404)
check("empty note rejected by component",
      client.post("/v1/clinical-notes",
                  json={"subject_id": P1, "note_text": "  "}).json()["status"] == "error")
r = client.post("/v1/subjects", json={"mrn": "NHSL-2026-0142"})
check("re-enrolling same MRN reuses subject", r.json()["subject_id"] == P1)
check("...but issues a fresh pairing code", r.json()["pairing_code"] != code1)




# ═════════════════════════════════════════════════════════════════════════════
section("14 · Auto-trigger — fusion fires on events, debounced for the stream")
import main as main_mod  # noqa: E402

from db_models import FusionResult  # noqa: E402

def _fusion_count(subject_id):
    db = SessionLocal()
    n = len(db.scalars(select(FusionResult).where(FusionResult.subject_id == subject_id)).all())
    db.close()
    return n

# fresh subject so counts are clean
r = client.post("/v1/subjects", json={"mrn": "NHSL-2026-0777"})
P3 = r.json()["subject_id"]
client.post("/v1/subjects/pair", json={"pairing_code": r.json()["pairing_code"],
                                       "app_user_id": "phone-ccc"})

# (a) contextual ingest triggers fusion immediately, every time
n0 = _fusion_count(P3)
r = client.post("/v1/ingest/contextual", json={"app_user_id": "phone-ccc",
                                               "gender": "female", "age": 22}).json()
check("contextual ingest reports fusion_triggered", r.get("fusion_triggered") is True)
check("contextual ingest created a fusion row", _fusion_count(P3) == n0 + 1)
check("day-one guard still holds through the auto path",
      r["fusion"]["tier"] is None and r["fusion"]["band"] == "GREY")

# (b) two physiological ingests seconds apart -> only the debounce message, no 2nd row
client.post("/v1/clinical-notes", json={"subject_id": P3, "note_text": "Baseline note."})
n1 = _fusion_count(P3)
r1 = client.post("/v1/ingest/physiological", json={"app_user_id": "phone-ccc"}).json()
r2 = client.post("/v1/ingest/physiological", json={"app_user_id": "phone-ccc"}).json()
print(f"  first physio tick : triggered={r1.get('fusion_triggered')}")
print(f"  second physio tick: triggered={r2.get('fusion_triggered')}  "
      f"({r2.get('fusion_skipped_reason','')})")
check("back-to-back physio ticks fuse at most once",
      _fusion_count(P3) <= n1 + 1)
check("skipped tick explains itself",
      r2.get("fusion_triggered") is False and "debounced" in r2.get("fusion_skipped_reason", ""))

# (c) physio tick after the debounce window -> fuses again.
#     Simulate time passing by backdating every fusion row for P3.
db = SessionLocal()
for row in db.scalars(select(FusionResult).where(FusionResult.subject_id == P3)).all():
    row.computed_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=main_mod.AUTO_FUSION_DEBOUNCE_MIN + 1)
db.commit(); db.close()
n2 = _fusion_count(P3)
r3 = client.post("/v1/ingest/physiological", json={"app_user_id": "phone-ccc"}).json()
check("physio tick after the window fuses again",
      r3.get("fusion_triggered") is True and _fusion_count(P3) == n2 + 1)

# (d) behavioural ingest never auto-triggers — it cannot change the composite
n3 = _fusion_count(P3)
r4 = client.post("/v1/ingest/behavioural", json={"app_user_id": "phone-ccc",
                                                 "observations": {"steps": 100}}).json()
check("behavioural ingest does not auto-trigger",
      "fusion_triggered" not in r4 and _fusion_count(P3) == n3)

# (e) manual endpoint unchanged
r5 = client.post("/v1/fusion/run", json={"subject_id": P3, "trigger": "manual"})
check("manual fusion endpoint still works", r5.status_code == 200 and _fusion_count(P3) == n3 + 1)




# ═════════════════════════════════════════════════════════════════════════════
section("15 · Conformal prediction — honest sets, calibrated only when earned")
import conformal  # noqa: E402
import fusion as fusion_maths2  # noqa: E402 (already on sys.path from section 8b)

from db_models import Verdict  # noqa: E402

# band edges must match fusion.py's BANDS — a silent drift here would make the
# conformal guarantee apply to the wrong intervals
fusion_edges = [e for e, _ in fusion_maths2.BANDS]
check("conformal intervals match fusion band edges",
      abs(conformal.TIER_INTERVALS["Low"][1] - fusion_edges[0]) < 1e-9
      and abs(conformal.TIER_INTERVALS["Medium"][1] - fusion_edges[1]) < 1e-9)

# (a) zero verdicts -> full set, explicitly uncalibrated, with a stated reason
r = client.post("/v1/fusion/run", json={"subject_id": P1, "trigger": "conformal-test"}).json()
print(f"  set={r.get('conformal_set')}  calibrated={r.get('conformal_calibrated')}  n={r.get('conformal_n')}")
check("uncalibrated set is ALL tiers", r.get("conformal_set") == ["Low", "Medium", "High"])
check("calibrated flag is false with no labels", r.get("conformal_calibrated") is False)
check("a reason is stated", bool(r.get("conformal_note")))
check("point tier still present and unchanged", r.get("tier") in ("Low", "Medium", "High"))

# (b) verdict endpoint: record labels, reject junk
fid = None
db = SessionLocal()
fid = db.scalars(select(FusionResult).where(FusionResult.subject_id == P1,
                                            FusionResult.composite.is_not(None))
                 .order_by(FusionResult.id.desc())).first().id
db.close()
r = client.post("/v1/verdict", json={"fusion_result_id": fid, "tier_label": "Medium",
                                     "author": "dr.perera"}).json()
check("verdict recorded", "verdict_id" in r)
check("agreement with model computed", r.get("agrees_with_model") in (True, False))
check("junk tier label rejected",
      client.post("/v1/verdict", json={"fusion_result_id": fid,
                                       "tier_label": "CATASTROPHIC"}).status_code == 422)
check("unknown fusion result 404",
      client.post("/v1/verdict", json={"fusion_result_id": 999999,
                                       "tier_label": "High"}).status_code == 404)

# (c) seed enough clean labels to earn calibration, then the set must tighten.
#     Synthetic fusion rows spread across [0,1], each labelled with its true band.
db = SessionLocal()
import random as _rnd
_rnd.seed(7)
for i in range(24):
    c = _rnd.uniform(0.02, 0.98)
    band = "Low" if c < 0.33 else ("Medium" if c < 0.66 else "High")
    fr = FusionResult(subject_id=P1, composite=c, tier=band, band="GREEN",
                      confidence=0.8, modalities_used=3, weights={}, contributions={},
                      harmonisation={}, trigger="calib-seed", model_version="seed")
    db.add(fr); db.flush()
    db.add(Verdict(subject_id=P1, fusion_result_id=fr.id, tier_label=band,
                   agrees_with_model=True, author="seed"))
db.commit(); db.close()

r = client.post("/v1/fusion/run", json={"subject_id": P1, "trigger": "conformal-test-2"}).json()
print(f"  after 25 labels: set={r.get('conformal_set')}  calibrated={r.get('conformal_calibrated')}  "
      f"q={r.get('conformal_quantile')}")
check("calibration earned at n>=20", r.get("conformal_calibrated") is True)
check("set tightened below all-three", 1 <= len(r.get("conformal_set", [])) <= 2)
check("point tier is inside its own conformal set", r.get("tier") in r.get("conformal_set", []))

# (d) the paper's two numbers: empirical coverage AND mean set size, together
pairs = [(c, ("Low" if c < 0.33 else "Medium" if c < 0.66 else "High"))
         for c in [_rnd.uniform(0.02, 0.98) for _ in range(60)]]
rep = conformal.coverage_report(pairs[:40], pairs[40:], alpha=0.10)
print(f"  coverage report: {rep}")
check("empirical coverage meets nominal on clean data",
      rep["empirical_coverage"] >= rep["nominal_coverage"] - 0.05)
check("sets are not trivially size 3", rep["mean_set_size"] < 3.0)


# ═════════════════════════════════════════════════════════════════════════════
section("16 · CARE-AnxRAG client — separate HTTP service, never fabricates")
import rag_client  # noqa: E402

# (a) local crisis pre-screen fires BEFORE any network call — verified by
# monkeypatching httpx.Client.post to explode ONLY if a call reaches /v1/ask.
# Everything else (including the FastAPI TestClient's own internal use of
# httpx to simulate requests to our app) must pass through untouched.
def _explode(self, url, *a, **k):
    if str(url).endswith("/v1/ask"):
        raise AssertionError("network call made despite local crisis pre-screen")
    return _real_post(self, url, *a, **k)

_real_post = httpx.Client.post
httpx.Client.post = _explode
try:
    res = rag_client.call_rag("patient mentioned wanting to end his life")
finally:
    httpx.Client.post = _real_post
check("crisis pre-screen bypasses the network entirely", res.local_crisis_bypass is True)
check("crisis response has safety_level=crisis", res.safety_level == "crisis")
check("crisis response has no answer", res.answer is None)
check("crisis response is still marked available (this IS the correct behaviour)",
      res.available is True)

# (b) normal question, RAG unreachable (nothing is listening on RAG_URL here)
# -> available=False with an error, NEVER a fabricated answer.
rag_client.RAG_URL = "http://127.0.0.1:1"   # guaranteed nothing listens here
res = rag_client.call_rag("What are anxiety disorders?")
print(f"  unreachable RAG: available={res.available}  error={res.error}")
check("unreachable RAG returns available=False", res.available is False)
check("no fabricated answer when RAG is unreachable", res.answer is None)
check("error message is informative", bool(res.error))

# (c) stub a healthy RAG response and verify the endpoint wires it through
# end-to-end, including citations.
class _FakeResp:
    def __init__(self, status, data): self.status_code = status; self._d = data
    def json(self): return self._d

def _stub_post(self, url, *a, **k):
    if str(url).endswith("/v1/ask"):
        return _FakeResp(200, {
            "answer": "Anxiety disorders involve excessive fear or worry [S1].",
            "citations": [{"citation_id": "S1", "title": "Anxiety disorders",
                          "source_name": "World Health Organization",
                          "source_id": "who_anxiety", "url": "https://who.int/x",
                          "evidence_level": "government_health_information",
                          "excerpt": "..."}],
            "confidence": 0.7992, "conflict_score": 0.0, "abstained": False,
            "abstention_reason": None, "safety_level": "normal", "safety_message": None,
            "knowledge_base_last_sync_at": "2026-08-01T00:00:00Z"})
    return _real_post(self, url, *a, **k)

httpx.Client.post = _stub_post
try:
    rag_client.RAG_URL = "http://127.0.0.1:8000"
    r = client.post(f"/v1/doctor/patients/{P1}/evidence",
                    json={"question": "What are anxiety disorders?"}).json()
finally:
    httpx.Client.post = _real_post

print(f"  answer: {r.get('answer')}")
print(f"  citations: {[c['citation_id'] for c in r.get('citations', [])]}")
check("evidence endpoint returns the RAG's answer", r.get("answer") and "[S1]" in r["answer"])
check("citations passed through", len(r.get("citations", [])) == 1
      and r["citations"][0]["source_name"] == "World Health Organization")
check("confidence passed through", r.get("confidence") == 0.7992)
check("not abstained on a good answer", r.get("abstained") is False)
check("safety_level normal passed through", r.get("safety_level") == "normal")

# (d) stub CARE-AnxRAG's OWN abstention — must be surfaced, not overridden
def _stub_post_abstain(self, url, *a, **k):
    if str(url).endswith("/v1/ask"):
        return _FakeResp(200, {
            "answer": None, "citations": [], "confidence": 0.0, "conflict_score": 0.41,
            "abstained": True, "abstention_reason": "conflicting evidence across sources",
            "safety_level": "normal", "safety_message": None,
            "knowledge_base_last_sync_at": "2026-08-01T00:00:00Z"})
    return _real_post(self, url, *a, **k)

httpx.Client.post = _stub_post_abstain
try:
    r = client.post(f"/v1/doctor/patients/{P1}/evidence",
                    json={"question": "Does X cause anxiety?"}).json()
finally:
    httpx.Client.post = _real_post
check("RAG's own abstention is surfaced, not hidden", r.get("abstained") is True)
check("abstention reason passed through", "conflicting" in (r.get("abstention_reason") or ""))
check("no answer fabricated on abstention", r.get("answer") is None)

# (e) endpoint rejects an empty question rather than calling the RAG with nothing
check("empty question rejected",
      client.post(f"/v1/doctor/patients/{P1}/evidence", json={"question": ""}).status_code == 422)
check("unknown subject 404 on evidence endpoint",
      client.post("/v1/doctor/patients/does-not-exist/evidence",
                  json={"question": "x"}).status_code == 404)

# (f) health endpoint reports RAG configuration without raising
h = client.get("/health").json()
check("health endpoint includes rag block", "rag" in h)
check("rag block reports configured", h["rag"].get("configured") is True)







# ═════════════════════════════════════════════════════════════════════════════
section("17 · Real component payloads — C1, C2, C3 integration")
# ═════════════════════════════════════════════════════════════════════════════
import math  # noqa: E402

# ── C3's confidence field is the score restated; entropy is the real measure ──
print("  C3 published: risk_score=0.6715, confidence=0.671, entropy=0.6331")
conf = mc.confidence_from_entropy(0.6331)
print(f"  entropy-derived confidence: {conf:.4f}")
check("entropy converts to an honest confidence, not the score restated",
      abs(conf - 0.0866) < 0.002, f"got {conf}")
check("a certain prediction (H=0) gives confidence 1.0",
      mc.confidence_from_entropy(0.0) == 1.0)
check("a maximally uncertain prediction (H=ln2) gives confidence 0.0",
      abs(mc.confidence_from_entropy(math.log(2))) < 1e-9)
check("missing entropy returns None (so caller can fall back)",
      mc.confidence_from_entropy(None) is None)
check("malformed entropy returns None rather than crashing",
      mc.confidence_from_entropy("not-a-number") is None)
check("entropy-derived confidence is far lower than C3's own claim",
      conf < 0.671 / 2,
      "if these were close, the circularity concern would be moot")

# ── the subject-echo safety check ────────────────────────────────────────────
check("matching subject echo passes",
      mc.verify_subject_echo("P_123", "P_123", "C2") is None)
check("MISMATCHED subject echo is caught",
      "SUBJECT MISMATCH" in (mc.verify_subject_echo("P_123", "P_999", "C2") or ""))
check("absent echo is tolerated (not all services echo)",
      mc.verify_subject_echo("P_123", None, "C2") is None)

# ── C2: the experimental score must NEVER become the fused score ─────────────
r = client.post("/v1/ingest/behavioural", json={
    "app_user_id": "phone-aaa", "observations": {"steps": 4200}}).json()
print(f"  C2 ingest: status={r['status']}  score={r['score']}")
check("C2 stored as not_validated", r["status"] == "not_validated")
check("C2 score is null, NOT the 0.0254 experimental value", r["score"] is None)

db = SessionLocal()
c2_row = db.scalar(select(ModalityReading)
                   .where(ModalityReading.subject_id == P1,
                          ModalityReading.modality == "c2_behavioral")
                   .order_by(ModalityReading.id.desc()))
detail = c2_row.detail or {}
resp = (detail.get("response") or {})
db.close()
check("C2 raw_score column is NULL in the database", c2_row.raw_score is None)
check("the experimental value IS preserved for the clinician timeline",
      resp.get("behavioral_vulnerability_score") == 0.025467511026434304)
check("...but it is never promoted to raw_score",
      c2_row.raw_score != 0.025467511026434304)

# ── three independent locks keep C2 out of the composite ────────────────────
import gate as gate_mod  # noqa: E402
import fusion as fusion_mod2  # noqa: E402
check("LOCK 1: their service reports fusion_eligible=false",
      resp.get("fusion_eligible") is False)
check("LOCK 2: our gate excludes c2_behavioral unconditionally",
      "c2_behavioral" in gate_mod.EXCLUDED_MODALITIES)
check("LOCK 3: fusion weight for c2 is exactly 0.0",
      fusion_mod2.base_weights()["c2_behavioral"] == 0.0)

r = client.post("/v1/fusion/run", json={"subject_id": P1}).json()
check("c2 absent from usable modalities after a real C2 ingest",
      "c2_behavioral" not in r["gate"]["usable_modalities"])

# ── external id mapping (C2 keys on P_65DC..., we key on UUIDs) ──────────────
r = client.post(f"/v1/subjects/{P1}/external-ids",
                json={"modality": "c2_behavioral",
                      "external_id": "P_65DC4002E7863773"}).json()
check("external id registered", r.get("external_id") == "P_65DC4002E7863773")

from db_models import SessionLocal as _SL  # noqa: E402
db = _SL()
check("backend resolves our UUID to C2's own id",
      main._external_id(db, P1, "c2_behavioral") == "P_65DC4002E7863773")
check("unmapped modality falls back to our subject_id",
      main._external_id(db, P1, "c1_physiological") == P1)
db.close()

check("re-registering updates rather than duplicating",
      client.post(f"/v1/subjects/{P1}/external-ids",
                  json={"modality": "c2_behavioral",
                        "external_id": "P_NEWVALUE"}).status_code == 200)
check("bad modality name rejected",
      client.post(f"/v1/subjects/{P1}/external-ids",
                  json={"modality": "c9_nonsense", "external_id": "x"}).status_code == 422)

# one external id must never map to two patients
r2 = client.post("/v1/subjects", json={"mrn": "NHSL-2026-0555"}).json()
check("the same external id cannot be claimed by a second patient",
      client.post(f"/v1/subjects/{r2['subject_id']}/external-ids",
                  json={"modality": "c2_behavioral",
                        "external_id": "P_NEWVALUE"}).status_code == 409)


print(f"\n{'=' * 74}")
print(f"  {passed} passed, {failed} failed")
print("=" * 74)
os.unlink(_tmpdb.name)
sys.exit(1 if failed else 0)
