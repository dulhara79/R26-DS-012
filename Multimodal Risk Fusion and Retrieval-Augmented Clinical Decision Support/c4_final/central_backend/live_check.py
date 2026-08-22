"""
LIVE integration check — hits the REAL C1, C2, and C3 services.

Everything in test_backend.py runs against stubs, because a test suite that
depends on three sleeping Hugging Face Spaces and a Vercel function is a test
suite that fails for reasons unrelated to your code. That is the right default.

But stubs prove your code handles the shape you EXPECT. This script proves the
services actually return that shape. Those are different claims, and only this
one can be made from your machine.

Run:  python live_check.py

It never writes to your database and never calls the backend — it talks to the
three component services directly, so a failure here is unambiguously theirs,
not yours.
"""

from __future__ import annotations

import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

import httpx

import modality_clients as mc

# Fill these in — or set them in .env and they'll be picked up automatically.
C2_TEST_SUBJECT = os.getenv("C2_TEST_SUBJECT", "P_65DC4002E7863773")
C1_TEST_SUBJECT = os.getenv("C1_TEST_SUBJECT", "P_8A0840A798B81072")
C3_TEST_NOTE = ("Patient reports persistent worry over the past two weeks, "
                "difficulty controlling the worry, poor sleep and restlessness.")

results = {}


def hr(title):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def show(label, value, indent=2):
    print(f"{' ' * indent}{label:<28} {value}")


# ═════════════════════════════════════════════════════════════════════════════
hr("Configuration")
show("C1_URL", mc.C1_BASE or "(not set)")
show("C2_URL", mc.C2_BASE or "(not set)")
show("C3_URL", mc.C3_BASE or "(not set)")
show("C4_URL", mc.C4_BASE or "(not set)")
show("timeout", f"{mc.TIMEOUT_S}s")
print("\n  NOTE: Hugging Face Spaces sleep after inactivity. The FIRST call to a")
print("  sleeping Space can take 30-60s while it wakes. That is not a failure.")


# ═════════════════════════════════════════════════════════════════════════════
hr("C2 · behavioural (Vercel)")
if not mc.C2_BASE:
    print("  SKIPPED — C2_URL not set in .env")
    results["C2"] = "skipped"
else:
    t0 = time.time()
    r = mc.call_c2(C2_TEST_SUBJECT)
    show("elapsed", f"{time.time() - t0:.1f}s")
    show("status", r.status)
    show("raw_score (fused)", r.raw_score)
    show("coverage", round(r.coverage, 4))
    show("model_version", r.model_version)
    show("note", (r.note or "")[:90])

    body = r.detail or {}
    experimental = body.get("behavioral_vulnerability_score")
    show("experimental score", experimental)
    show("fusion_eligible", body.get("fusion_eligible"))

    ok = True
    if r.raw_score is not None:
        print("\n  *** CRITICAL: C2 returned a fusable score. Expected null. ***")
        ok = False
    if experimental is not None and r.raw_score == experimental:
        print("\n  *** CRITICAL: the experimental score leaked into raw_score! ***")
        ok = False
    if r.status == "error":
        print(f"\n  Service error: {r.note}")
        ok = False
    print(f"\n  -> {'OK — stored, never fused (correct)' if ok else 'PROBLEM, see above'}")
    results["C2"] = "ok" if ok else "problem"


# ═════════════════════════════════════════════════════════════════════════════
hr("C3 · clinical notes (Hugging Face)")
if not mc.C3_BASE:
    print("  SKIPPED — C3_URL not set in .env")
    results["C3"] = "skipped"
else:
    t0 = time.time()
    r = mc.call_c3(C3_TEST_NOTE, subject_external_id=None)
    elapsed = time.time() - t0
    show("elapsed", f"{elapsed:.1f}s")
    show("status", r.status)
    show("raw_score", r.raw_score)
    show("confidence (ours)", round(r.confidence, 4))
    show("model_version", r.model_version)
    show("note", (r.note or "")[:110])

    body = r.detail or {}
    show("their 'confidence'", body.get("confidence"))
    show("their 'entropy'", body.get("entropy"))
    show("their 'risk_score'", body.get("risk_score"))
    show("calibrated_probability", body.get("calibrated_probability"))

    if r.status == "error" and "404" in (r.note or ""):
        print("\n  /predict returned 404. The API path may differ from the UI path.")
        print("  Open these in a browser to find the real one:")
        print(f"    {mc.C3_BASE}/docs")
        print(f"    {mc.C3_BASE}/openapi.json")
        results["C3"] = "wrong-path"
    elif r.status == "error":
        print(f"\n  Service error: {r.note}")
        results["C3"] = "problem"
    else:
        their_conf = body.get("confidence")
        their_score = body.get("risk_score") or body.get("calibrated_probability")
        if (their_conf is not None and their_score is not None
                and abs(float(their_conf) - float(their_score)) < 0.01):
            print("\n  CONFIRMED: C3's `confidence` equals its risk score — it is the")
            print("  probability restated, not an uncertainty measure. We are correctly")
            print("  using the entropy-derived value instead. Tell Dulhara: the field")
            print("  named `confidence` is misleading and should be renamed or fixed.")
        if body.get("calibrated_probability") is None:
            print("\n  NOTE: no `calibrated_probability` field. The frozen contract says")
            print("  fusion should consume the CALIBRATED probability, not the raw score.")
            print("  Ask Dulhara to publish it.")
        if elapsed > 8:
            print(f"\n  NOTE: {elapsed:.0f}s is slow. Fine for a demo, but the Flutter app")
            print("  will need a loading state rather than a frozen screen.")
        results["C3"] = "ok"


# ═════════════════════════════════════════════════════════════════════════════
hr("C1 · physiological (Hugging Face)")
if not mc.C1_BASE:
    print("  SKIPPED — C1_URL not set in .env")
    results["C1"] = "skipped"
else:
    print("  Step 1 — enrolling P_8A0840A798B81072 with Dewdu's synthetic baseline...")
    with httpx.Client() as hc:
        norm_r = hc.post(
            f"{mc.C1_BASE}/set_norm_params/{C1_TEST_SUBJECT}",
            headers=mc._headers(mc.C1_TOKEN),
            json={
                "b_mean": [71.0,845.18,46.0,39.0,15.5,0.85,34.25,0.085,1.00233,0.01333],
                "b_std":  [1.0,15.0,2.0,2.0,0.5,0.1,0.05,0.01,0.01,0.005],
                "baseline_windows": [
                    [70.0,857.14,45.0,38.0,15.0,0.8,34.2,0.08,1.001,0.012],
                    [72.0,833.33,47.0,40.0,16.0,0.9,34.3,0.09,1.004,0.015],
                    [71.0,845.07,46.0,39.0,15.5,0.85,34.25,0.085,1.002,0.013],
                ]
            }, timeout=30)
    show("set_norm_params HTTP", norm_r.status_code)
    if norm_r.status_code not in (200, 201):
        print(f"  Enrolment failed: {norm_r.text[:200]}")
        results["C1"] = "enrol-failed"
    else:
        print(f"  Enrolment: {norm_r.text[:120]}")
        print("\n  Step 2 — calling /predict...")
        t0 = time.time()
        r = mc.call_c1(C1_TEST_SUBJECT, window=None)
        show("elapsed", f"{time.time()-t0:.1f}s")
        show("status", r.status)
        show("raw_score", r.raw_score)
        show("captured_at", r.captured_at)
        show("note", (r.note or "")[:120])
        for key in ("status","current_risk_index","latest_reading_at","message"):
            if r.detail and key in r.detail:
                show(f"  body.{key}", r.detail[key], indent=4)
        if r.status in ("ok","warming_up","poor_signal"):
            print(f"\n  -> OK (status={r.status})")
            results["C1"] = "ok"
        else:
            print(f"\n  -> Problem: status='{r.status}'")
            results["C1"] = "problem"


# ═════════════════════════════════════════════════════════════════════════════
hr("Summary")
for k in ("C1", "C2", "C3"):
    show(k, results.get(k, "not run"))

print("\n  What each result means:")
print("    ok          service answered in the expected shape")
print("    warming_up  answered, but not yet usable (correct, not an error)")
print("    no-data     answered, but has nothing stored for that test subject")
print("    wrong-path  reachable, but the API path differs from what we call")
print("    problem     see the section above")
print("    skipped     URL not configured in .env")

print("\n  Reminder: C2 showing status=not_validated with score=None is the")
print("  CORRECT result, not a failure. It is excluded by three independent")
print("  locks and must never contribute to a composite.")

sys.exit(0 if all(v in ("ok", "warming_up", "skipped", "no-data")
                  for v in results.values()) else 1)
