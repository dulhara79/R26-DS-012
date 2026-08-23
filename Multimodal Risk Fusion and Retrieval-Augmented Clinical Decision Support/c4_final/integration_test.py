"""
Clean integration test — hits C1, C2, C3 directly.
No enrolment, no database writes, just raw service checks.
"""

import httpx
import os
from dotenv import load_dotenv

load_dotenv()

C1_URL = os.getenv("C1_URL", "").rstrip("/")
C2_URL = os.getenv("C2_URL", "").rstrip("/")
C3_URL = os.getenv("C3_URL", "").rstrip("/")
C3_TOKEN = os.getenv("C3_TOKEN", "")

C1_USER = "P_8A0840A798B81072"   # Dewdu's real simulation user
TIMEOUT  = 60.0

results = {}

def section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# ─── C1 ──────────────────────────────────────────────────────
section("C1 · Physiological (Dewdu)")
try:
    r = httpx.get(f"{C1_URL}/predict/{C1_USER}", timeout=TIMEOUT)
    body = r.json()
    print(f"  HTTP status      : {r.status_code}")
    print(f"  service status   : {body.get('status')}")
    print(f"  current_risk_index: {body.get('current_risk_index')}")
    print(f"  latest_reading_age: {body.get('latest_reading_age_seconds')}s ago")

    if r.status_code == 200 and body.get("current_risk_index") is not None:
        print("  RESULT           : ✅ PASS")
        results["C1"] = "PASS"
    else:
        print(f"  RESULT           : ❌ FAIL — {body}")
        results["C1"] = "FAIL"
except Exception as e:
    print(f"  RESULT           : ❌ ERROR — {e}")
    results["C1"] = "ERROR"

# ─── C2 ──────────────────────────────────────────────────────
section("C2 · Behavioural (Vercel)")
try:
    r = httpx.get(
        f"{C2_URL}/api/score/P_65DC4002E7863773",
        timeout=TIMEOUT
    )
    body = r.json()
    print(f"  HTTP status      : {r.status_code}")
    print(f"  service status   : {body.get('status')}")
    print(f"  fusion_eligible  : {body.get('fusion_eligible')}")
    print(f"  experimental_score: {body.get('experimental_score')}")

    if r.status_code == 200 and body.get("status") == "not_validated":
        print("  RESULT           : ✅ PASS (not_validated is correct for C2)")
        results["C2"] = "PASS"
    else:
        print(f"  RESULT           : ❌ FAIL — {body}")
        results["C2"] = "FAIL"
except Exception as e:
    print(f"  RESULT           : ❌ ERROR — {e}")
    results["C2"] = "ERROR"

# ─── C3 ──────────────────────────────────────────────────────
section("C3 · Clinical NLP (Dulhara)")
try:
    headers = {"Authorization": f"Bearer {C3_TOKEN}"} if C3_TOKEN else {}
    r = httpx.post(
        f"{C3_URL}/predict",
        json={"note_text": "Patient reports persistent worry and restlessness for two weeks."},
        headers=headers,
        timeout=TIMEOUT
    )
    body = r.json()
    print(f"  HTTP status      : {r.status_code}")
    print(f"  risk_score       : {body.get('risk_score')}")
    print(f"  confidence       : {body.get('confidence')}")
    print(f"  entropy          : {body.get('entropy')}")

    if r.status_code == 200 and body.get("risk_score") is not None:
        print("  RESULT           : ✅ PASS")
        results["C3"] = "PASS"
    elif r.status_code == 401:
        print("  RESULT           : ❌ FAIL — HTTP 401, C3_TOKEN missing or wrong")
        results["C3"] = "FAIL (need token)"
    else:
        print(f"  RESULT           : ❌ FAIL — {body}")
        results["C3"] = "FAIL"
except Exception as e:
    print(f"  RESULT           : ❌ ERROR — {e}")
    results["C3"] = "ERROR"

# ─── Summary ─────────────────────────────────────────────────
section("Summary")
for k, v in results.items():
    icon = "✅" if v == "PASS" else "❌"
    print(f"  {icon}  {k}  →  {v}")
print()
