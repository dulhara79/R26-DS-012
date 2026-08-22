"""
Fusion service self-test.

Start the two servers in two terminals first:

    terminal 1:   uvicorn mock_components:app --port 7900
    terminal 2:   uvicorn app:app --port 7861

Then run this:

    python test_fusion_service.py

Every check below either prints PASS or explains what went wrong. If all seven
pass, your fusion layer works and you can screenshot the output for your report.
"""

from __future__ import annotations

import sys

import httpx

BASE = "http://127.0.0.1:7861"
passed, failed = 0, 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")


def header(t):
    print(f"\n{'=' * 72}\n{t}\n{'=' * 72}")


try:
    httpx.get(f"{BASE}/health", timeout=5)
except Exception:
    print(f"Cannot reach the fusion service at {BASE}.")
    print("Start it first:  uvicorn app:app --port 7861")
    sys.exit(1)


# ── 1 ────────────────────────────────────────────────────────────────────────
header("1 · Service is up and configured")
h = httpx.get(f"{BASE}/health").json()
print(f"  base weights: {h['base_weights']}")
print(f"  references  : {h['reference_distributions']}")
check("behavioural has zero base weight", h["base_weights"]["c2_behavioral"] == 0.0)
check("at least one reference distribution loaded", len(h["reference_distributions"]) >= 1)


# ── 2 ────────────────────────────────────────────────────────────────────────
header("2 · Manual fuse — reproducible, no network")
r = httpx.post(f"{BASE}/v1/fuse/manual", json={
    "mrn": "TEST-1", "already_harmonised": True,
    "components": {
        "c1_physiological": {"score": 0.62, "confidence": 0.74, "coverage": 0.81},
        "c3_clinical_nlp": {"score": 0.68, "confidence": 0.83},
        "c4_demographic": {"score": 0.55, "confidence": 0.61},
    }}).json()
print(f"  composite {r['composite_score']}   tier {r['tier']}")
print(f"  weights   { {k: v for k, v in r['weights'].items() if v > 0} }")
check("composite lies between the lowest and highest input", 0.55 <= r["composite_score"] <= 0.68)
check("weights sum to 1", abs(sum(r["weights"].values()) - 1.0) < 1e-6)
check("clinical notes carry the most weight",
      max(r["weights"], key=r["weights"].get) == "c3_clinical_nlp")


# ── 3 ────────────────────────────────────────────────────────────────────────
header("3 · Day-one guard — demographics only must NOT produce a tier")
r = httpx.post(f"{BASE}/v1/fuse/manual", json={
    "mrn": "TEST-2", "already_harmonised": True,
    "components": {"c4_demographic": {"score": 0.55, "confidence": 0.61}}}).json()
print(f"  tier   : {r['tier']}")
print(f"  reason : {r['reason']}")
check("no tier returned", r["tier"] is None)
check("a reason is given", bool(r.get("reason")))


# ── 4 ────────────────────────────────────────────────────────────────────────
header("4 · Zero-weight rule — behavioural cannot move the composite")
base = {"c1_physiological": {"score": 0.62, "confidence": 0.74, "coverage": 0.81},
        "c3_clinical_nlp": {"score": 0.68, "confidence": 0.83},
        "c4_demographic": {"score": 0.55, "confidence": 0.61}}
without = httpx.post(f"{BASE}/v1/fuse/manual", json={
    "mrn": "TEST-3", "already_harmonised": True, "components": base}).json()
with_c2 = httpx.post(f"{BASE}/v1/fuse/manual", json={
    "mrn": "TEST-3", "already_harmonised": True,
    "components": {**base, "c2_behavioral": {"score": 0.99, "confidence": 0.95}}}).json()
print(f"  without behavioural : {without['composite_score']}")
print(f"  with behavioural 0.99: {with_c2['composite_score']}")
check("composite unchanged", without["composite_score"] == with_c2["composite_score"])


# ── 5 ────────────────────────────────────────────────────────────────────────
header("5 · Recency — a stale reading loses its weight")
fresh = httpx.post(f"{BASE}/v1/fuse/manual", json={
    "mrn": "TEST-4", "already_harmonised": True, "components": {
        "c1_physiological": {"score": 0.62, "confidence": 0.74, "coverage": 0.81},
        "c3_clinical_nlp": {"score": 0.68, "confidence": 0.83},
        "c4_demographic": {"score": 0.55, "confidence": 0.61}}}).json()
stale = httpx.post(f"{BASE}/v1/fuse/manual", json={
    "mrn": "TEST-4", "already_harmonised": True, "components": {
        "c1_physiological": {"score": 0.62, "confidence": 0.74, "coverage": 0.81,
                             "captured_at": "2026-01-01T00:00:00Z"},
        "c3_clinical_nlp": {"score": 0.68, "confidence": 0.83},
        "c4_demographic": {"score": 0.55, "confidence": 0.61}}}).json()
print(f"  physio weight, reading 1 minute old : {fresh['weights']['c1_physiological']}")
print(f"  physio weight, reading months old   : {stale['weights']['c1_physiological']}")
check("stale physiology loses almost all weight",
      stale["weights"]["c1_physiological"] < fresh["weights"]["c1_physiological"] / 10)


# ── 6 ────────────────────────────────────────────────────────────────────────
header("6 · Live fan-out — calls the component services")
try:
    r = httpx.post(f"{BASE}/v1/fuse", json={"mrn": "NHSL-0142"}, timeout=30).json()
    for m, hh in r["harmonisation"].items():
        if "raw" in hh:
            print(f"  {m:20s} raw {hh['raw']:.4f}  ->  percentile {hh['harmonised']:.4f}")
        else:
            print(f"  {m:20s} unavailable — {hh.get('note')}")
    check("at least two components answered", r["modalities_available"] >= 2)
    check("harmonisation changed the scale for at least one component",
          any("raw" in hh and abs(hh["raw"] - hh["harmonised"]) > 0.01
              for hh in r["harmonisation"].values()))
except Exception as exc:
    print(f"  could not reach components: {exc}")
    print("  start the mocks:  uvicorn mock_components:app --port 7900")


# ── 7 ────────────────────────────────────────────────────────────────────────
header("7 · Live physiological stream — smoothing")
httpx.post(f"{BASE}/v1/fuse", json={"mrn": "TEST-LIVE"}, timeout=30)
series = []
for s in [0.08] * 3 + [0.28] * 6 + [0.08] * 5:
    out = httpx.post(f"{BASE}/v1/physio/tick", json={"mrn": "TEST-LIVE", "score": s}).json()
    series.append(out["composite_score"])
    print(f"  raw {s:.2f}   composite {out['composite_score']:.3f}   tier {out['tier']}")
rises = series[3:9]
check("composite rises gradually rather than jumping",
      all(b >= a for a, b in zip(rises, rises[1:])) and (rises[-1] - rises[0]) < 0.35,
      "smoothing should spread a step change over several minutes")


print(f"\n{'=' * 72}")
print(f"  {passed} passed, {failed} failed")
print("=" * 72)
sys.exit(1 if failed else 0)
