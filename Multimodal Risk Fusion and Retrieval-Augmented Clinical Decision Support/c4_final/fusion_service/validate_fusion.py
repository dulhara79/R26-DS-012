"""
Fusion model — comprehensive validation.

This is DELIBERATELY NOT test_fusion_service.py or test_backend.py run again.
Those prove specific hand-picked scenarios behave correctly. This file asks a
different, harder question: does the maths hold up in general, under inputs
nobody hand-picked, and does it match an INDEPENDENT hand-derivation rather
than just being self-consistent with its own code?

Five layers, each catching a different class of bug that the existing test
suites structurally cannot catch:

  1. INDEPENDENT ARITHMETIC — weights are recomputed here from first
     principles, in a few lines that do NOT call fusion.base_weights(), and
     compared to what the real code produces. A bug shared between fusion.py
     and its own test file would pass every existing test and still be wrong;
     it cannot survive an independently written computation.

  2. DETERMINISM — the same input must produce the same composite every time.
     A research paper's reported numbers are worthless if the pipeline isn't
     reproducible. Checked by hashing the wire output (minus timestamps).

  3. PROPERTY-BASED INVARIANTS UNDER RANDOMISATION — thousands of random
     input combinations, checked against rules that must ALWAYS hold:
     weights sum to 1, composite in [0,1], C2 always exactly 0, monotonicity
     (more evidence of risk never DECREASES the composite), recency
     monotonicity (older never outweighs fresher), the day-one guard.
     A handful of hand-picked examples can miss a bug that only shows up at
     an unusual combination of scores; random sampling is far more likely to
     find it.

  4. SENSITIVITY / ABLATION — systematic sweeps (composite vs. age, composite
     vs. confidence, composite vs. each modality's score in isolation) that
     produce a table suitable for the paper's sensitivity-analysis section,
     and that make it visually obvious if a modality's influence is absurdly
     large or small relative to its weight.

  5. LIVE-SYSTEM CHECKLIST — the things NO script run on your laptop can
     validate: real component connections, real reference distributions, real
     labelled outcomes. Printed as an explicit checklist, not silently
     skipped, so "the script passed" is never confused with "the system is
     fully validated."

Run:  python validate_fusion.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timedelta, timezone

from fusion import (BANDS, CLEARS_PERMUTATION_NULL, HALF_LIFE_MIN, PRIOR_CAP,
                    VALIDATION_AUROC, MODALITIES, Reading, base_weights, fuse)
from harmonise import Harmoniser, REFERENCE_DIR

passed, failed = 0, 0


def check(name: str, cond: bool, detail: str = ""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}   {detail}")


def section(title: str):
    print(f"\n{'=' * 76}\n{title}\n{'=' * 76}")


def now():
    return datetime.now(timezone.utc)


# ═════════════════════════════════════════════════════════════════════════════
section("1 · INDEPENDENT ARITHMETIC — hand-derived, not imported from fusion.py")
# ═════════════════════════════════════════════════════════════════════════════
print("  Recomputing base weights from VALIDATION_AUROC using a formula written")
print("  fresh here, then comparing against fusion.base_weights()'s real output.")

hand_omega = {}
for m in MODALITIES:
    auroc = VALIDATION_AUROC[m]
    clears = CLEARS_PERMUTATION_NULL[m]
    hand_omega[m] = max(auroc - 0.5, 0.0) if clears else 0.0

hand_total = sum(hand_omega.values())
hand_weights = ({m: v / hand_total for m, v in hand_omega.items()}
               if hand_total > 0 else {m: 0.0 for m in MODALITIES})

real_weights = base_weights()
print(f"  hand-derived : {{k: round(v,4) for k,v in hand_weights.items()}}"
      .replace("{k: round(v,4) for k,v in hand_weights.items()}",
               str({k: round(v, 4) for k, v in hand_weights.items()})))
print(f"  fusion.py's  : {dict((k, round(v,4)) for k,v in real_weights.items())}")

check("independently-derived weights match fusion.py's own output",
      all(abs(hand_weights[m] - real_weights[m]) < 1e-9 for m in MODALITIES))

check("C2 (failed permutation null) is exactly 0 by construction, not by luck",
      hand_weights["c2_behavioral"] == 0.0 and real_weights["c2_behavioral"] == 0.0)

check("C4's AUROC is the real measured value (0.6220), not the 0.66 placeholder",
      VALIDATION_AUROC["c4_demographic"] == 0.6220,
      f"found {VALIDATION_AUROC['c4_demographic']} — was this reverted?")

# Hand-derive one full composite by arithmetic alone, independent of fuse().
print("\n  Hand-computing one full composite, start to finish, no shortcuts:")
fixed_now_1 = now()
c1_raw, c3_raw = 0.61, 0.68           # already-harmonised percentiles, by assumption
w1, w3 = hand_weights["c1_physiological"], hand_weights["c3_clinical_nlp"]
# only two modalities present -> renormalise over just those two
denom = w1 + w3
w1n, w3n = w1 / denom, w3 / denom
expected_composite = w1n * c1_raw + w3n * c3_raw
print(f"    renormalised weights: c1={w1n:.4f} c3={w3n:.4f}")
print(f"    expected composite  : {expected_composite:.4f}")

readings = {
    "c1_physiological": Reading(score=c1_raw, available=True, confidence=1.0,
                                coverage=1.0, captured_at=fixed_now_1),
    "c3_clinical_nlp": Reading(score=c3_raw, available=True, confidence=1.0,
                               coverage=1.0, captured_at=fixed_now_1),
}
out = fuse(readings, now=fixed_now_1)
print(f"    fuse()'s composite  : {out.composite:.4f}")
# fusion.py deliberately rounds the returned composite to 4dp (line ~197,
# `composite=round(composite, 4)`) for display, same as the weights. Comparing
# an unrounded hand value against that rounded output at a tight tolerance will
# always show a ~5e-5 "mismatch" that is not a bug — round the expectation the
# same way fuse() does, THEN compare, which is the correct apples-to-apples check.
check("hand-computed composite matches fuse()'s real output (both rounded to 4dp)",
      out.composite == round(expected_composite, 4),
      f"expected {round(expected_composite, 4)}, got {out.composite}")


# ═════════════════════════════════════════════════════════════════════════════
section("2 · DETERMINISM — identical input, identical output, every time")
# ═════════════════════════════════════════════════════════════════════════════
fixed_time = now()
readings_fixed = {
    "c1_physiological": Reading(score=0.61, available=True, confidence=0.74,
                                coverage=0.81, captured_at=fixed_time),
    "c3_clinical_nlp": Reading(score=0.68, available=True, confidence=0.83,
                               coverage=1.0, captured_at=fixed_time),
    "c4_demographic": Reading(score=0.55, available=True, confidence=0.61,
                              coverage=1.0, captured_at=fixed_time),
}


def _fingerprint(wire: dict) -> str:
    w = dict(wire)
    w.pop("computed_at", None)      # the only field allowed to vary
    return hashlib.sha256(json.dumps(w, sort_keys=True).encode()).hexdigest()


fingerprints = set()
for _ in range(50):
    result = fuse(readings_fixed, now=fixed_time)   # pinned `now` -> recency identical too
    fingerprints.add(_fingerprint(result.to_wire()))

check("50 runs of identical input produce byte-identical output (minus timestamp)",
      len(fingerprints) == 1, f"got {len(fingerprints)} distinct outputs, expected 1")


# ═════════════════════════════════════════════════════════════════════════════
section("3 · PROPERTY-BASED INVARIANTS — 2,000 randomised scenarios")
# ═════════════════════════════════════════════════════════════════════════════
rng = random.Random(20260822)   # seeded for a reproducible report
N = 2000
violations = {
    "weights_sum_to_1": 0, "composite_in_range": 0, "c2_never_weighted": 0,
    "day_one_guard": 0, "renormalised_flag_consistent": 0,
}
# fuse() itself rounds each weight to 4dp before returning (fusion.py line ~198,
# `weights={m: round(alpha[m], 4) for m in MODALITIES}`) — this is a deliberate
# display choice, not a bug, and up to 4 modalities' rounding can accumulate to
# ~4*0.00005=2e-4 of drift in the sum. The tolerance below is set just above that
# ceiling. This is the SAME class of issue test_backend.py hit and fixed earlier
# (1e-6 -> 1e-3 for the same reason) — recorded here so it isn't "rediscovered"
# and mis-tolerance'd a third time somewhere else.
WEIGHT_SUM_TOLERANCE = 3e-4

def random_reading(available_bias=0.75):
    if rng.random() > available_bias:
        return Reading(available=False)
    age_min = rng.choice([0, 1, 5, 30, 60, 300, 1000, 40000, 200000])
    return Reading(score=rng.random(), available=True,
                   confidence=rng.random(), coverage=rng.random(),
                   captured_at=now() - timedelta(minutes=age_min))

for _ in range(N):
    r = {m: random_reading() for m in MODALITIES}
    out = fuse(r)

    wsum = sum(out.weights.values())
    if out.composite is not None and abs(wsum - 1.0) > WEIGHT_SUM_TOLERANCE:
        violations["weights_sum_to_1"] += 1

    if out.composite is not None and not (0.0 <= out.composite <= 1.0):
        violations["composite_in_range"] += 1

    if out.weights.get("c2_behavioral", 0.0) != 0.0:
        violations["c2_never_weighted"] += 1

    only_c4 = (r["c4_demographic"].available
              and not any(r[m].available for m in MODALITIES if m != "c4_demographic"))
    if only_c4 and out.tier is not None:
        violations["day_one_guard"] += 1

    n_avail = sum(1 for m in MODALITIES if r[m].available and m != "c2_behavioral")
    if out.composite is not None:
        should_be_renorm = n_avail < 3   # fewer than the 3 real (non-excluded) modalities
        # renormalised is allowed to be True even at n_avail==3 (PRIOR_CAP redistribution),
        # but must be True whenever a modality is genuinely missing
        if n_avail < 3 and not out.renormalised:
            violations["renormalised_flag_consistent"] += 1

for check_name, count in violations.items():
    check(f"{check_name}  (0 violations in {N} random scenarios)", count == 0,
          f"{count} violations out of {N}")


# ═════════════════════════════════════════════════════════════════════════════
section("4 · MONOTONICITY — more evidence of risk must never LOWER the composite")
# ═════════════════════════════════════════════════════════════════════════════
print("  This is the property a clinician would assume without being told:")
print("  if C1's score alone goes up and nothing else changes, the composite")
print("  must not go down. 500 random base scenarios, each perturbed once.\n")

mono_violations = 0
for _ in range(500):
    base = {m: random_reading(available_bias=0.9) for m in MODALITIES}
    if not base["c1_physiological"].available:
        continue
    lo = min(base["c1_physiological"].score, 0.9)
    hi = min(lo + rng.uniform(0.01, 0.3), 1.0)

    base["c1_physiological"].score = lo
    out_lo = fuse({m: base[m] for m in MODALITIES})
    base["c1_physiological"].score = hi
    out_hi = fuse({m: base[m] for m in MODALITIES})

    if out_lo.composite is not None and out_hi.composite is not None:
        if out_hi.composite < out_lo.composite - 1e-9:
            mono_violations += 1

check("raising one modality's score never lowers the composite (500 trials)",
      mono_violations == 0, f"{mono_violations} violations")

# recency monotonicity: a FRESHER reading of the same score must carry >= weight
recency_violations = 0
for _ in range(500):
    score = rng.random()
    age_a, age_b = sorted([rng.uniform(0, 500), rng.uniform(0, 500)])   # age_a <= age_b
    r_fresh = {"c1_physiological": Reading(score=score, available=True, confidence=1,
                                           coverage=1, captured_at=now() - timedelta(minutes=age_a)),
              "c3_clinical_nlp": Reading(score=0.5, available=True, confidence=1,
                                        coverage=1, captured_at=now())}
    r_stale = {"c1_physiological": Reading(score=score, available=True, confidence=1,
                                           coverage=1, captured_at=now() - timedelta(minutes=age_b)),
              "c3_clinical_nlp": Reading(score=0.5, available=True, confidence=1,
                                        coverage=1, captured_at=now())}
    w_fresh = fuse(r_fresh).weights.get("c1_physiological", 0.0)
    w_stale = fuse(r_stale).weights.get("c1_physiological", 0.0)
    if w_fresh < w_stale - 1e-9:
        recency_violations += 1

check("a fresher reading never carries less weight than a staler one (500 trials)",
      recency_violations == 0, f"{recency_violations} violations")


# ═════════════════════════════════════════════════════════════════════════════
section("5 · SENSITIVITY SWEEPS — for the paper's sensitivity-analysis section")
# ═════════════════════════════════════════════════════════════════════════════
print("  composite vs. C1 age (C1=0.8 high, C3=0.3 low, C4=0.5, all else fixed):\n")
sweep_rows = []
for age_min in [0, 15, 30, 60, 120, 240, 480, 1000]:
    r = {
        "c1_physiological": Reading(score=0.8, available=True, confidence=0.8,
                                    coverage=1.0, captured_at=now() - timedelta(minutes=age_min)),
        "c3_clinical_nlp": Reading(score=0.3, available=True, confidence=0.8,
                                   coverage=1.0, captured_at=now()),
        "c4_demographic": Reading(score=0.5, available=True, confidence=0.8,
                                  coverage=1.0, captured_at=now()),
    }
    out = fuse(r)
    sweep_rows.append((age_min, out.composite, out.weights.get("c1_physiological", 0.0)))
    print(f"    C1 age {age_min:5d} min   composite {out.composite:.4f}   "
          f"C1 weight {out.weights.get('c1_physiological', 0.0):.4f}")

check("composite strictly decreases as the high-risk C1 reading grows stale",
      all(sweep_rows[i][1] >= sweep_rows[i+1][1] - 1e-9 for i in range(len(sweep_rows)-1)))
check("C1's weight decays toward zero as it goes stale",
      sweep_rows[-1][2] < sweep_rows[0][2] * 0.05,
      f"C1 weight at 1000min = {sweep_rows[-1][2]:.4f}, expected < 5% of fresh weight")

print("\n  composite vs. C4 (demographic) score alone, capped by PRIOR_CAP="
      f"{PRIOR_CAP} (C1/C3 absent, C4-only -> no tier, but composite is still")
print("  reported for inspection):\n")
for c4_score in [0.1, 0.3, 0.5, 0.7, 0.9]:
    r = {"c4_demographic": Reading(score=c4_score, available=True, confidence=1.0,
                                   coverage=1.0, captured_at=now())}
    out = fuse(r)
    print(f"    C4 score {c4_score:.1f}   composite {out.composite}   tier {out.tier}   "
          f"weight {out.weights.get('c4_demographic', 0.0):.4f}")


# ═════════════════════════════════════════════════════════════════════════════
section("6 · REFERENCE-DISTRIBUTION SANITY — are these real, or still placeholders?")
# ═════════════════════════════════════════════════════════════════════════════
h = Harmoniser()
avail = h.available()
print(f"  loaded reference distributions: {avail}")
placeholder_warnings = 0
for m, meta in h.meta.items():
    source = meta.get("source", "")
    is_placeholder = "PLACEHOLDER" in source.upper()
    if is_placeholder:
        placeholder_warnings += 1
        print(f"  WARN  {m} reference distribution is still a PLACEHOLDER — "
              f"source='{source}'")
    else:
        print(f"  OK    {m} reference distribution looks real — source='{source}'")
print(f"\n  {placeholder_warnings}/{len(h.meta)} reference distributions are still "
      f"placeholders. This is NOT a code bug — the harmoniser, gate, and fusion "
      f"maths are all correct regardless. It means every percentile the system "
      f"currently reports is being measured against fake data, and nothing in "
      f"this codebase can fix that — only real held-out scores from C1/C3, and "
      f"a rebuild from dcar_reference_scores.npy for C4, can. Tracked separately "
      f"from CODE-LEVEL VALIDATION below on purpose.")


# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 76}")
print(f"  CODE-LEVEL VALIDATION: {passed} passed, {failed} failed")
print("=" * 76)

section("7 · LIVE-SYSTEM CHECKLIST — cannot be validated by any script")
live_items = [
    "C1's real Space called at least once with a real response (not a stub)",
    "C3's real Space called at least once with a real response (not a stub)",
    "C1's reference distribution built from Dewdu's real held-out scores",
    "C3's reference distribution built from Dulhara's real held-out scores",
    "C4's reference distribution rebuilt from the notebook's real dcar_reference_scores.npy",
    "At least one real clinician verdict recorded via POST /v1/verdict",
    "coverage_report() run against real (not synthetic) calibration + test pairs",
    "The fused composite compared against real clinical outcomes (once labels exist)",
]
for item in live_items:
    print(f"  [ ]  {item}")
print("\n  None of these can be checked from a laptop in isolation. Track them")
print("  explicitly — a green run of THIS script is necessary, not sufficient,")
print("  for calling the fusion model fully validated.")

print(f"\n{'=' * 76}")
import sys
sys.exit(1 if failed else 0)
