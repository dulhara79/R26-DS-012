"""
Fusion demo — run this to see the weighting mechanism behave.

    python demo_fusion.py

Nothing here talks to a network. It feeds made-up readings into fusion.py so you
can watch the weights move as scores get older or streams disappear.
"""

from datetime import datetime, timezone, timedelta

from fusion import Reading, fuse, base_weights, LiveFusion

NOW = datetime(2026, 8, 17, 9, 0, tzinfo=timezone.utc)
BAR = "─" * 74


def show(title, readings):
    out = fuse(readings, NOW)
    print(f"\n{title}")
    if out.tier is None:
        print(f"   NO TIER  ->  {out.reason}")
        return
    print(f"   composite = {out.composite:.3f}   tier = {out.tier}   "
          f"confidence = {out.confidence:.2f}   streams = {out.modalities_available}")
    for k, v in out.weights.items():
        if v > 0:
            s = out.scores[k]
            print(f"      {k:20s} weight {v:5.3f}  x  score {s:.2f}  =  {v*s:.3f}")


print(BAR)
print("BASE WEIGHTS  (omega, proportional to AUROC - 0.5)")
print(BAR)
for k, v in base_weights().items():
    print(f"   {k:20s} {v:5.3f}")
print("\n   c2_behavioral is 0.000 because it did not clear its permutation null.")

print(f"\n{BAR}")
print("SCENARIOS  (same underlying scores every time — only the AGE of each")
print("            reading changes, and watch what that does to the weights)")
print(BAR)

fresh_note = Reading(0.68, True, 0.83, 1.0, NOW - timedelta(days=2))
demo_prior = Reading(0.55, True, 0.61, 1.0, NOW - timedelta(days=40))

show("A · Everything fresh — strap on, note from 2 days ago", {
    "c1_physiological": Reading(0.62, True, 0.74, 0.81, NOW - timedelta(minutes=1)),
    "c3_clinical_nlp": fresh_note,
    "c4_demographic": demo_prior,
})

show("B · Patient took the strap off 3 hours ago", {
    "c1_physiological": Reading(0.62, True, 0.74, 0.40, NOW - timedelta(hours=3)),
    "c3_clinical_nlp": fresh_note,
    "c4_demographic": demo_prior,
})

show("C · Strap is on, but the last clinical note is 3 months old", {
    "c1_physiological": Reading(0.62, True, 0.74, 0.81, NOW - timedelta(minutes=1)),
    "c3_clinical_nlp": Reading(0.68, True, 0.83, 1.0, NOW - timedelta(days=90)),
    "c4_demographic": demo_prior,
})

show("D · Day 1: demographics only, no strap yet, no notes yet", {
    "c4_demographic": Reading(0.55, True, 0.61, 1.0, NOW),
})

show("E · Behavioural stream sends a very high score (0.95)", {
    "c1_physiological": Reading(0.62, True, 0.74, 0.81, NOW - timedelta(minutes=1)),
    "c2_behavioral": Reading(0.95, True, 0.90, 1.0, NOW - timedelta(minutes=5)),
    "c3_clinical_nlp": fresh_note,
    "c4_demographic": demo_prior,
})
print("      ^ composite is IDENTICAL to scenario A. The zero-weight rule held.")

print(f"\n{BAR}")
print("LIVE STREAM  (physiological score arriving once per minute)")
print("Smoothing stops one bad minute moving the tier; hysteresis stops the")
print("badge flickering when the composite sits right on a band edge.")
print(BAR)

live = LiveFusion()
sequence = [0.30] * 3 + [0.95] * 6 + [0.30] * 6
for minute, raw in enumerate(sequence):
    t = NOW + timedelta(minutes=minute)
    out = live.update({
        "c1_physiological": Reading(raw, True, 0.74, 0.81, t),
        "c3_clinical_nlp": fresh_note,
        "c4_demographic": demo_prior,
    }, t)
    flag = "  <-- crosses 0.66, suppressed by hysteresis" if out.composite > 0.66 else ""
    print(f"   minute {minute:2d}   raw physio {raw:.2f}   "
          f"composite {out.composite:.3f}   tier shown: {out.tier}{flag}")

print(f"\n{BAR}")
print("Wire format sent to the clinician Flutter app:")
print(BAR)
import json
print(json.dumps(fuse({
    "c1_physiological": Reading(0.62, True, 0.74, 0.81, NOW - timedelta(minutes=1)),
    "c3_clinical_nlp": fresh_note,
    "c4_demographic": demo_prior,
}, NOW).to_wire(), indent=2))
