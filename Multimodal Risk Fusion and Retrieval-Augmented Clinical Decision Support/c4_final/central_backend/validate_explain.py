"""
validate_explain.py — independent cross-check for CARE-X.

Deliberately standalone, in the same spirit as validate_fusion.py: if the
explanation layer and its test suite share helper code, a bug in the helper
passes both. Everything here is recomputed by hand.
"""

import copy
import json
import sys

import explain

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))


def hr(t):
    print("\n" + "=" * 78)
    print(f"{t}")
    print("=" * 78)


# ── fixture: the REAL live fusion result from the deployed system ───────────
LIVE_FUSION = {
    "composite": 0.8592,
    "tier": "High",
    "band": "RED",
    "confidence": 0.4339,
    "reason": None,
    "renormalised": True,
    "weights": {
        "c1_physiological": 0.3649,
        "c2_behavioral": 0.0,
        "c3_clinical_nlp": 0.6351,
        "c4_demographic": 0.0,
    },
    "contributions": {
        "c1_physiological": 0.3649,
        "c2_behavioral": 0.0,
        "c3_clinical_nlp": 0.4943,
        "c4_demographic": 0.0,
    },
    "harmonisation": {
        "c1_physiological": {"raw": 100.0, "harmonised": 1.0, "drift": False, "note": None},
        "c2_behavioral": {"available": False},
        "c3_clinical_nlp": {"raw": 0.6634, "harmonised": 0.7783, "drift": False, "note": None},
        "c4_demographic": {"available": False},
        "gate": {"passed": True,
                 "usable_modalities": ["c1_physiological", "c3_clinical_nlp"],
                 "rejected": {}, "reason": None},
        "conformal": {"conformal_set": ["Low", "Medium", "High"], "conformal_alpha": 0.1,
                      "conformal_calibrated": False, "conformal_n": 0,
                      "conformal_quantile": None,
                      "conformal_note": "only 0 clinician verdicts available"},
    },
}

LIVE_MODALITIES = {
    "c1_physiological": {"status": "ok", "score": 100.0, "confidence": 0.5, "coverage": 0.5,
                         "age_minutes": 221.4, "fresh": False, "model_version": None,
                         "excluded": False},
    "c2_behavioral": {"status": "absent", "score": None, "excluded": True},
    "c3_clinical_nlp": {"status": "ok", "score": 0.663425, "confidence": 0.078, "coverage": 1.0,
                        "age_minutes": 250.5, "fresh": True,
                        "model_version": "tcwpn-clean-benchmark-36d7413", "excluded": False},
    "c4_demographic": {"status": "absent", "score": None},
}

PLACEHOLDER_REFS = {
    "c1_physiological": {"source": "PLACEHOLDER - replace with C1 held-out scores"},
    "c3_clinical_nlp": {"source": "PLACEHOLDER - replace with C3 held-out scores"},
    "c4_demographic": {"source": "PLACEHOLDER - replace with dcar_reference_scores.npy"},
}

THRESHOLDS = {"Low": 0.0, "Medium": 0.34, "High": 0.67}

# base_m proportional to (AUROC - 0.5). C1 0.6977, C3 0.8989, C4 0.6220 (DCAR,
# measured). C2 is permanently excluded and carries no base weight.
BASE_WEIGHTS = {"c1_physiological": 0.1977, "c3_clinical_nlp": 0.3989,
                "c4_demographic": 0.1220, "c2_behavioral": 0.0}


# ═══════════════════════════════════════════════════════════════════════════
hr("1 · REAL LIVE DATA — does it explain the actual deployed result?")
e = explain.explain_fusion(LIVE_FUSION, LIVE_MODALITIES, THRESHOLDS, PLACEHOLDER_REFS, BASE_WEIGHTS)

check("assessed == True", e["assessed"] is True)
check("composite preserved exactly", e["composite"] == 0.8592, f"got {e['composite']}")
check("tier preserved", e["tier"] == "High")
check("band preserved", e["band"] == "RED")

attr = e["attribution"]
check("only the 2 usable modalities are ranked", len(attr["ranked"]) == 2,
      f"got {[r['modality'] for r in attr['ranked']]}")
check("c3 identified as dominant", attr["dominant"] == "c3_clinical_nlp",
      f"got {attr['dominant']}")

# Hand-computed: 0.4943 / 0.8592 = 0.57530...  -> 57.5%
c3_share = round(0.4943 / 0.8592 * 100, 1)
c1_share = round(0.3649 / 0.8592 * 100, 1)
got_c3 = [r for r in attr["ranked"] if r["modality"] == "c3_clinical_nlp"][0]["share_pct"]
got_c1 = [r for r in attr["ranked"] if r["modality"] == "c1_physiological"][0]["share_pct"]
check(f"c3 share == hand-computed {c3_share}%", got_c3 == c3_share, f"got {got_c3}")
check(f"c1 share == hand-computed {c1_share}%", got_c1 == c1_share, f"got {got_c1}")
check("shares sum to ~100%", abs(got_c3 + got_c1 - 100.0) < 0.2,
      f"got {got_c3 + got_c1}")


# ═══════════════════════════════════════════════════════════════════════════
hr("2 · DROP-ONE COUNTERFACTUALS — verified against hand arithmetic")
cf = e["counterfactuals"]
drop = {d["modality"]: d for d in cf["drop_one"]}

# Drop c1 -> only c3 remains -> weight renormalises to 1.0 -> composite = h_c3
expect_wo_c1 = round(0.7783, 4)
check("drop c1 -> composite == h_c3 == 0.7783",
      drop["c1_physiological"]["composite_without"] == expect_wo_c1,
      f"got {drop['c1_physiological']['composite_without']}")
check("drop c1 -> delta == 0.7783 - 0.8592 == -0.0809",
      drop["c1_physiological"]["delta"] == round(0.7783 - 0.8592, 4),
      f"got {drop['c1_physiological']['delta']}")

# Drop c3 -> only c1 remains -> composite = h_c1 = 1.0
check("drop c3 -> composite == h_c1 == 1.0",
      drop["c3_clinical_nlp"]["composite_without"] == 1.0,
      f"got {drop['c3_clinical_nlp']['composite_without']}")
check("drop c3 -> tier still High (1.0 >= 0.67)",
      drop["c3_clinical_nlp"]["tier_without"] == "High",
      f"got {drop['c3_clinical_nlp']['tier_without']}")
check("drop c1 -> tier still High (0.7783 >= 0.67)",
      drop["c1_physiological"]["tier_without"] == "High",
      f"got {drop['c1_physiological']['tier_without']}")


# ═══════════════════════════════════════════════════════════════════════════
hr("3 · TIER-FLIP POINTS — exact algebra, verified independently")
flips = {(f["modality"], f["target_tier"]): f for f in cf["tier_flip_points"]}

# h_c3* to reach the High cut 0.67:  (0.67 - w_c1*h_c1) / w_c3
#                                  = (0.67 - 0.3649*1.0) / 0.6351 = 0.48009...
expect = round((0.67 - 0.3649 * 1.0) / 0.6351, 4)
got = flips[("c3_clinical_nlp", "High")]["harmonised_value_required"]
check(f"c3 High-flip point == hand-computed {expect}", got == expect, f"got {got}")
check("c3 High-flip is reachable (0 <= x <= 1)",
      flips[("c3_clinical_nlp", "High")]["reachable"] is True)
check("c3 would need to FALL to lose High tier",
      flips[("c3_clinical_nlp", "High")]["direction"] == "would need to FALL")

# h_c1* to reach High cut:  (0.67 - w_c3*h_c3) / w_c1
#                         = (0.67 - 0.6351*0.7783) / 0.3649 = 0.48098...
expect_c1 = round((0.67 - 0.6351 * 0.7783) / 0.3649, 4)
got_c1f = flips[("c1_physiological", "High")]["harmonised_value_required"]
check(f"c1 High-flip point == hand-computed {expect_c1}", got_c1f == expect_c1,
      f"got {got_c1f}")

# Medium cut 0.34: (0.34 - 0.3649)/0.6351 = negative -> unreachable
med = flips[("c3_clinical_nlp", "Medium")]
check("c3 cannot reach Medium cut at any value (unreachable)",
      med["reachable"] is False,
      f"required={med['harmonised_value_required']} reachable={med['reachable']}")
check("thresholds flagged as authoritative when supplied",
      cf["thresholds_authoritative"] is True)


# ═══════════════════════════════════════════════════════════════════════════
hr("4 · RELIABILITY AUDIT — the TIER-MoE gap")
audit = e["reliability_audit"]
findings = {f["modality"]: f for f in audit["findings"]}

# Verdicts now come from the MEASURED inflation ratio, not weight thresholds.
# Hand-computed below in section 12; the old rule gave the opposite answer.
check("c1 verdict is moderately-inflated (1.33x)",
      findings["c1_physiological"]["verdict"] == "moderately-inflated",
      f"got {findings['c1_physiological']['verdict']}")
check("c3 verdict is earned (1.14x, not scarcity)",
      findings["c3_clinical_nlp"]["verdict"] == "earned",
      f"got {findings['c3_clinical_nlp']['verdict']}")
check("verdicts disagree with the old weight-threshold rule",
      findings["c3_clinical_nlp"]["verdict"] != "scarcity-inflated")
check("2 modalities in composite", audit["modalities_in_composite"] == 2)
check("2 modalities missing", audit["modalities_missing"] == 2,
      f"got {audit['modalities_missing']}")
check("c1 detected as stale-but-used (fresh=False)",
      audit["stale_but_used"] == ["c1_physiological"],
      f"got {audit['stale_but_used']}")
check("stale warning is emitted", audit["stale_warning"] is not None)


# ═══════════════════════════════════════════════════════════════════════════
hr("5 · HONESTY LEDGER")
led = e["honesty_ledger"]
issues = " | ".join(i["issue"] for i in led["items"])
check("placeholder reference distributions flagged",
      "PLACEHOLDER" in issues.upper() or "placeholder" in issues.lower(), issues)
check("uncalibrated conformal flagged", "conformal" in issues.lower(), issues)
check("c1 unknown coverage NOT falsely flagged (coverage=0.5 is known)",
      "coverage unknown" not in issues.lower(), issues)
check("c1 missing model_version flagged", "model version" in issues.lower(), issues)
check("highest severity is high", led["highest_severity"] == "high",
      f"got {led['highest_severity']}")
check("items sorted high->low severity",
      [i["severity"] for i in led["items"]] ==
      sorted([i["severity"] for i in led["items"]], key=lambda s: {"high": 0, "medium": 1, "low": 2}[s]))


# ═══════════════════════════════════════════════════════════════════════════
hr("6 · DETERMINISM — same input must always give byte-identical output")
runs = [json.dumps(explain.explain_fusion(copy.deepcopy(LIVE_FUSION),
                                          copy.deepcopy(LIVE_MODALITIES),
                                          THRESHOLDS, PLACEHOLDER_REFS), sort_keys=True)
        for _ in range(5)]
check("5 identical runs produce identical output", len(set(runs)) == 1)
check("no randomness in module source",
      not any(t in open("explain.py").read() for t in ("random.", "uuid4", "shuffle", "time.time")))


# ═══════════════════════════════════════════════════════════════════════════
hr("7 · EDGE CASES — must degrade, never crash")

cases = {
    "no composite (gate refused)": ({"composite": None, "tier": None, "band": "GREY",
                                     "reason": "insufficient evidence: 1 usable modality, need 2",
                                     "weights": {}, "contributions": {}, "harmonisation": {}}, {}),
    "empty dict": ({}, {}),
    "weights present but no harmonisation": ({"composite": 0.5, "tier": "Medium", "band": "AMBER",
                                              "weights": {"c1_physiological": 1.0},
                                              "contributions": {"c1_physiological": 0.5},
                                              "harmonisation": {}}, {}),
    "composite is zero": ({"composite": 0.0, "tier": "Low", "band": "GREEN",
                           "weights": {"c1_physiological": 1.0},
                           "contributions": {"c1_physiological": 0.0},
                           "harmonisation": {"c1_physiological": {"harmonised": 0.0}}}, {}),
    "single modality only": ({"composite": 0.9, "tier": "High", "band": "RED",
                              "weights": {"c3_clinical_nlp": 1.0},
                              "contributions": {"c3_clinical_nlp": 0.9},
                              "harmonisation": {"c3_clinical_nlp": {"harmonised": 0.9}}},
                             {"c3_clinical_nlp": {"confidence": 0.9, "coverage": 1.0,
                                                  "fresh": True, "model_version": "v1"}}),
    "null-ridden modality view": (LIVE_FUSION, {"c1_physiological": {"confidence": None,
                                                                     "coverage": None,
                                                                     "fresh": None},
                                                "c3_clinical_nlp": None}),
    "string numbers instead of floats": ({"composite": "0.75", "tier": "High", "band": "RED",
                                          "weights": {"c1_physiological": "1.0"},
                                          "contributions": {"c1_physiological": "0.75"},
                                          "harmonisation": {"c1_physiological": {"harmonised": "0.75"}}}, {}),
}
for name, (fus, mods) in cases.items():
    try:
        out = explain.explain_fusion(fus, mods, THRESHOLDS, PLACEHOLDER_REFS)
        ok = isinstance(out, dict) and "narrative" in out and out["narrative"]
        check(f"survives: {name}", ok, f"output missing narrative: {out}")
    except Exception as exc:                                     # noqa: BLE001
        check(f"survives: {name}", False, f"{type(exc).__name__}: {exc}")

# no thresholds supplied at all
try:
    out = explain.explain_fusion(LIVE_FUSION, LIVE_MODALITIES, None, PLACEHOLDER_REFS)
    check("survives: no thresholds supplied", True)
    check("flags that thresholds were not authoritative",
          out["counterfactuals"]["thresholds_authoritative"] is False)
    check("emits caveat when thresholds are guessed",
          out["counterfactuals"]["caveat"] is not None)
    check("ledger records the guessed-thresholds limitation",
          any("cut-point" in i["issue"].lower() for i in out["honesty_ledger"]["items"]))
except Exception as exc:                                         # noqa: BLE001
    check("survives: no thresholds supplied", False, f"{type(exc).__name__}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
hr("8 · SAFETY — the refusal case must not read as 'low risk'")
refusal = explain.explain_fusion(
    {"composite": None, "tier": None, "band": "GREY",
     "reason": "insufficient evidence: 1 usable modality, need 2",
     "weights": {}, "contributions": {}, "harmonisation": {}}, {}, THRESHOLDS)
check("refusal marked assessed=False", refusal["assessed"] is False)
check("refusal narrative says it is NOT an assessment of low risk",
      "not an assessment of low risk" in refusal["narrative"].lower(),
      refusal["narrative"])
check("refusal carries the gate's own reason",
      "1 usable modality" in refusal["narrative"])
check("patient text does not imply a clear result",
      "not enough recent information" in refusal["patient_narrative"].lower(),
      refusal["patient_narrative"])


# ═══════════════════════════════════════════════════════════════════════════
hr("9 · PATIENT-SAFE OUTPUT — must leak nothing clinical")
pn = e["patient_narrative"]
leaks = []
for term in ["0.8592", "0.6351", "0.3649", "0.4943", "c1_", "c3_", "weight",
             "clinical notes", "physiological", "composite", "scarcity",
             "0.7783", "conformal", "AUROC", "modality"]:
    if term.lower() in pn.lower():
        leaks.append(term)
check("patient narrative leaks no scores/weights/modality names",
      not leaks, f"LEAKED: {leaks} in {pn!r}")
check("patient narrative is non-empty and human", len(pn) > 40, pn)


# ═══════════════════════════════════════════════════════════════════════════
hr("10 · INLINE SUMMARY — compact form for doctor_timeline")
s = explain.explanation_summary(e)
check("summary has headline", bool(s.get("headline")))
check("summary does NOT carry the full narrative", "narrative" not in s)
check("summary points to the full explanation endpoint",
      "explanation" in (s.get("full_explanation_at") or ""))
check("summary names dominant modality", s.get("dominant_modality") == "c3_clinical_nlp")
check("summary surfaces measured inflation per modality",
      s.get("inflation", {}).get("c1_physiological") is not None,
      f"got {s.get('inflation')}")
check("summary carries minimal sufficient set size",
      s.get("minimal_sufficient_size") == 2, f"got {s.get('minimal_sufficient_size')}")
check("summary surfaces stale list", s.get("stale_but_used") == ["c1_physiological"])
check("summary carries limitation count", s.get("limitations_count", 0) > 0)
check("summary is compact (< 700 chars serialised)",
      len(json.dumps(s)) < 700, f"{len(json.dumps(s))} chars")
check("headline is genuinely short (< 250 chars)",
      len(s["headline"]) < 250, f"{len(s['headline'])} chars: {s['headline']}")
check("headline still carries the tier and a caution/note",
      "High" in s["headline"] and ("aution" in s["headline"] or "Note" in s["headline"]),
      s["headline"])

srefusal = explain.explanation_summary(refusal)
check("summary handles the refusal case", srefusal["assessed"] is False)
check("refusal headline warns it is not low risk",
      "low-risk" in (srefusal.get("headline") or "").lower()
      or "not a low" in (srefusal.get("headline") or "").lower(),
      srefusal.get("headline"))

check("no awkward 'modality/modalities' pluralisation anywhere",
      "modality/modalities" not in json.dumps(e),
      "found 'modality/modalities' in output")


# ═══════════════════════════════════════════════════════════════════════════
hr("11 · MONOTONICITY — a higher modality score must not lower its own share")
prev = None
mono_ok = True
for h3 in [0.2, 0.4, 0.6, 0.8, 1.0]:
    f2 = copy.deepcopy(LIVE_FUSION)
    f2["harmonisation"]["c3_clinical_nlp"]["harmonised"] = h3
    contrib3 = round(0.6351 * h3, 4)
    f2["contributions"]["c3_clinical_nlp"] = contrib3
    f2["composite"] = round(0.3649 * 1.0 + contrib3, 4)
    out = explain.explain_fusion(f2, LIVE_MODALITIES, THRESHOLDS, PLACEHOLDER_REFS)
    share = [r for r in out["attribution"]["ranked"]
             if r["modality"] == "c3_clinical_nlp"][0]["share_pct"]
    if prev is not None and share < prev - 0.01:
        mono_ok = False
    prev = share
check("c3 share rises monotonically as its harmonised score rises", mono_ok)


# ═══════════════════════════════════════════════════════════════════════════
hr("12 · QUANTIFIED INFLATION — verified against independent hand algebra")
inf = e["weight_inflation"]

# Hand-computed from base_m*(AUROC-0.5) and rel = 0.5 + 0.5*conf*cov:
#   eligible panel = c1, c3, c4  (c2 permanently excluded)
#   scarcity = (0.1977+0.3989+0.1220) / (0.1977+0.3989) = 1.20449...
hand_scarcity = round((0.1977 + 0.3989 + 0.1220) / (0.1977 + 0.3989), 3)
check(f"scarcity factor == hand-computed {hand_scarcity}",
      inf["scarcity_factor"] == hand_scarcity, f"got {inf['scarcity_factor']}")
check("inflation is marked quantified", inf["quantified"] is True)
check("c2 excluded from the eligible panel (never 'missing')",
      "c2_behavioral" not in inf["eligible_panel"], f"got {inf['eligible_panel']}")

rows = {r["modality"]: r for r in inf["per_modality"]}
rel_c1 = 0.5 + 0.5 * 0.5 * 0.5          # 0.625
rel_c3 = 0.5 + 0.5 * 0.078 * 1.0        # 0.539
sum_all = 0.1977 + 0.3989 + 0.1220
hand_ideal_c1 = round(0.1977 / sum_all, 4)
hand_ideal_c3 = round(0.3989 / sum_all, 4)
check(f"c1 full-panel ideal weight == {hand_ideal_c1}",
      rows["c1_physiological"]["full_panel_ideal_weight"] == hand_ideal_c1,
      f"got {rows['c1_physiological']['full_panel_ideal_weight']}")
check(f"c3 full-panel ideal weight == {hand_ideal_c3}",
      rows["c3_clinical_nlp"]["full_panel_ideal_weight"] == hand_ideal_c3,
      f"got {rows['c3_clinical_nlp']['full_panel_ideal_weight']}")

hand_infl_c1 = round(0.3649 / hand_ideal_c1, 3)
hand_infl_c3 = round(0.6351 / hand_ideal_c3, 3)
check(f"c1 inflation == hand-computed {hand_infl_c1}x",
      abs(rows["c1_physiological"]["inflation"] - hand_infl_c1) < 0.002,
      f"got {rows['c1_physiological']['inflation']}")
check(f"c3 inflation == hand-computed {hand_infl_c3}x",
      abs(rows["c3_clinical_nlp"]["inflation"] - hand_infl_c3) < 0.002,
      f"got {rows['c3_clinical_nlp']['inflation']}")
check("c1 is MORE inflated than c3 (opposite of the old threshold rule)",
      rows["c1_physiological"]["inflation"] > rows["c3_clinical_nlp"]["inflation"])

# The decomposition must multiply back out exactly.
for m, r in rows.items():
    prod = r["scarcity_factor"] * r["relative_reliability_factor"]
    check(f"{m}: scarcity x rel_factor == inflation",
          abs(prod - r["inflation"]) < 0.005, f"{prod:.4f} vs {r['inflation']}")

# Without base weights the layer must degrade, not guess.
e_nb = explain.explain_fusion(LIVE_FUSION, LIVE_MODALITIES, THRESHOLDS, PLACEHOLDER_REFS, None)
check("without base weights, inflation is not quantified",
      e_nb["weight_inflation"]["quantified"] is False)
check("without base weights, a reason is stated not a value invented",
      e_nb["weight_inflation"]["unavailable_reason"] is not None
      and e_nb["weight_inflation"]["scarcity_factor"] is None)
check("without base weights, verdicts fall back to 'unquantified'",
      all(f["verdict"] == "unquantified" for f in e_nb["reliability_audit"]["findings"]))


hr("13 · SUFFICIENCY — necessity, decision relevance, minimal sufficient sets")
su = e["sufficiency"]
per = {p["modality"]: p for p in su["per_modality"]}

# Drop c1 -> 0.7783 -> still High. Drop c3 -> 1.0 -> still High. Neither necessary.
check("c1 is not necessary (tier survives its removal)",
      per["c1_physiological"]["necessary"] is False)
check("c3 is not necessary (tier survives its removal)",
      per["c3_clinical_nlp"]["necessary"] is False)

# Alone: h_c1 = 1.0 -> High; h_c3 = 0.7783 -> High. Both alone-sufficient.
check("c1 alone-sufficient (h=1.0 -> High)", per["c1_physiological"]["alone_sufficient"] is True)
check("c3 alone-sufficient (h=0.7783 -> High)", per["c3_clinical_nlp"]["alone_sufficient"] is True)

# Decision relevance: composite over h_c3 in [0,1] spans [0.3649, 1.0] -> Medium..High
check("c3 is decision-relevant (tier varies over its achievable range)",
      per["c3_clinical_nlp"]["decision_relevant"] is True)
check("c1 is decision-relevant", per["c1_physiological"]["decision_relevant"] is True)
check("tier range is ordered by threshold, not alphabetically",
      per["c3_clinical_nlp"]["attainable_tier_range"] == ["Medium", "High"],
      f"got {per['c3_clinical_nlp']['attainable_tier_range']}")

# MSS: neither singleton pins the tier (the free one can swing it), so size 2.
check("minimal sufficient set requires both modalities",
      su["minimal_sufficient_size"] == 2, f"got {su['minimal_sufficient_size']}")
check("exactly one minimal sufficient set at that size",
      len(su["minimal_sufficient_sets"]) == 1)
check("no decision-irrelevant modality in this case",
      su["decision_irrelevant"] == [], f"got {su['decision_irrelevant']}")
check("definitions distinguish alone_sufficient from minimal_sufficient_set",
      "minimal_sufficient_set" in su["definitions"]["alone_sufficient"])
check("scope caveat names the gate-boundary limitation",
      "gate" in su["scope_caveat"].lower() or "eject" in su["scope_caveat"].lower())

# A case where one modality genuinely cannot move the tier.
tiny = {"composite": 0.95, "tier": "High", "band": "RED", "renormalised": True,
        "weights": {"c1_physiological": 0.02, "c3_clinical_nlp": 0.98},
        "contributions": {"c1_physiological": 0.02, "c3_clinical_nlp": 0.93},
        "harmonisation": {"c1_physiological": {"harmonised": 1.0},
                          "c3_clinical_nlp": {"harmonised": 0.949}}}
et = explain.explain_fusion(tiny, {}, THRESHOLDS, None, BASE_WEIGHTS)
check("a tiny-weight modality is flagged decision-irrelevant",
      "c1_physiological" in et["sufficiency"]["decision_irrelevant"],
      f"got {et['sufficiency']['decision_irrelevant']}")
check("narrative states that it cannot change the tier",
      "would change this tier" in et["narrative"] or "cannot" in et["narrative"].lower(),
      et["narrative"])
check("a single modality can be a minimal sufficient set",
      et["sufficiency"]["minimal_sufficient_size"] == 1,
      f"got {et['sufficiency']['minimal_sufficient_size']}")


hr("RESULT")
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\n  FAILURES:")
    for f in FAIL:
        print(f"    - {f}")
print("=" * 78)
sys.exit(1 if FAIL else 0)
