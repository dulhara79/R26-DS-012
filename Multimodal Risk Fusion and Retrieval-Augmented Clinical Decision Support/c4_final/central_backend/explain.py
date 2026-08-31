"""
CARE-X — Counterfactual Audit of Reliability-weighted Evidence
=============================================================
Explanation layer for the RAGF composite. Component 4 · R26-DS-012.

WHY THIS IS NOT SHAP / LIME / DiCE
----------------------------------
Those methods exist to reverse-engineer a model whose internals are opaque:
they perturb inputs, observe outputs, and *estimate* an attribution. The RAGF
fusion layer is not opaque. It is closed-form:

    composite = SUM_over_usable( weight_m * harmonised_m )
    weights renormalise to exactly 1.0 over the usable set

Verified against live output: w_c1=0.3649 x h=1.0 = 0.3649, w_c3=0.6351 x
h=0.7783 = 0.4943, sum = 0.8592 = the reported composite.

Because the form is known, every quantity SHAP would approximate can instead be
read or solved exactly. That is a strictly stronger guarantee, not a weaker one:
a DiCE counterfactual is the result of a search and may be a local optimum; the
counterfactuals here are algebraic solutions and are unique.

WHAT IS NOVEL HERE
------------------
The literature on reliability-weighted late fusion (ACE 2026, TIER-MoE 2026,
QMF, PDF, Cred-MF) computes per-modality reliability and uses it to set weights.
Those papers treat the weight vector itself as the explanation. None of them
explain to a clinician *why a particular patient's weights came out that way*,
and none separate the two cases TIER-MoE explicitly names as conflated:

    (a) this modality mattered because it was informative
    (b) this modality mattered because everything else was missing

CARE-X addresses that gap with five layers:

    L1 attribution        who actually drove this score
    L2 weight provenance  why each weight is what it is, decomposed to causes
    L3 counterfactuals    exact drop-one deltas and exact tier-flip thresholds
    L4 reliability audit  separates (a) from (b) above
    L5 honesty ledger     what the system does not know, surfaced not buried

DESIGN RULES
------------
1. DETERMINISTIC. No LLM, no sampling, no randomness. The same fusion result
   must always produce the identical explanation, or a clinician who reopens
   yesterday's assessment sees a different rationale and stops trusting it.
   This mirrors the same rule support_bank.py applies to support selection.

2. STORED DATA ONLY. Everything is derived from the persisted FusionResult
   (weights, contributions, harmonisation) plus the modality view already built
   by doctor_timeline. No component is re-called. An explanation must be
   reproducible months later when those Spaces may be gone.

3. A GAP IS STATED, NEVER FILLED. Where a value is unknown (C1 coverage), a
   distribution is synthetic (all three reference sets today), or a probability
   is uncalibrated (C3), the explanation says so. Same rule modality_clients.py
   applies to a timeout: absent is absent, never zero.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# ── module contract ─────────────────────────────────────────────────────────
EXPLAINER_VERSION = "care-x-v1.1"

# Human-facing names. The backend's internal modality keys are stable; these
# are for prose only and are never used as lookup keys.
MODALITY_LABEL = {
    "c1_physiological": "physiological (wearable)",
    "c2_behavioral": "behavioural (smartphone)",
    "c3_clinical_nlp": "clinical notes",
    "c4_demographic": "demographic and contextual",
}

# Tier cut-points on the composite. Passed in by the caller where possible so
# this module never becomes a second source of truth that can drift from
# fusion.py — see _resolve_thresholds.
DEFAULT_TIER_THRESHOLDS = {"Low": 0.0, "Medium": 0.34, "High": 0.67}

_EPSILON = 1e-9


# ── helpers ─────────────────────────────────────────────────────────────────
def _f(value: Any) -> Optional[float]:
    """float() that returns None instead of raising.

    Also rejects NaN and infinity. A stored JSON field can carry either after a
    bad upstream computation, and both poison every downstream comparison
    silently: NaN fails all ordering tests, so a tier check against it returns
    False rather than raising, and the explanation would quietly claim a tier
    was unreachable when the real problem is a corrupt input. None is an honest
    gap; NaN is a lie that looks like a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):    # NaN / +-inf
        return None
    return f


def _d(value: Any) -> Dict[str, Any]:
    """Coerce to a dict, or an empty one.

    `x or {}` is NOT sufficient: a non-empty string is truthy, so a stored field
    that came back as "junk" instead of an object passes the `or` and then
    raises on .get(). Fuzzing over 3000 malformed inputs found exactly this in
    four separate call sites."""
    return value if isinstance(value, dict) else {}


def _pct(x: float) -> float:
    return round(x * 100.0, 1)


def _resolve_thresholds(supplied: Optional[Dict[str, float]]) -> Tuple[Dict[str, float], bool]:
    """Prefer thresholds handed in by the caller (which should read them from
    fusion.py) over this module's defaults. Returns (thresholds, is_authoritative)
    so the explanation can flag when it is guessing the cut-points."""
    if supplied:
        return dict(supplied), True
    try:                                            # best effort, never fatal
        import fusion  # type: ignore
        for attr in ("TIER_THRESHOLDS", "TIERS_THRESHOLDS", "BANDS"):
            t = getattr(fusion, attr, None)
            if isinstance(t, dict) and t:
                return dict(t), True
    except Exception:                               # noqa: BLE001
        pass
    return dict(DEFAULT_TIER_THRESHOLDS), False


def _tier_for(composite: float, thresholds: Dict[str, float]) -> Optional[str]:
    """Highest tier whose cut-point the composite clears."""
    best, best_cut = None, None
    for tier, cut in thresholds.items():
        c = _f(cut)
        if c is None or composite + _EPSILON < c:
            continue
        if best_cut is None or c > best_cut:
            best, best_cut = tier, c
    return best


def _tier_order(thresholds: Dict[str, float]) -> List[str]:
    """Tier names in ascending cut-point order. Alphabetical sorting would put
    'High' before 'Medium' and make any range read backwards."""
    return [t for t, _ in sorted(((k, _f(v) or 0.0) for k, v in _d(thresholds).items()),
                                 key=lambda kv: kv[1])]


def _usable(weights: Dict[str, float], harmonisation: Dict[str, Any]) -> List[str]:
    """A modality is in the composite iff it carries non-zero weight AND has a
    harmonised value. Weight alone is not enough: a gated-out modality can
    still appear in the weights dict with 0.0."""
    out = []
    for m, w in (weights or {}).items():
        wf = _f(w)
        if wf is None or wf <= _EPSILON:
            continue
        h = _d(harmonisation).get(m)
        if isinstance(h, dict) and _f(h.get("harmonised")) is not None:
            out.append(m)
    return sorted(out)


# ── L1 · attribution ────────────────────────────────────────────────────────
def _layer_attribution(contributions: Dict[str, float], composite: float,
                       usable: List[str]) -> Dict[str, Any]:
    """Who actually drove this score.

    Share is contribution / composite, i.e. the fraction of the final number
    this modality is responsible for. Reported only for modalities in the
    composite; a modality with zero weight has no share, which is different
    from having a share of zero.
    """
    rows = []
    for m in usable:
        c = _f(_d(contributions).get(m)) or 0.0
        share = (c / composite) if composite and abs(composite) > _EPSILON else None
        rows.append({
            "modality": m,
            "label": MODALITY_LABEL.get(m, m),
            "contribution": round(c, 4),
            "share_pct": _pct(share) if share is not None else None,
        })
    rows.sort(key=lambda r: r["contribution"], reverse=True)
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return {
        "ranked": rows,
        "dominant": rows[0]["modality"] if rows else None,
        "note": ("Share is the fraction of the composite this modality produced. "
                 "It is read from the stored contribution, not estimated."),
    }


# ── L2 · weight provenance ──────────────────────────────────────────────────
def _layer_weight_provenance(weights: Dict[str, float], usable: List[str],
                             modality_view: Dict[str, Any],
                             renormalised: Optional[bool],
                             excluded: List[str], absent: List[str]) -> Dict[str, Any]:
    """Why each weight is the number it is.

    A weight in RAGF is not a statement about which model is better. It is a
    product of (i) the model's measured discrimination, (ii) how reliable THIS
    reading is, and (iii) how many other modalities survived the gate. A
    clinician reading "clinical notes 0.64" will assume (i) alone unless told
    otherwise. Separating the three is the point of this layer.
    """
    rows = []
    for m in usable:
        mv = _d(_d(modality_view).get(m))
        conf, cov = _f(mv.get("confidence")), _f(mv.get("coverage"))
        # The reliability multiplier the fusion service applies. Recorded here
        # for transparency; the weight itself is read from storage, never
        # recomputed, so this cannot silently disagree with the stored value.
        mult = (0.5 + 0.5 * conf * cov) if (conf is not None and cov is not None) else None
        drivers = []
        if conf is not None and conf < 0.25:
            drivers.append(f"low reading confidence ({conf:.3f}) held this weight down")
        if cov is not None and cov < 1.0:
            drivers.append(f"partial coverage ({cov:.2f}) held this weight down")
        if cov is None:
            drivers.append("coverage was not reported by the component, so no coverage "
                           "penalty could be applied — the weight may be optimistic")
        rows.append({
            "modality": m,
            "label": MODALITY_LABEL.get(m, m),
            "final_weight": round(_f(weights.get(m)) or 0.0, 4),
            "reading_confidence": conf,
            "coverage": cov,
            "reliability_multiplier": round(mult, 4) if mult is not None else None,
            "drivers": drivers,
        })
    rows.sort(key=lambda r: r["final_weight"], reverse=True)

    renorm_note = None
    if len(usable) < len(MODALITY_LABEL):
        missing = len(MODALITY_LABEL) - len(usable)
        renorm_note = (
            f"{missing} of {len(MODALITY_LABEL)} modalities did not enter the composite, "
            f"so the remaining {len(usable)} were renormalised to sum to 1.0. Each surviving "
            f"weight is therefore larger than it would be in a full four-modality assessment. "
            f"A high weight here reflects scarcity as much as strength.")
    return {
        "per_modality": rows,
        "renormalised": bool(renormalised) if renormalised is not None else (len(usable) < len(MODALITY_LABEL)),
        "renormalisation_note": renorm_note,
        "excluded_by_policy": excluded,
        "absent": absent,
    }


# ── L3 · exact counterfactuals ──────────────────────────────────────────────
def _layer_counterfactuals(weights: Dict[str, float], harmonisation: Dict[str, Any],
                           composite: float, usable: List[str],
                           thresholds: Dict[str, float],
                           thresholds_authoritative: bool) -> Dict[str, Any]:
    """Drop-one deltas and exact tier-flip thresholds.

    DROP-ONE. Removing modality m and renormalising the survivors is exact
    arithmetic on the stored normalised weights: w'_k = w_k / (1 - w_m). No
    re-call and no re-fit is needed, and the result is not an approximation.

    TIER FLIP. composite = SUM w_k h_k is linear in each h_m, so the harmonised
    value at which the composite crosses a cut-point T solves in one step:

        h_m* = ( T - SUM_{k != m} w_k h_k ) / w_m

    If h_m* falls outside [0, 1] the modality CANNOT flip the tier at any
    achievable value. That is a clinically meaningful statement and it is the
    kind of guarantee a search-based counterfactual method cannot give.
    """
    h = {m: (_f(_d(_d(harmonisation).get(m)).get("harmonised")) or 0.0) for m in usable}
    w = {m: (_f(_d(weights).get(m)) or 0.0) for m in usable}

    drop_one = []
    for m in usable:
        remaining = [k for k in usable if k != m]
        if not remaining:
            drop_one.append({
                "modality": m, "label": MODALITY_LABEL.get(m, m),
                "composite_without": None, "delta": None, "tier_without": None,
                "note": ("Removing this modality leaves no usable evidence — the gate "
                         "would refuse to produce a composite at all."),
            })
            continue
        denom = sum(w[k] for k in remaining)
        if denom <= _EPSILON:
            continue
        new_comp = sum((w[k] / denom) * h[k] for k in remaining)
        drop_one.append({
            "modality": m,
            "label": MODALITY_LABEL.get(m, m),
            "composite_without": round(new_comp, 4),
            "delta": round(new_comp - composite, 4),
            "tier_without": _tier_for(new_comp, thresholds),
            "note": None,
        })
    drop_one.sort(key=lambda r: abs(r["delta"]) if r["delta"] is not None else -1, reverse=True)

    flips = []
    for m in usable:
        if w[m] <= _EPSILON:
            continue
        rest = sum(w[k] * h[k] for k in usable if k != m)
        for tier, cut in sorted(thresholds.items(), key=lambda kv: _f(kv[1]) or 0.0):
            T = _f(cut)
            if T is None:
                continue
            needed = (T - rest) / w[m]
            reachable = -_EPSILON <= needed <= 1.0 + _EPSILON
            flips.append({
                "modality": m,
                "label": MODALITY_LABEL.get(m, m),
                "target_tier": tier,
                "target_cut": round(T, 4),
                "harmonised_value_required": round(needed, 4),
                "current_harmonised": round(h[m], 4),
                "reachable": reachable,
                "direction": ("would need to RISE" if needed > h[m] else "would need to FALL"),
            })
    return {
        "drop_one": drop_one,
        "tier_flip_points": flips,
        "thresholds_used": {k: _f(v) for k, v in thresholds.items()},
        "thresholds_authoritative": thresholds_authoritative,
        "method_note": ("Exact algebraic solutions, not a search. The fusion is linear in "
                        "each harmonised score, so the flip point is solved in closed form "
                        "and is unique. 'reachable: false' means no achievable value of that "
                        "modality can move the tier."),
        "caveat": (None if thresholds_authoritative else
                   "Tier cut-points were not supplied by the fusion service and fall back to "
                   "module defaults. Flip points are arithmetically correct but the tier "
                   "labels attached to them may not match the deployed cut-points."),
    }


# ── base weights ────────────────────────────────────────────────────────────
def load_base_weights() -> Optional[Dict[str, float]]:
    """Pull base (reliability-free) modality weights from the fusion service.

    These are needed for the scarcity term of the inflation measure: without
    them the per-modality reliability ratio is still computable, but the
    panel-level scarcity factor is not, because it depends on the base weight
    of modalities that are ABSENT — a quantity nothing in the stored fusion
    result can recover. Returns None rather than a guess.
    """
    import os
    import sys
    from pathlib import Path

    # fusion.py is not on the Python path by default — it lives in a sibling
    # folder, not central_backend/. Mirror the exact resolution fusion_client.py
    # already uses so both modules always agree on where to look.
    fusion_dir = Path(os.getenv("FUSION_SERVICE_DIR",
                                Path(__file__).resolve().parent.parent / "fusion_service"))
    if str(fusion_dir) not in sys.path:
        sys.path.insert(0, str(fusion_dir))

    try:
        import fusion  # type: ignore
        bw = getattr(fusion, "base_weights", None)
        if callable(bw):
            out = {k: v for k, v in bw().items() if _f(v) is not None}
            return out or None
        if isinstance(bw, dict) and bw:
            return {k: v for k, v in bw.items() if _f(v) is not None}
    except Exception:                                       # noqa: BLE001
        pass
    return None


# ── L4a · quantified weight inflation ───────────────────────────────────────
def _layer_inflation(weights: Dict[str, float], usable: List[str],
                     modality_view: Dict[str, Any],
                     base_weights: Optional[Dict[str, float]],
                     excluded: List[str]) -> Dict[str, Any]:
    """How much of each weight is evidence, and how much is scarcity.

    RAGF sets w_m proportional to base_m * rel_m, renormalised over the usable
    set U. Comparing that against the weight the modality would carry in a full
    panel at ideal reliability gives an inflation ratio that factors EXACTLY:

        inflation(m) = w_m^obs / w_m^ideal
                     = ( SUM_ALL base / SUM_U base )      <- scarcity, panel-level
                       * ( rel_m / mean_rel_U )           <- relative reliability

    where mean_rel_U is the base-weighted mean reliability of the survivors.
    Verified algebraically and numerically against the deployed result.

    This replaces a boolean threshold rule, and the replacement matters: on real
    data the threshold rule flagged the clinical-notes modality as inflated when
    the measured ratios show the physiological modality is the more inflated of
    the two (1.33x vs 1.14x). Hand-picked cut-offs encode an intuition; this
    encodes the arithmetic.

    'ALL' excludes permanently-excluded modalities: a modality that can never
    enter the panel is not a missing one, and counting it would inflate the
    scarcity term forever.
    """
    rel: Dict[str, Optional[float]] = {}
    for m in usable:
        mv = _d(_d(modality_view).get(m))
        conf, cov = _f(mv.get("confidence")), _f(mv.get("coverage"))
        rel[m] = (0.5 + 0.5 * conf * cov) if (conf is not None and cov is not None) else None

    eligible = [m for m in MODALITY_LABEL if m not in excluded]
    have_base = bool(base_weights) and all(
        _f(_d(base_weights).get(m)) is not None for m in eligible)

    scarcity, mean_rel = None, None
    if have_base:
        sum_all = sum(_f(base_weights[m]) or 0.0 for m in eligible)
        sum_u = sum(_f(_d(base_weights).get(m)) or 0.0 for m in usable)
        if sum_u > _EPSILON:
            scarcity = sum_all / sum_u
            num = sum((_f(_d(base_weights).get(m)) or 0.0) * (rel[m] or 0.0)
                      for m in usable if rel[m] is not None)
            den = sum(_f(_d(base_weights).get(m)) or 0.0
                      for m in usable if rel[m] is not None)
            mean_rel = (num / den) if den > _EPSILON else None

    rows = []
    for m in usable:
        w_obs = _f(_d(weights).get(m))
        rel_factor = ((rel[m] / mean_rel) if (rel[m] is not None and mean_rel
                                              and mean_rel > _EPSILON) else None)
        w_ideal, infl = None, None
        if have_base and scarcity is not None:
            sum_all = sum(_f(base_weights[k]) or 0.0 for k in eligible)
            if sum_all > _EPSILON:
                w_ideal = (_f(_d(base_weights).get(m)) or 0.0) / sum_all
                if w_ideal > _EPSILON and w_obs is not None:
                    infl = w_obs / w_ideal

        if infl is None:
            reading = ("Inflation could not be quantified: base weights for the absent "
                       "modalities are unavailable, so the scarcity term is unknown.")
        elif infl >= 1.5:
            reading = (f"This weight is {infl:.2f}x what it would be in a full panel at ideal "
                       f"reliability. Most of that is structural, not evidential.")
        elif infl >= 1.15:
            reading = (f"This weight is moderately inflated ({infl:.2f}x full-panel ideal), "
                       f"largely because other modalities are missing.")
        elif infl >= 0.85:
            reading = f"This weight is close to its full-panel ideal ({infl:.2f}x)."
        else:
            reading = (f"This weight is SUPPRESSED relative to a full panel ({infl:.2f}x) — "
                       f"the reading's own reliability held it down.")
        rows.append({
            "modality": m, "label": MODALITY_LABEL.get(m, m),
            "observed_weight": round(w_obs, 4) if w_obs is not None else None,
            "full_panel_ideal_weight": round(w_ideal, 4) if w_ideal is not None else None,
            "inflation": round(infl, 3) if infl is not None else None,
            "scarcity_factor": round(scarcity, 3) if scarcity is not None else None,
            "relative_reliability_factor": round(rel_factor, 3) if rel_factor is not None else None,
            "reliability": round(rel[m], 4) if rel[m] is not None else None,
            "interpretation": reading,
        })
    rows.sort(key=lambda r: (r["inflation"] is None, -(r["inflation"] or 0)))

    return {
        "per_modality": rows,
        "scarcity_factor": round(scarcity, 3) if scarcity is not None else None,
        "mean_reliability_of_survivors": round(mean_rel, 4) if mean_rel is not None else None,
        "quantified": have_base and scarcity is not None,
        "eligible_panel": eligible,
        "method": ("inflation(m) = observed weight / full-panel-ideal weight, factoring exactly "
                   "into a panel-level scarcity term and a per-modality relative-reliability "
                   "term. Replaces threshold heuristics."),
        "unavailable_reason": (None if (have_base and scarcity is not None) else
                               "base_weights for the full eligible panel were not supplied; "
                               "the scarcity term depends on the base weight of ABSENT "
                               "modalities and cannot be recovered from a stored result."),
    }


# ── L4b · necessity, sufficiency, decision relevance ────────────────────────
def _layer_sufficiency(weights: Dict[str, float], harmonisation: Dict[str, Any],
                       composite: float, usable: List[str],
                       thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Formal sufficiency for the tier, in the abductive-XAI sense.

    Attribution magnitude and DECISION RELEVANCE are orthogonal, and only the
    second is actionable. A modality can produce 40% of the composite while no
    achievable value of it changes the tier; SHAP would rank it important, and
    for this decision it is not. Three distinct questions are answered:

      NECESSARY            removing m changes the tier (others at actual values)
      ALONE-SUFFICIENT     if m were the only modality, the tier is unchanged
      DECISION-RELEVANT    some h_m in [0,1] changes the tier, others held fixed

    and one set-level question:

      MINIMAL SUFFICIENT SET   smallest S whose actual values pin the tier no
                               matter what the modalities outside S report

    The last is the formal 'sufficient reason' of abductive XAI (Marques-Silva,
    Ignatiev, Darwiche), which carries a logical guarantee that a SHAP ranking
    does not. Because the composite is linear in each h given a fixed usable
    set, its extremes over the free variables are attained at h = 0 and h = 1,
    so S is checked in one step instead of searched.

    SCOPE. This holds the usable SET fixed. That is sound for score-valued
    counterfactuals, because the gate keys on status, freshness and coverage,
    not on the score. It is NOT sound for counterfactuals over confidence,
    coverage or freshness: those can eject a modality, which triggers
    renormalisation and moves every other weight discontinuously. Reasoning
    across that gate boundary is out of scope here and is stated as such.
    """
    import itertools

    h = {m: (_f(_d(_d(harmonisation).get(m)).get("harmonised")) or 0.0) for m in usable}
    w = {m: (_f(_d(weights).get(m)) or 0.0) for m in usable}
    actual_tier = _tier_for(composite, thresholds)

    per = []
    for m in usable:
        rest_h = sum(w[k] * h[k] for k in usable if k != m)
        rest_w = sum(w[k] for k in usable if k != m)

        if rest_w > _EPSILON:
            drop_comp = sum((w[k] / rest_w) * h[k] for k in usable if k != m)
            drop_tier = _tier_for(drop_comp, thresholds)
            necessary = (drop_tier != actual_tier)
        else:
            drop_comp, drop_tier, necessary = None, None, None

        alone_tier = _tier_for(h[m], thresholds)
        lo_tier = _tier_for(rest_h, thresholds)
        hi_tier = _tier_for(rest_h + w[m], thresholds)
        per.append({
            "modality": m, "label": MODALITY_LABEL.get(m, m),
            "necessary": necessary,
            "alone_sufficient": (alone_tier == actual_tier),
            "tier_if_alone": alone_tier,
            "decision_relevant": (lo_tier != hi_tier),
            "attainable_tier_range": [t for t in _tier_order(thresholds)
                                      if t in {lo_tier, hi_tier}],
            "tier_without": drop_tier,
            "composite_without": round(drop_comp, 4) if drop_comp is not None else None,
        })

    minimal_sets, checked = [], []
    for r in range(0, len(usable) + 1):
        for S in itertools.combinations(sorted(usable), r):
            lo = sum(w[k] * h[k] for k in S)
            hi = lo + sum(w[j] for j in usable if j not in S)
            if _tier_for(lo, thresholds) == _tier_for(hi, thresholds) == actual_tier:
                minimal_sets.append({
                    "modalities": list(S),
                    "labels": [MODALITY_LABEL.get(k, k) for k in S],
                    "composite_range": [round(lo, 4), round(hi, 4)],
                })
            checked.append(S)
        if minimal_sets:
            break

    irrelevant = [p["modality"] for p in per if p["decision_relevant"] is False]
    return {
        "tier": actual_tier,
        "per_modality": per,
        "minimal_sufficient_sets": minimal_sets,
        "minimal_sufficient_size": (len(minimal_sets[0]["modalities"])
                                    if minimal_sets else None),
        "decision_irrelevant": irrelevant,
        "subsets_checked": len(checked),
        "definitions": {
            "necessary": "removing it changes the tier (others at their actual values)",
            "alone_sufficient": ("if it were the ONLY modality present (its weight "
                                 "renormalised to 1.0), the tier would be unchanged. Distinct "
                                 "from minimal_sufficient_set, which keeps every weight as-is "
                                 "and lets the other modalities' scores range freely"),
            "decision_relevant": "some achievable value of it changes the tier",
            "minimal_sufficient_set": ("smallest set whose actual values pin the tier "
                                       "regardless of what the others report"),
        },
        "scope_caveat": ("Holds the usable SET fixed. Sound for score counterfactuals; NOT "
                         "sound for counterfactuals over confidence, coverage or freshness, "
                         "which can eject a modality and discontinuously renormalise every "
                         "other weight."),
    }


# ── L4 · counterfactual reliability audit ───────────────────────────────────
def _layer_reliability_audit(weights: Dict[str, float], usable: List[str],
                             modality_view: Dict[str, Any],
                             drop_one: List[Dict[str, Any]],
                             excluded: List[str], absent: List[str],
                             inflation: Dict[str, Any],
                             sufficiency: Dict[str, Any]) -> Dict[str, Any]:
    """Separates 'this modality was informative' from 'this modality was alone'.

    TIER-MoE (arXiv 2607.27289) names this exact conflation as an open problem:
    global modality strength does not describe subject-level reliability, and
    subject-level reliability does not describe complementary value. A weight of
    0.64 means something completely different when three other modalities were
    present than when two were missing, and current systems present both
    identically.

    The test used here: a modality is SCARCITY-INFLATED when it carries a large
    weight while the reading itself is weakly reliable. That is the case where a
    clinician should discount the number.
    """
    findings, n_absent = [], len(excluded) + len(absent)
    delta_by_m = {d["modality"]: d.get("delta") for d in drop_one}
    infl_by_m = {r["modality"]: r for r in _d(inflation).get("per_modality", [])}
    suff_by_m = {p["modality"]: p for p in _d(sufficiency).get("per_modality", [])}

    for m in usable:
        mv = _d(_d(modality_view).get(m))
        conf, cov = _f(mv.get("confidence")), _f(mv.get("coverage"))
        wt = _f(_d(weights).get(m)) or 0.0
        infl = _d(infl_by_m.get(m))
        suff = _d(suff_by_m.get(m))
        ratio = _f(infl.get("inflation"))
        rel_factor = _f(infl.get("relative_reliability_factor"))

        # Verdict from the MEASURED ratio, not a threshold on raw weight. The
        # two disagree on real data: the threshold rule flags whichever modality
        # happens to carry >= 0.5, which conflates "large share" with "share it
        # did not earn". Inflation separates them, and its own factorisation
        # says which of the two causes dominates.
        if ratio is None:
            verdict = "unquantified"
            reading = ("Inflation could not be measured (base weights unavailable), so it is "
                       "not possible to say how much of this weight was earned.")
        elif ratio >= 1.5:
            verdict = "scarcity-inflated"
            reading = (f"Weight is {ratio:.2f}x its full-panel ideal. Influence here reflects "
                       f"the absence of other modalities more than this reading's evidence.")
        elif ratio >= 1.15:
            verdict = "moderately-inflated"
            reading = (f"Weight is {ratio:.2f}x its full-panel ideal — partly structural.")
        elif ratio < 0.85:
            verdict = "suppressed"
            reading = (f"Weight is {ratio:.2f}x its full-panel ideal; this reading's own "
                       f"reliability held it below what its discrimination would justify.")
        else:
            verdict = "earned"
            reading = (f"Weight is close to its full-panel ideal ({ratio:.2f}x) — influence "
                       f"reflects measured discrimination, not scarcity.")
        if rel_factor is not None and ratio is not None:
            reading += (" Reliability relative to the surviving panel: "
                        f"{rel_factor:.2f}x " +
                        ("(above average)." if rel_factor > 1.02 else
                         "(below average)." if rel_factor < 0.98 else "(about average)."))

        findings.append({
            "modality": m, "label": MODALITY_LABEL.get(m, m),
            "weight": round(wt, 4), "reading_confidence": conf, "coverage": cov,
            "inflation": ratio,
            "relative_reliability_factor": rel_factor,
            "verdict": verdict, "interpretation": reading,
            "composite_delta_if_removed": delta_by_m.get(m),
            "decision_relevant": suff.get("decision_relevant"),
            "necessary": suff.get("necessary"),
            "alone_sufficient": suff.get("alone_sufficient"),
        })

    # `.get(m, {})` is NOT enough here: if the key exists with an explicit null
    # (which a JSON round-trip can produce), the default never fires and the
    # chained .get() raises. Coerce with `or {}` instead.
    stale = [m for m in usable
             if _d(_d(modality_view).get(m)).get("fresh") is False]
    return {
        "findings": findings,
        "modalities_in_composite": len(usable),
        "modalities_missing": n_absent,
        "stale_but_used": stale,
        "stale_warning": (
            None if not stale else
            f"{len(stale)} {'modality was' if len(stale) == 1 else 'modalities were'} outside "
            f"{'its' if len(stale) == 1 else 'their'} freshness window when this explanation was "
            f"generated ({', '.join(MODALITY_LABEL.get(m, m) for m in stale)}). The composite was "
            f"computed while still fresh; it should not be read as a current-moment assessment."),
        "scarcity_factor": _d(inflation).get("scarcity_factor"),
        "inflation_quantified": _d(inflation).get("quantified", False),
        "decision_irrelevant": _d(sufficiency).get("decision_irrelevant", []),
        "gap_addressed": ("Distinguishes influence-from-evidence from influence-from-scarcity "
                          "via a measured inflation ratio rather than a threshold, addressing "
                          "the conflation TIER-MoE (2026) identifies as unresolved in "
                          "reliability-weighted fusion."),
    }


# ── L5 · epistemic honesty ledger ───────────────────────────────────────────
def _layer_honesty_ledger(harmonisation: Dict[str, Any], conformal: Optional[Dict[str, Any]],
                          modality_view: Dict[str, Any], usable: List[str],
                          reference_status: Optional[Dict[str, Any]],
                          thresholds_authoritative: bool) -> Dict[str, Any]:
    """What this number does NOT know.

    Deliberately not a footnote. Each entry names the limitation, what it
    affects, and what would resolve it, so a reader can judge how far to trust
    the assessment rather than being handed a bare number.
    """
    items: List[Dict[str, str]] = []

    # Reference distributions — the percentile floor under every harmonised value.
    placeholders = []
    if isinstance(reference_status, dict):
        for m, meta in reference_status.items():
            src = _d(meta).get("source", "") if isinstance(meta, dict) else str(meta)
            if "PLACEHOLDER" in str(src).upper():
                placeholders.append(m)
    if placeholders:
        items.append({
            "issue": "Reference distributions are synthetic placeholders",
            "detail": (f"Harmonisation converts each raw score to a population percentile. For "
                       f"{', '.join(placeholders)} that population is synthetic, not real "
                       f"held-out patient scores. The harmonised values, and therefore the "
                       f"composite, are internally consistent but are not yet calibrated "
                       f"against a real cohort."),
            "affects": "every harmonised value and the composite",
            "resolved_by": "replacing each reference set with real held-out component scores",
            "severity": "high",
        })

    # Conformal calibration.
    if isinstance(conformal, dict):
        if conformal.get("conformal_calibrated") is False:
            n = conformal.get("conformal_n", 0)
            items.append({
                "issue": "Tier prediction set is not conformal-calibrated",
                "detail": (f"Only {n} clinician verdicts are on file. The system returns the full "
                           f"tier set rather than a narrower one it cannot certify. The point "
                           f"tier shown is the model's best estimate, not a calibrated claim."),
                "affects": "the reliability of the tier label",
                "resolved_by": "recording clinician verdicts via POST /v1/verdict",
                "severity": "high",
            })

    # Per-modality epistemic gaps, read from what each component reported.
    for m in usable:
        mv = _d(_d(modality_view).get(m))
        h = _d(_d(harmonisation).get(m))
        if _f(mv.get("coverage")) is None:
            items.append({
                "issue": f"{MODALITY_LABEL.get(m, m)}: coverage unknown",
                "detail": ("The component did not report how much of the expected data window "
                           "was actually present, so no coverage penalty was applied to its "
                           "weight. The weight may be optimistic."),
                "affects": f"the weight assigned to {m}",
                "resolved_by": "the component publishing a coverage figure",
                "severity": "medium",
            })
        if h.get("drift"):
            items.append({
                "issue": f"{MODALITY_LABEL.get(m, m)}: distribution drift detected",
                "detail": str(h.get("note") or "The raw score sat outside the reference "
                                                "distribution's expected range."),
                "affects": "the harmonised value for this modality",
                "resolved_by": "refreshing the reference distribution for this component",
                "severity": "medium",
            })
        mver = mv.get("model_version")
        if not mver:
            items.append({
                "issue": f"{MODALITY_LABEL.get(m, m)}: model version not recorded",
                "detail": ("The component did not report which model version produced this "
                           "score, so this reading cannot be tied to a specific model build."),
                "affects": "reproducibility of this reading",
                "resolved_by": "the component returning model_version",
                "severity": "low",
            })

    if not thresholds_authoritative:
        items.append({
            "issue": "Tier cut-points not supplied by the fusion service",
            "detail": ("The explainer fell back to its own default cut-points. Any tier label "
                       "attached to a counterfactual may not match the deployed thresholds."),
            "affects": "tier labels on counterfactuals",
            "resolved_by": "passing tier_thresholds from the fusion service",
            "severity": "medium",
        })

    order = {"high": 0, "medium": 1, "low": 2}
    items.sort(key=lambda i: order.get(i["severity"], 3))
    return {
        "items": items,
        "count": len(items),
        "highest_severity": items[0]["severity"] if items else None,
        "note": ("An explanation that omits what the system does not know is not an "
                 "explanation. Every entry names the limitation, what it affects, and what "
                 "would resolve it."),
    }


# ── narrative ───────────────────────────────────────────────────────────────
def _clinician_narrative(composite: Optional[float], tier: Optional[str], band: Optional[str],
                         attribution: Dict[str, Any], provenance: Dict[str, Any],
                         audit: Dict[str, Any], counterfactuals: Dict[str, Any],
                         ledger: Dict[str, Any], reason: Optional[str],
                         sufficiency: Optional[Dict[str, Any]] = None) -> str:
    """Deterministic template assembly. No language model: the same fusion
    result must always yield the same words."""
    if composite is None:
        return (f"No composite was produced. {reason or 'The evidence gate was not satisfied.'} "
                f"This is a refusal to assess, not an assessment of low risk.")

    parts = [f"Composite {composite:.4f} — tier {tier or 'unassigned'}, band {band or 'unknown'}. "
             f"Based on {audit['modalities_in_composite']} of "
             f"{audit['modalities_in_composite'] + audit['modalities_missing']} modalities."]

    ranked = attribution.get("ranked") or []
    if ranked:
        top = ranked[0]
        seg = f"{top['label'].capitalize()} was the largest driver"
        if top.get("share_pct") is not None:
            seg += f", producing {top['share_pct']}% of the composite"
        parts.append(seg + ".")
        if len(ranked) > 1:
            rest = "; ".join(
                f"{r['label']} {r['share_pct']}%" for r in ranked[1:] if r.get("share_pct") is not None)
            if rest:
                parts.append(f"Remaining contributions: {rest}.")

    for f in audit.get("findings", []):
        if f["verdict"] == "scarcity-inflated":
            parts.append(
                f"Caution — {f['label']} carries {f['inflation']:.2f}x the weight it would hold "
                f"in a full panel. Its influence reflects missing modalities more than evidence.")
        elif f["verdict"] == "suppressed":
            parts.append(
                f"{f['label'].capitalize()} is weighted at {f['inflation']:.2f}x its full-panel "
                f"ideal — its own reliability held it below what its discrimination justifies.")

    # Decision relevance is the actionable question, and it is not the same as
    # attribution size: a modality can produce a large share of the composite
    # while no achievable value of it moves the tier.
    suff = _d(sufficiency)
    irrelevant = suff.get("decision_irrelevant") or []
    if irrelevant:
        names = ", ".join(MODALITY_LABEL.get(m, m) for m in irrelevant)
        parts.append(f"No achievable value of {names} would change this tier — "
                     f"contributing to the score is not the same as determining it.")
    mss = suff.get("minimal_sufficient_sets") or []
    if mss:
        if len(mss[0]["modalities"]) == 1:
            parts.append(f"{mss[0]['labels'][0].capitalize()} alone pins this tier regardless "
                         f"of what the others report.")
        elif len(mss) == 1:
            parts.append(f"Holding the weights as they are, the tier is pinned only by "
                         f"{' and '.join(mss[0]['labels'])} together — no single one of them "
                         f"fixes it while the other is free to vary.")
        else:
            parts.append(f"{len(mss)} different {len(mss[0]['modalities'])}-modality "
                         f"combinations each pin this tier independently.")

    if provenance.get("renormalisation_note"):
        parts.append(provenance["renormalisation_note"])
    if audit.get("stale_warning"):
        parts.append(audit["stale_warning"])

    for d in (counterfactuals.get("drop_one") or [])[:1]:
        if d.get("delta") is not None and d.get("composite_without") is not None:
            direction = "fall" if d["delta"] < 0 else "rise"
            parts.append(
                f"Without {d['label']}, the composite would {direction} to "
                f"{d['composite_without']:.4f} (tier {d['tier_without'] or 'unassigned'}).")

    unreachable = [f for f in (counterfactuals.get("tier_flip_points") or [])
                   if not f["reachable"]]
    if unreachable and len(unreachable) == len(counterfactuals.get("tier_flip_points") or []):
        parts.append("No single modality can change this tier at any achievable value.")

    if ledger.get("count"):
        hi = [i for i in ledger["items"] if i["severity"] == "high"]
        if hi:
            parts.append("Limitations affecting this number: " +
                         "; ".join(i["issue"] for i in hi) + ".")
        else:
            parts.append(f"{ledger['count']} recorded limitation(s) apply — see the ledger.")
    return " ".join(parts)


def _headline(composite: Optional[float], tier: Optional[str],
              attribution: Dict[str, Any], audit: Dict[str, Any],
              ledger: Dict[str, Any], reason: Optional[str]) -> str:
    """Two sentences, for inline embedding in doctor_timeline.

    The full narrative is the right depth for a dedicated explanation panel but
    too heavy to repeat on every timeline poll, where it would bury the clinical
    data it is meant to annotate. This carries the headline plus the single most
    important caveat; the full prose stays on the explanation endpoint.
    """
    if composite is None:
        return f"No composite produced — {reason or 'evidence gate not satisfied'}. Not a low-risk result."
    ranked = attribution.get("ranked") or []
    lead = ranked[0] if ranked else None
    s = f"{tier or 'Unassigned'} ({composite:.2f}) from {audit.get('modalities_in_composite', 0)} of " \
        f"{audit.get('modalities_in_composite', 0) + audit.get('modalities_missing', 0)} modalities"
    if lead and lead.get("share_pct") is not None:
        s += f"; {lead['label']} drove {lead['share_pct']}%."
    else:
        s += "."
    inflated = [(f["label"], f.get("inflation")) for f in audit.get("findings", [])
                if f["verdict"] == "scarcity-inflated"]
    irrelevant = audit.get("decision_irrelevant") or []
    if inflated:
        lbl, ratio = inflated[0]
        s += f" Caution: {lbl} weight is {ratio:.2f}x full-panel ideal."
    elif irrelevant:
        s += f" Note: {len(irrelevant)} modality/ies cannot change this tier."
    elif audit.get("stale_but_used"):
        s += " Caution: includes data outside its freshness window."
    elif ledger.get("highest_severity") == "high":
        s += f" {ledger.get('count', 0)} limitation(s) apply."
    return s


def _patient_narrative(band: Optional[str], composite: Optional[float]) -> str:
    """Patient-safe. Mirrors the /v1/patients/{id}/risk contract: band only, no
    per-modality scores, no note content, no weights. A patient reading 'your
    clinical notes score 0.66' without a clinician present is a harm."""
    if composite is None:
        return ("There is not enough recent information to complete an assessment yet. "
                "This is not a result — it means more information is needed.")
    return {
        "RED": ("Your recent information suggests you may be going through a difficult period. "
                "Your care team has been notified and will follow up with you."),
        "AMBER": ("Your recent information shows some signs worth keeping an eye on. "
                  "Your care team can talk this through with you."),
        "GREEN": ("Your recent information does not show signs of concern at the moment."),
    }.get(band or "", "An assessment is available. Your care team can talk it through with you.")


# ── public entry point ──────────────────────────────────────────────────────
def explain_fusion(fusion: Dict[str, Any],
                   modality_view: Optional[Dict[str, Any]] = None,
                   tier_thresholds: Optional[Dict[str, float]] = None,
                   reference_status: Optional[Dict[str, Any]] = None,
                   base_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Build the CARE-X explanation for one fusion result.

    fusion          the stored FusionResult as a dict — composite, tier, band,
                    weights, contributions, harmonisation, reason, renormalised
    modality_view   the per-modality block doctor_timeline already assembles
                    (status, confidence, coverage, fresh, model_version)
    tier_thresholds cut-points from the fusion service. Strongly preferred; the
                    module falls back to defaults and FLAGS that it did so.
    reference_status per-modality reference metadata, used to detect placeholders

    Never raises on partial input: a missing field produces a stated gap, not a
    fabricated value.
    """
    modality_view = _d(modality_view)
    fusion = _d(fusion)
    weights = _d(fusion.get("weights"))
    contributions = _d(fusion.get("contributions"))
    _harm_all = _d(fusion.get("harmonisation"))
    harmonisation = {k: v for k, v in _harm_all.items()
                     if k not in ("gate", "conformal")}
    conformal = _harm_all.get("conformal") or fusion.get("conformal")
    composite = _f(fusion.get("composite"))
    tier, band = fusion.get("tier"), fusion.get("band")

    thresholds, authoritative = _resolve_thresholds(tier_thresholds)
    usable = _usable(weights, harmonisation)

    excluded = [m for m, v in modality_view.items()
                if isinstance(v, dict) and v.get("excluded") is True]
    absent = [m for m in MODALITY_LABEL
              if m not in usable and m not in excluded]

    if composite is None or not usable:
        ledger = _layer_honesty_ledger(harmonisation, conformal, modality_view, usable,
                                       reference_status, authoritative)
        return {
            "explainer_version": EXPLAINER_VERSION,
            "assessed": False,
            "composite": None, "tier": None, "band": band or "GREY",
            "refusal_reason": fusion.get("reason") or "insufficient usable evidence",
            "attribution": None, "weight_provenance": None,
            "counterfactuals": None, "reliability_audit": None,
            "weight_inflation": None, "sufficiency": None,
            "honesty_ledger": ledger,
            "narrative": _clinician_narrative(None, None, band, {}, {},
                                              {"modalities_in_composite": 0,
                                               "modalities_missing": len(MODALITY_LABEL)},
                                              {}, ledger, fusion.get("reason"), None),
            "headline": _headline(None, None, {}, {}, ledger, fusion.get("reason")),
            "patient_narrative": _patient_narrative(band, None),
        }

    attribution = _layer_attribution(contributions, composite, usable)
    provenance = _layer_weight_provenance(weights, usable, modality_view,
                                          fusion.get("renormalised"), excluded, absent)
    counterfactuals = _layer_counterfactuals(weights, harmonisation, composite, usable,
                                             thresholds, authoritative)
    inflation = _layer_inflation(weights, usable, modality_view, base_weights, excluded)
    sufficiency = _layer_sufficiency(weights, harmonisation, composite, usable, thresholds)
    audit = _layer_reliability_audit(weights, usable, modality_view,
                                     counterfactuals["drop_one"], excluded, absent,
                                     inflation, sufficiency)
    ledger = _layer_honesty_ledger(harmonisation, conformal, modality_view, usable,
                                   reference_status, authoritative)

    return {
        "explainer_version": EXPLAINER_VERSION,
        "assessed": True,
        "composite": round(composite, 4),
        "tier": tier,
        "band": band,
        "attribution": attribution,
        "weight_provenance": provenance,
        "counterfactuals": counterfactuals,
        "weight_inflation": inflation,
        "sufficiency": sufficiency,
        "reliability_audit": audit,
        "honesty_ledger": ledger,
        "narrative": _clinician_narrative(composite, tier, band, attribution, provenance,
                                          audit, counterfactuals, ledger, fusion.get("reason"),
                                          sufficiency),
        "headline": _headline(composite, tier, attribution, audit, ledger, fusion.get("reason")),
        "patient_narrative": _patient_narrative(band, composite),
    }


def load_reference_status(reference_dir: Any = None) -> Dict[str, Any]:
    """Read each reference distribution's metadata so the ledger can tell a real
    cohort from a synthetic placeholder.

    Best effort by design: if the directory is missing or unreadable the ledger
    simply omits the placeholder warning rather than the endpoint failing. It
    returns {} and never raises, because an explanation that cannot be produced
    is worse than one missing a single caveat.
    """
    import json
    import os
    from pathlib import Path

    if reference_dir is None:
        reference_dir = os.getenv(
            "FUSION_SERVICE_DIR",
            str(Path(__file__).resolve().parent.parent / "fusion_service"))
        reference_dir = Path(reference_dir) / "reference"
    reference_dir = Path(reference_dir)

    out: Dict[str, Any] = {}
    try:
        for path in sorted(reference_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    blob = json.load(fh)
                out[path.stem] = {
                    "source": (_d(blob).get("source")
                               or _d(_d(blob).get("meta")).get("source") or ""),
                    "n": _d(blob).get("n") or len(_d(blob).get("scores") or []),
                }
            except Exception:                               # noqa: BLE001
                continue
    except Exception:                                       # noqa: BLE001
        return {}
    return out


def load_tier_thresholds() -> Optional[Dict[str, float]]:
    """Pull the deployed tier cut-points from the fusion service if importable.

    Returning None (rather than a guess) is deliberate: explain_fusion then
    falls back to defaults AND records in the ledger that it did so, so a
    mismatch surfaces instead of silently mislabelling counterfactual tiers.
    """
    thresholds, authoritative = _resolve_thresholds(None)
    return thresholds if authoritative else None


def explanation_summary(explanation: Dict[str, Any]) -> Dict[str, Any]:
    """Compact form for embedding inline in doctor_timeline, where the full
    object would bury the clinical data it is meant to annotate."""
    if not explanation.get("assessed"):
        return {"explainer_version": explanation.get("explainer_version"),
                "assessed": False,
                "headline": explanation.get("headline"),
                "refusal_reason": explanation.get("refusal_reason"),
                "full_explanation_at": "/v1/doctor/patients/{subject_id}/explanation"}
    attr = _d(explanation.get("attribution"))
    audit = _d(explanation.get("reliability_audit"))
    ledger = _d(explanation.get("honesty_ledger"))
    return {
        "explainer_version": explanation.get("explainer_version"),
        "assessed": True,
        "headline": explanation.get("headline"),
        "full_explanation_at": "/v1/doctor/patients/{subject_id}/explanation",
        "dominant_modality": attr.get("dominant"),
        "top_contributors": [
            {"modality": r["modality"], "share_pct": r["share_pct"]}
            for r in (attr.get("ranked") or [])[:2]],
        "scarcity_inflated": [f["modality"] for f in audit.get("findings", [])
                              if f["verdict"] == "scarcity-inflated"],
        "inflation": {f["modality"]: f.get("inflation") for f in audit.get("findings", [])
                      if f.get("inflation") is not None},
        "decision_irrelevant": audit.get("decision_irrelevant", []),
        "minimal_sufficient_size": _d(explanation.get("sufficiency")).get("minimal_sufficient_size"),
        "stale_but_used": audit.get("stale_but_used", []),
        "limitations_count": ledger.get("count", 0),
        "highest_severity": ledger.get("highest_severity"),
    }
