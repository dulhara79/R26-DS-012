"""
apply_carex_patch.py — wires CARE-X into main.py.

Same defensive contract as apply_support_bank_patch.py: it verifies each anchor
before writing, prints a WARNING and skips anything it cannot match exactly, and
never guesses. Run it from central_backend/.

Idempotent: running it twice is a no-op.
"""

import re
import sys

MAIN = "main.py"


def patch_import():
    content = open(MAIN, encoding="utf-8").read()
    if "import explain" in content:
        print("[main.py] explain already imported — skipping")
        return content, False

    block = ("\n# CARE-X — explanation layer for the fused composite.\n"
             "import explain as carex\n"
             "_CAREX_THRESHOLDS = carex.load_tier_thresholds()\n"
             "_CAREX_REFERENCE_STATUS = carex.load_reference_status()\n"
             "_CAREX_BASE_WEIGHTS = carex.load_base_weights()\n")

    m = re.search(r"^from support_bank import[^\n]*(\n\s+[^\n]*)*", content, re.M)
    if m:
        content = content[:m.end()] + "\n" + block + content[m.end():]
        print("[main.py] Added explain import after support_bank import")
        return content, True

    m = re.search(r"^import modality_clients as mc\s*$", content, re.M)
    if m:
        content = content[:m.end()] + "\n" + block + content[m.end():]
        print("[main.py] Added explain import after modality_clients import")
        return content, True

    idx = content.find("app = FastAPI(")
    if idx != -1:
        content = content[:idx] + block + "\n\n" + content[idx:]
        print("[main.py] Added explain import before app = FastAPI(")
        return content, True

    print("[main.py] WARNING: no anchor found for the import. Add manually near the top:")
    print(block)
    return content, False


def patch_inline_field(content):
    """Add the compact explanation into doctor_timeline's response."""
    if "carex.explanation_summary" in content:
        print("[main.py] inline explanation already present — skipping")
        return content, False

    old = ('        "trend": [{"composite": h.composite, "tier": h.tier, "band": h.band,\n'
           '                   "computed_at": h.computed_at, "trigger": h.trigger}\n'
           '                  for h in reversed(history)],\n'
           '    }')
    new = ('        "trend": [{"composite": h.composite, "tier": h.tier, "band": h.band,\n'
           '                   "computed_at": h.computed_at, "trigger": h.trigger}\n'
           '                  for h in reversed(history)],\n'
           '        # CARE-X: compact form only. The full explanation (weight provenance,\n'
           '        # exact counterfactuals, honesty ledger) is on the /explanation endpoint\n'
           '        # so it is not re-serialised on every timeline poll.\n'
           '        "explanation": carex.explanation_summary(carex.explain_fusion(\n'
           '            _carex_fusion_dict(latest), modality_view,\n'
           '            _CAREX_THRESHOLDS, _CAREX_REFERENCE_STATUS,\n            _CAREX_BASE_WEIGHTS)),\n'
           '    }')
    if old not in content:
        print("[main.py] WARNING: doctor_timeline return block not matched.")
        print('           Add this key by hand to the return dict of doctor_timeline:')
        print('             "explanation": carex.explanation_summary(carex.explain_fusion(')
        print('                 _carex_fusion_dict(latest), modality_view,')
        print('                 _CAREX_THRESHOLDS, _CAREX_REFERENCE_STATUS,\n            _CAREX_BASE_WEIGHTS)),')
        return content, False
    print("[main.py] Added inline explanation to doctor_timeline")
    return content.replace(old, new, 1), True


def patch_helper_and_endpoint(content):
    """Add the row->dict adapter and the dedicated explanation endpoint."""
    if "_carex_fusion_dict" in content and "/explanation" in content:
        print("[main.py] helper + endpoint already present — skipping")
        return content, False

    code = '''

# ═════════════════════════════════════════════════════════════════════════════
# CARE-X — explanation of the fused composite
# ═════════════════════════════════════════════════════════════════════════════
def _carex_fusion_dict(row) -> dict:
    """FusionResult row -> the plain dict CARE-X consumes.

    Kept as an adapter rather than passing the ORM object so the explainer has
    no SQLAlchemy dependency and stays unit-testable against fixtures.
    """
    if row is None:
        return {}
    return {
        "composite": row.composite,
        "tier": row.tier,
        "band": row.band,
        "confidence": row.confidence,
        "reason": row.reason,
        "renormalised": row.renormalised,
        "weights": row.weights or {},
        "contributions": row.contributions or {},
        "harmonisation": row.harmonisation or {},
    }


@app.get("/v1/doctor/patients/{subject_id}/explanation", tags=["egress"])
def doctor_explanation(subject_id: str, db: Session = Depends(get_session),
                       authorization: Optional[str] = Header(None)):
    """Why this composite came out the way it did.

    Reads only the stored fusion result — no component is re-called — so the
    explanation is reproducible months later, when those Spaces may be gone.
    Deterministic: the same fusion result always yields the same explanation, or
    a clinician reopening yesterday's assessment would see a different rationale
    and stop trusting the number.
    """
    _auth(authorization)
    _require_subject(db, subject_id)

    latest = _latest_fusion(db, subject_id)
    if latest is None:
        return {"subject_id": subject_id, "assessed": False,
                "narrative": "No assessment has been produced for this patient yet. "
                             "This is an absence of evidence, not a low-risk result.",
                "explainer_version": carex.EXPLAINER_VERSION}

    readings = _latest_readings(db, subject_id)
    now = dt.datetime.now(dt.timezone.utc)
    modality_view = {}
    for modality in ALL_MODALITIES:
        r = readings.get(modality)
        if not r:
            modality_view[modality] = {"status": "absent", "score": None}
            continue
        captured = r["captured_at"]
        if captured.tzinfo is None:
            captured = captured.replace(tzinfo=dt.timezone.utc)
        age_min = (now - captured).total_seconds() / 60.0
        max_age = gate.MAX_AGE_MINUTES.get(modality)
        modality_view[modality] = {
            "status": r["status"], "score": r["raw_score"],
            "confidence": r["confidence"], "coverage": r["coverage"],
            "age_minutes": round(age_min, 1),
            "fresh": (max_age is None) or (age_min <= max_age),
            "model_version": r["model_version"],
            "excluded": modality in gate.EXCLUDED_MODALITIES,
        }

    explanation = carex.explain_fusion(
        _carex_fusion_dict(latest), modality_view,
        _CAREX_THRESHOLDS, _CAREX_REFERENCE_STATUS, _CAREX_BASE_WEIGHTS)

    _audit(db, subject_id, "egress.explanation",
           {"fusion_result_id": latest.id, "explainer": carex.EXPLAINER_VERSION})
    db.commit()

    explanation["subject_id"] = subject_id
    explanation["fusion_result_id"] = latest.id
    explanation["computed_at"] = latest.computed_at
    return explanation
'''
    m = re.search(r'^@app\.get\("/health"', content, re.M)
    if m:
        content = content[:m.start()] + code.strip("\n") + "\n\n\n" + content[m.start():]
        print("[main.py] Added _carex_fusion_dict + /explanation endpoint before /health")
        return content, True

    m = re.search(r'^if __name__ == "__main__":', content, re.M)
    if m:
        content = content[:m.start()] + code.strip("\n") + "\n\n\n" + content[m.start():]
        print("[main.py] Added _carex_fusion_dict + /explanation endpoint before __main__")
        return content, True

    print("[main.py] WARNING: no anchor for the endpoint. Append this block manually:")
    print(code)
    return content, False


if __name__ == "__main__":
    try:
        open(MAIN, encoding="utf-8")
    except FileNotFoundError:
        print(f"ERROR: {MAIN} not found. Run this from central_backend/.")
        sys.exit(1)

    content, a = patch_import()
    content, b = patch_helper_and_endpoint(content)
    content, c = patch_inline_field(content)

    if a or b or c:
        import ast
        try:
            ast.parse(content)
        except SyntaxError as exc:
            print(f"\nABORTED — the patched file would not parse: {exc}")
            print("No changes written. main.py is untouched.")
            sys.exit(1)
        open(MAIN, "w", encoding="utf-8").write(content)
        print(f"\n{MAIN} written and syntax-verified.")
    else:
        print("\nNothing to do — already patched.")
    print("Fix any WARNING lines above before starting the server.")
