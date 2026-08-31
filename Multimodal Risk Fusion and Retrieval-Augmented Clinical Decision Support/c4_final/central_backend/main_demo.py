"""Opt-in wrapper around the production central backend for investor demos.

Normal behavior is unchanged unless INVESTOR_DEMO_SIMULATION=1. Keeping the
simulation in this wrapper makes the temporary path easy to remove after the
demo and avoids edits to the research implementation in main.py.
"""

from __future__ import annotations

import os

import main as core
from db_models import FusionResult
from demo_simulation import clinical_result, physiological_composite, tier_and_band

app = core.app

_real_call_c3 = core.mc.call_c3
_real_auto_fuse = core._auto_fuse
_real_run_fusion = core.run_fusion
_real_select_support_set = core.select_support_set


def _demo_enabled() -> bool:
    return os.getenv("INVESTOR_DEMO_SIMULATION", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }


def _demo_call_c3(note_text, note_type="progress", anxiety_support=None,
                  control_support=None, support_set=None,
                  support_set_version=None, note_date=None, visit_count=None,
                  subject_external_id=None, client=None):
    if not _demo_enabled():
        return _real_call_c3(
            note_text, note_type, anxiety_support, control_support,
            support_set=support_set,
            support_set_version=support_set_version,
            note_date=note_date,
            visit_count=visit_count,
            subject_external_id=subject_external_id,
            client=client,
        )
    return clinical_result(note_text)


def _demo_select_support_set(*args, **kwargs):
    if not _demo_enabled():
        return _real_select_support_set(*args, **kwargs)
    # The demo C3 path does not use prototypes, but main.py selects a support set
    # before calling C3. Return a harmless placeholder so a broken/empty support
    # bank cannot block the deterministic demo simulation.
    return [{
        "id": "investor-demo-support",
        "text": "temporary investor demo support placeholder",
        "label": "control",
        "note_date": "2026-01-01",
    }]


def _demo_run_fusion(db, subject_id: str, trigger: str = "manual"):
    if not _demo_enabled():
        return _real_run_fusion(db, subject_id, trigger)

    core._require_subject(db, subject_id)
    readings = core._latest_readings(db, subject_id)
    c1 = readings.get("c1_physiological")
    score = physiological_composite(
        c1.get("raw_score") if c1 and c1.get("status") == "ok" else None
    )

    if score is None:
        row = FusionResult(
            subject_id=subject_id,
            composite=None,
            tier=None,
            band="GREY",
            confidence=0.0,
            modalities_used=0,
            renormalised=True,
            weights={},
            contributions={},
            harmonisation={
                "assessment": {
                    "status": "insufficient",
                    "missing_modalities": ["c1_physiological"],
                },
                "simulation": {"enabled": True, "source": "c1_physiological"},
            },
            reason="investor demo simulation: no usable physiological score yet",
            trigger=trigger,
            model_version="investor-demo-physio-mirror-v1",
        )
    else:
        tier, band = tier_and_band(score)
        c1_confidence = float(c1.get("confidence", 0.5) or 0.5)
        c1_coverage = float(c1.get("coverage", 1.0) or 1.0)
        missing = [
            m for m in core.FUSION_REQUIRED_MODALITIES
            if m not in readings or readings[m].get("status") != "ok"
        ]
        assessment_status = "complete" if not missing else "provisional"
        weights = {m: 0.0 for m in core.ALL_MODALITIES}
        contributions = {m: 0.0 for m in core.ALL_MODALITIES}
        weights["c1_physiological"] = 1.0
        contributions["c1_physiological"] = score
        row = FusionResult(
            subject_id=subject_id,
            composite=score,
            tier=tier,
            band=band,
            confidence=round(c1_confidence * c1_coverage, 4),
            modalities_used=1,
            renormalised=True,
            weights=weights,
            contributions=contributions,
            harmonisation={
                "assessment": {
                    "status": assessment_status,
                    "missing_modalities": missing,
                },
                "simulation": {
                    "enabled": True,
                    "source": "c1_physiological.current_risk_index",
                    "raw_score_0_100": c1.get("raw_score"),
                    "composite_0_1": score,
                },
            },
            reason=(
                "investor demo simulation: fusion composite mirrors the latest "
                "physiological current_risk_index"
            ),
            trigger=trigger,
            model_version="investor-demo-physio-mirror-v1",
        )

    db.add(row)
    core._audit(
        db,
        subject_id,
        "fusion.demo_simulation",
        {"composite": row.composite, "trigger": trigger},
    )
    db.commit()
    return row


def _demo_auto_fuse(db, subject_id: str, trigger: str, debounce: bool = False):
    if not _demo_enabled():
        return _real_auto_fuse(db, subject_id, trigger, debounce=debounce)
    # Investor demo requirement: physiological updates are reflected immediately,
    # so the production five-minute debounce is intentionally bypassed here.
    try:
        row = _demo_run_fusion(db, subject_id, trigger)
        assessment = core._assessment_for_row(row)
        return {
            "fusion_triggered": True,
            "fusion": {
                "fusion_result_id": row.id,
                "composite": row.composite,
                "tier": row.tier,
                "band": row.band,
                "reason": row.reason,
                "assessment_status": assessment["status"],
                "missing_modalities": assessment["missing_modalities"],
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "fusion_triggered": False,
            "fusion_error": f"{type(exc).__name__}: {exc}"[:160],
        }


# main.py's route functions resolve these globals at request time, so swapping
# only these seams changes no route, request model, response model, or mobile UI.
core.mc.call_c3 = _demo_call_c3
core.select_support_set = _demo_select_support_set
core.run_fusion = _demo_run_fusion
core._auto_fuse = _demo_auto_fuse
