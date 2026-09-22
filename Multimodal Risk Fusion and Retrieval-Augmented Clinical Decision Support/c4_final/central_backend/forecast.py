from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import select

from db_models import (AttentionEvent, EscalationEpisode, ForecastResult,
                       FusionResult, ModalityReading, Subject, utcnow)


def _normalise(value):
    if value is None: return None
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def _tier(score):
    if score is None: return None
    return "High" if score >= .70 else "Medium" if score >= .45 else "Low"


def persist_c1_forecast_and_event(db, subject_id: str, reading: ModalityReading):
    # PostgreSQL serializes policy evaluation for one subject; the partial
    # unique index remains the final idempotency guard on every database.
    db.scalar(select(Subject).where(Subject.subject_id == subject_id).with_for_update())
    if reading.status != "ok": return None, None
    captured = reading.captured_at
    if captured.tzinfo is None: captured = captured.replace(tzinfo=dt.timezone.utc)
    now = utcnow()
    if (now - captured).total_seconds() > 120: return None, None
    response = ((reading.detail or {}).get("response") or {})
    values = response.get("risk_forecast") or []
    score = _normalise(values[9] if len(values) >= 10 else values[-1] if values else reading.raw_score)
    current = _normalise(reading.raw_score)
    predicted = bool(score is not None and current is not None and score >= .70 and score-current >= .10)
    forecast = ForecastResult(forecast_result_id=f"fcst_{uuid.uuid4().hex}", subject_id=subject_id,
        source_reading_id=reading.id, scope="physiological", horizon_minutes=10, score=score,
        tier=_tier(score), escalation_probability=None, escalation_predicted=predicted,
        generated_at=now, valid_until=now + dt.timedelta(minutes=10), model_version=reading.model_version)
    db.add(forecast); db.flush()
    event = None
    if predicted:
        previous = db.scalar(select(ForecastResult).where(ForecastResult.subject_id == subject_id,
            ForecastResult.forecast_result_id != forecast.forecast_result_id,
            ForecastResult.escalation_predicted.is_(True),
            ForecastResult.generated_at >= now-dt.timedelta(minutes=2),
            ForecastResult.generated_at <= now-dt.timedelta(seconds=20))
            .order_by(ForecastResult.generated_at.desc()).limit(1))
        active = db.scalar(select(EscalationEpisode).where(EscalationEpisode.subject_id == subject_id,
            EscalationEpisode.status == "active").limit(1))
        if previous and active is None:
            episode = EscalationEpisode(episode_id=f"ep_{uuid.uuid4().hex}", subject_id=subject_id)
            db.add(episode); db.flush()
            fusion = db.scalar(select(FusionResult).where(FusionResult.subject_id == subject_id)
                .order_by(FusionResult.computed_at.desc(), FusionResult.id.desc()).limit(1))
            event = AttentionEvent(id=f"evt_{uuid.uuid4().hex}", subject_id=subject_id,
                fusion_result_id=fusion.id if fusion else None,
                forecast_result_id=forecast.forecast_result_id, episode_id=episode.episode_id,
                severity="high", reason="Forecast crossed versioned escalation policy",
                forecast_horizon=10)
            db.add(event)
    elif current is not None and current < .40:
        active = db.scalar(select(EscalationEpisode).where(EscalationEpisode.subject_id == subject_id,
            EscalationEpisode.status == "active").limit(1))
        if active: active.status = "closed"; active.closed_at = now
    db.commit()
    return forecast, event
