from __future__ import annotations

import datetime as dt
import math
import uuid
from dataclasses import dataclass

from sqlalchemy import select

from notifications import queue_event_delivery

from db_models import (AttentionEvent, AuditLog, EscalationEpisode, ForecastResult,
                       FusionResult, ModalityReading, Subject, utcnow)


@dataclass(frozen=True)
class EscalationPolicy:
    version: str = "escalation-v1"
    forecast_threshold: float = .70
    minimum_increase: float = .10
    confirmation_min_seconds: int = 20
    confirmation_max_seconds: int = 120
    recovery_threshold: float = .40
    cooldown_minutes: int = 10


POLICY = EscalationPolicy()


def _normalise(value):
    if value is None: return None
    value = float(value)
    return value / 100.0 if value > 1.0 else value


def _tier(score):
    if score is None: return None
    return "High" if score >= .70 else "Medium" if score >= .45 else "Low"


def _source_window(reading: ModalityReading):
    value = (((reading.detail or {}).get("response") or {}).get("latest_reading_at"))
    if not isinstance(value, str) or not value:
        return None
    try:
        timestamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, OverflowError):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    captured = reading.captured_at
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=dt.timezone.utc)
    # Never treat the client's receipt-time fallback for a malformed C1 source
    # timestamp as independent evidence of a new physiological observation.
    return timestamp if abs((timestamp - captured).total_seconds()) <= 1 else None



def _ten_minute_forecast(response: dict):
    """Extract the 10-minute endpoint from either published C1 wire shape.

    A two-point [5, 10]-minute payload MUST declare its horizons; otherwise
    interpreting its second point as 10 minutes is an unsupported assumption.
    """
    values = response.get("risk_forecast")
    if not isinstance(values, list) or not values:
        return None
    horizons = response.get("forecast_horizons_minutes")
    if horizons is not None:
        if not isinstance(horizons, list) or len(horizons) != len(values):
            return None
        if len(horizons) != len(set(map(str, horizons))):
            return None
        if len(values) == 2 and horizons == [5, 10]:
            target = values[1]
        elif len(values) >= 10 and horizons == list(range(1, len(values) + 1)):
            target = values[9]
        else:
            return None
    elif len(values) >= 10:
        target = values[9]
    else:
        return None
    # Reject malformed steps anywhere in the curve; do not cherry-pick a
    # plausible endpoint from an invalid source prediction.
    for point in values:
        if isinstance(point, bool):
            return None
        try:
            normalised = _normalise(point)
        except (TypeError, ValueError, OverflowError):
            return None
        if normalised is None or not math.isfinite(normalised) or not 0 <= normalised <= 1:
            return None
    try:
        score = _normalise(target)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if score is not None and math.isfinite(score) and 0 <= score <= 1 else None


def needs_confirmation_fusion(db, subject_id: str, reading: ModalityReading, *, as_of=None) -> bool:
    """Whether an independent second C1 window may trigger a new episode.

    The ingest caller may bypass its routine fusion debounce only for this
    candidate. The actual episode policy and deduplication remain authoritative
    in persist_c1_forecast_and_event.
    """
    now = as_of or utcnow()
    if reading.status != "ok":
        return False
    captured = reading.captured_at
    if captured.tzinfo is None:
        captured = captured.replace(tzinfo=dt.timezone.utc)
    if not 0 <= (now - captured).total_seconds() <= 120:
        return False
    score = _ten_minute_forecast(((reading.detail or {}).get("response") or {}))
    current = _normalise(reading.raw_score) if reading.raw_score is not None else None
    if (score is None or current is None or not math.isfinite(current)
            or score < POLICY.forecast_threshold
            or score - current < POLICY.minimum_increase):
        return False
    source_time = _source_window(reading)
    if source_time is None or source_time > now:
        return False
    prior = db.scalar(select(ForecastResult).where(
        ForecastResult.subject_id == subject_id,
        ForecastResult.escalation_predicted.is_(True),
        ForecastResult.generated_at >= now - dt.timedelta(seconds=POLICY.confirmation_max_seconds),
        ForecastResult.generated_at <= now - dt.timedelta(seconds=POLICY.confirmation_min_seconds))
        .order_by(ForecastResult.generated_at.desc()).limit(1))
    if prior is None:
        return False
    previous_reading = db.get(ModalityReading, prior.source_reading_id) if prior.source_reading_id else None
    prior_time = _source_window(previous_reading) if previous_reading else None
    if prior_time is None or prior_time > now:
        return False
    delta = (source_time - prior_time).total_seconds()
    return POLICY.confirmation_min_seconds <= delta <= POLICY.confirmation_max_seconds


def persist_c1_forecast_and_event(db, subject_id: str, reading: ModalityReading):
    # PostgreSQL serializes policy evaluation for one subject; the partial
    # unique index remains the final idempotency guard on every database.
    # The reading was already inserted with a FK KEY SHARE lock on Subject.
    # FOR UPDATE would conflict with that lock and two concurrent ingests
    # could deadlock while each tried to upgrade it. NO KEY UPDATE still
    # serializes policy writers and is compatible with the FK lock.
    db.scalar(select(Subject).where(Subject.subject_id == subject_id)
              .with_for_update(key_share=True))
    if reading.status != "ok": return None, None
    captured = reading.captured_at
    if captured.tzinfo is None: captured = captured.replace(tzinfo=dt.timezone.utc)
    now = utcnow()
    age_seconds = (now - captured).total_seconds()
    if not 0 <= age_seconds <= 120: return None, None
    try:
        current = _normalise(reading.raw_score)
    except (TypeError, ValueError, OverflowError):
        return None, None
    if current is None or not math.isfinite(current) or not 0 <= current <= 1:
        return None, None
    recovered = False
    # Recovery is based on the fresh, valid observation, even if C1's forecast
    # array is unavailable or invalid; otherwise an episode can remain stuck.
    if current < POLICY.recovery_threshold:
        active_episode = db.scalar(select(EscalationEpisode).where(
            EscalationEpisode.subject_id == subject_id,
            EscalationEpisode.status == "active").limit(1))
        if active_episode:
            active_episode.status = "closed"
            active_episode.closed_at = now
            db.flush()
            recovered = True
    def unavailable():
        if recovered:
            db.commit()
        return None, None
    response = ((reading.detail or {}).get("response") or {})
    score = _ten_minute_forecast(response)
    if score is None:
        return unavailable()
    predicted = score >= POLICY.forecast_threshold and score-current >= POLICY.minimum_increase
    fusion = db.scalar(select(FusionResult).where(FusionResult.subject_id == subject_id)
        .order_by(FusionResult.computed_at.desc(), FusionResult.id.desc()).limit(1))
    snapshot_ids = ((fusion.harmonisation or {}).get("source_reading_ids")
                    if fusion else None)
    source_fusion_id = (fusion.id if fusion and
                        (snapshot_ids is None or snapshot_ids.get("c1_physiological") == reading.id)
                        else None)
    forecast = ForecastResult(forecast_result_id=f"fcst_{uuid.uuid4().hex}", subject_id=subject_id,
        source_reading_id=reading.id, source_fusion_result_id=source_fusion_id,
        scope="physiological", horizon_minutes=10, score=score,
        tier=_tier(score), escalation_probability=None, escalation_predicted=predicted,
        generated_at=now, valid_until=now + dt.timedelta(minutes=10), model_version=reading.model_version)
    db.add(forecast); db.flush()
    event = None
    if predicted:
        previous = db.scalar(select(ForecastResult).where(ForecastResult.subject_id == subject_id,
            ForecastResult.forecast_result_id != forecast.forecast_result_id,
            ForecastResult.escalation_predicted.is_(True),
            ForecastResult.generated_at >= now-dt.timedelta(seconds=POLICY.confirmation_max_seconds),
            ForecastResult.generated_at <= now-dt.timedelta(seconds=POLICY.confirmation_min_seconds))
            .order_by(ForecastResult.generated_at.desc()).limit(1))
        # Polling the same prediction after 20 seconds is not independent
        # confirmation. Only two source-stamped C1 windows count; when C1 does
        # not publish a source timestamp, expose forecast but do not alert.
        previous_reading = db.get(ModalityReading, previous.source_reading_id) if previous else None
        current_window = _source_window(reading)
        previous_window = _source_window(previous_reading) if previous_reading else None
        source_delta = ((current_window - previous_window).total_seconds()
                        if current_window is not None and previous_window is not None else None)
        distinct_windows = (source_delta is not None and
                            POLICY.confirmation_min_seconds <= source_delta <= POLICY.confirmation_max_seconds
                            and current_window <= now and previous_window <= now)
        active = db.scalar(select(EscalationEpisode).where(EscalationEpisode.subject_id == subject_id,
            EscalationEpisode.status == "active").limit(1))
        last_closed = db.scalar(select(EscalationEpisode).where(EscalationEpisode.subject_id == subject_id,
            EscalationEpisode.status == "closed").order_by(EscalationEpisode.closed_at.desc()).limit(1))
        cooling_until = None
        if last_closed and last_closed.closed_at:
            closed_at = last_closed.closed_at
            if closed_at.tzinfo is None: closed_at = closed_at.replace(tzinfo=dt.timezone.utc)
            cooling_until = closed_at + dt.timedelta(minutes=POLICY.cooldown_minutes)
        rearmed = cooling_until is None or now >= cooling_until
        previous_is_new = previous is not None and (cooling_until is None or
            previous.generated_at.replace(tzinfo=dt.timezone.utc) >= cooling_until)
        if (previous_is_new and distinct_windows and active is None and rearmed
                and source_fusion_id is not None and fusion.composite is not None):
            episode = EscalationEpisode(episode_id=f"ep_{uuid.uuid4().hex}", subject_id=subject_id)
            db.add(episode); db.flush()
            event = AttentionEvent(id=f"evt_{uuid.uuid4().hex}", subject_id=subject_id,
                fusion_result_id=fusion.id,
                forecast_result_id=forecast.forecast_result_id, episode_id=episode.episode_id,
                severity="high", reason="Forecast crossed versioned escalation policy",
                forecast_horizon=10, policy_version=POLICY.version)
            db.add(event)
            db.flush()
            queue_event_delivery(db, event)
            db.add(AuditLog(subject_id=subject_id, event="attention.created", actor=f"policy:{POLICY.version}",
                            detail={"event_id": event.id, "fusion_result_id": fusion.id,
                                    "forecast_result_id": forecast.forecast_result_id}))
    db.commit()
    return forecast, event
