
from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client, create_client

from score_service import load_artifacts, score_participant_events

load_dotenv()

LOCAL_TZ = ZoneInfo("Asia/Colombo")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVER_KEY = (
    os.getenv("SUPABASE_SECRET_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
)
NORMALIZATION_LOOKBACK_DAYS = int(os.getenv("NORMALIZATION_LOOKBACK_DAYS", "90"))

app = FastAPI(
    title="Component 2 Digital Phenotyping API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

_supabase: Optional[Client] = None


def get_supabase() -> Client:
    global _supabase

    if _supabase is not None:
        return _supabase

    if not SUPABASE_URL:
        raise RuntimeError("SUPABASE_URL is not configured.")

    if not SUPABASE_SERVER_KEY:
        raise RuntimeError(
            "No Supabase server key configured. Set SUPABASE_SECRET_KEY "
            "(recommended) or legacy SUPABASE_SERVICE_ROLE_KEY."
        )

    _supabase = create_client(
        SUPABASE_URL,
        SUPABASE_SERVER_KEY,
    )
    return _supabase


def local_midnight_to_utc_iso(d: date) -> str:
    local_dt = datetime.combine(d, time.min, tzinfo=LOCAL_TZ)
    utc_dt = local_dt.astimezone(timezone.utc)
    return utc_dt.isoformat().replace("+00:00", "Z")


def fetch_sensor_events(
    participant_id: str,
    end_date_exclusive: date,
) -> pd.DataFrame:
    start_date = end_date_exclusive - timedelta(
        days=NORMALIZATION_LOOKBACK_DAYS
    )

    start_iso = local_midnight_to_utc_iso(start_date)
    end_iso = local_midnight_to_utc_iso(end_date_exclusive)

    response = (
        get_supabase()
        .table("sensor_events")
        .select(
            "participant_code,event_time,event_type,value_json,source"
        )
        .eq("participant_code", participant_id)
        .gte("event_time", start_iso)
        .lt("event_time", end_iso)
        .order("event_time")
        .execute()
    )

    return pd.DataFrame(response.data or [])


@app.on_event("startup")
def startup_check() -> None:
    load_artifacts()


@app.get("/health")
def health():
    try:
        _, metadata = load_artifacts()

        return {
            "status": "ok",
            "service": "c2_behavioral",
            "model_version": metadata.get(
                "model_version",
                "M2_mobile_screen_location_v1",
            ),
            "fusion_default": (
                "enabled"
                if os.getenv("ENABLE_C2_FUSION", "0").strip() == "1"
                else "disabled"
            ),
            "supabase_configured": bool(
                SUPABASE_URL and SUPABASE_SERVER_KEY
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model service not ready: {exc}",
        ) from exc


@app.get("/behavioral/{participant_id}")
def behavioral(
    participant_id: str,
    window_end_date: Optional[str] = Query(default=None),
):
    participant_id = participant_id.strip()

    if not participant_id:
        raise HTTPException(
            status_code=400,
            detail="participant_id cannot be empty.",
        )

    try:
        end_date = (
            date.fromisoformat(window_end_date)
            if window_end_date
            else datetime.now(LOCAL_TZ).date()
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="window_end_date must be YYYY-MM-DD.",
        ) from exc

    try:
        events = fetch_sensor_events(
            participant_id,
            end_date,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Supabase query failed: {exc}",
        ) from exc

    if events.empty:
        return {
            "subject_id": participant_id,
            "modality": "c2_behavioral",
            "score": None,
            "status": "insufficient_data",
            "fusion_eligible": False,
            "behavioral_vulnerability_score": None,
            "reason": "No sensor events found in the available history.",
            "window_end": end_date.isoformat(),
        }

    try:
        return score_participant_events(
            rows=events,
            participant_id=participant_id,
            window_end_date=end_date.isoformat(),
            normalization_rows=events,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction/model scoring failed: {exc}",
        ) from exc
