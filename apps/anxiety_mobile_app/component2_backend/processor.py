from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from statistics import mean, pstdev
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import httpx


DEFAULT_TIMEZONE = os.getenv("COMPONENT2_TIMEZONE", "Asia/Colombo")
BASELINE_DAYS = 28
MIN_BASELINE_USABLE_DAYS = int(os.getenv("COMPONENT2_MIN_BASELINE_USABLE_DAYS", "14"))
RECENT_WINDOW_DAYS = 7
CHANGE_DETECTION_START_DAY = 57
EWMA_ALPHA = float(os.getenv("COMPONENT2_EWMA_ALPHA", "0.30"))
EWMA_THRESHOLD = float(os.getenv("COMPONENT2_EWMA_THRESHOLD", "1.50"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value or "").strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _payload(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("value_json")
    return value if isinstance(value, dict) else {}


def _date_range(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(h)))


def _overlap_minutes(start: datetime, end: datetime, window_start: datetime, window_end: datetime) -> float:
    left = max(start, window_start)
    right = min(end, window_end)
    if right <= left:
        return 0.0
    return (right - left).total_seconds() / 60.0


def _normalized_entropy(values: list[tuple[float, float]]) -> float | None:
    if not values:
        return None
    counts = Counter(values)
    if len(counts) <= 1:
        return 0.0
    total = sum(counts.values())
    entropy = -sum((n / total) * math.log(n / total) for n in counts.values())
    return entropy / math.log(len(counts))


def _weighted_average(items: list[tuple[float, int]]) -> float | None:
    total_weight = sum(max(0, weight) for _, weight in items)
    if total_weight <= 0:
        return None
    return sum(value * max(0, weight) for value, weight in items) / total_weight


def _aggregate_app_usage_windows(
    windows: list[tuple[datetime, float, dict[str, float]]],
) -> dict[str, float]:
    """Sum category usage from independent app-usage windows.

    The Android collector queries UsageStats over the previous 15 minutes, so
    every App_Usage_Category_15m event is already a window total. It must not be
    differenced against the previous event. If Android reports overlapping app
    totals whose combined duration exceeds the window length, scale that single
    window proportionally so it cannot contribute more time than was observed.
    """
    category_seconds: dict[str, float] = defaultdict(float)

    for _, window_seconds, categories in sorted(windows, key=lambda item: item[0]):
        clean = {
            category: max(0.0, seconds)
            for category, seconds in categories.items()
            if seconds > 0.0
        }
        total_seconds = sum(clean.values())
        if total_seconds <= 0.0:
            continue

        bounded_window_seconds = max(0.0, window_seconds)
        scale = (
            min(1.0, bounded_window_seconds / total_seconds)
            if bounded_window_seconds > 0.0
            else 0.0
        )
        for category, seconds in clean.items():
            category_seconds[category] += seconds * scale

    return dict(category_seconds)


class SupabaseRest:
    def __init__(self, url: str, service_role_key: str) -> None:
        self.url = url.rstrip("/")
        self.service_role_key = service_role_key
        self.client = httpx.AsyncClient(timeout=30.0)

    @classmethod
    def from_env(cls) -> "SupabaseRest":
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required.")
        return cls(url, key)

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    async def close(self) -> None:
        await self.client.aclose()

    async def get_rows(self, table: str, params: dict[str, str], *, page_size: int = 1000) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            query = dict(params)
            query["limit"] = str(page_size)
            query["offset"] = str(offset)
            response = await self.client.get(
                f"{self.url}/rest/v1/{table}",
                params=query,
                headers=self._headers(),
            )
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list):
                raise RuntimeError(f"Unexpected response while reading {table}.")
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        return rows

    async def upsert_rows(self, table: str, rows: list[dict[str, Any]], on_conflict: str) -> None:
        if not rows:
            return
        response = await self.client.post(
            f"{self.url}/rest/v1/{table}",
            params={"on_conflict": on_conflict},
            headers=self._headers(prefer="resolution=merge-duplicates,return=minimal"),
            json=rows,
        )
        response.raise_for_status()

    async def auth_user_from_token(self, access_token: str) -> dict[str, Any] | None:
        response = await self.client.get(
            f"{self.url}/auth/v1/user",
            headers={
                "apikey": self.service_role_key,
                "Authorization": f"Bearer {access_token}",
            },
        )
        if response.status_code != 200:
            return None
        body = response.json()
        return body if isinstance(body, dict) else None


class Component2Processor:
    def __init__(self, db: SupabaseRest, timezone_name: str = DEFAULT_TIMEZONE) -> None:
        self.db = db
        self.tz = ZoneInfo(timezone_name)

    @classmethod
    def from_env(cls) -> "Component2Processor":
        return cls(SupabaseRest.from_env(), DEFAULT_TIMEZONE)

    async def participant(self, participant_code: str) -> dict[str, Any] | None:
        rows = await self.db.get_rows(
            "participants",
            {
                "participant_code": f"eq.{participant_code}",
                "select": "auth_user_id,participant_code,enrolled_at,active",
            },
        )
        return rows[0] if rows else None

    async def verify_participant_token(self, participant_code: str, access_token: str) -> bool:
        user = await self.db.auth_user_from_token(access_token)
        if not user or not user.get("id"):
            return False
        participant = await self.participant(participant_code)
        return bool(
            participant
            and participant.get("active") is True
            and participant.get("auth_user_id") == user.get("id")
        )

    def _local_day_bounds_utc(self, day: date) -> tuple[datetime, datetime]:
        start_local = datetime.combine(day, time.min, tzinfo=self.tz)
        end_local = start_local + timedelta(days=1)
        return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)

    async def _events_for_range(
        self,
        participant_code: str,
        start_day: date,
        end_day: date,
    ) -> list[dict[str, Any]]:
        start_utc, _ = self._local_day_bounds_utc(start_day)
        _, buffered_end_utc = self._local_day_bounds_utc(end_day + timedelta(days=1))
        return await self.db.get_rows(
            "sensor_events",
            {
                "participant_code": f"eq.{participant_code}",
                "event_time": f"gte.{start_utc.isoformat()}",
                "and": f"(event_time.lt.{buffered_end_utc.isoformat()})",
                "select": "event_time,event_type,value_json",
                "order": "event_time.asc",
            },
        )

    def _bucket_events(
        self,
        events: list[dict[str, Any]],
    ) -> dict[date, list[dict[str, Any]]]:
        buckets: dict[date, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            payload = _payload(event)
            event_type = str(event.get("event_type") or "")
            if event_type in {"Call_Stats_Daily", "SMS_Activity_Daily"} and payload.get("date"):
                try:
                    target_day = date.fromisoformat(str(payload["date"]))
                except ValueError:
                    target_day = _parse_timestamp(event["event_time"]).astimezone(self.tz).date()
            else:
                target_day = _parse_timestamp(event["event_time"]).astimezone(self.tz).date()
            buckets[target_day].append(event)
        return buckets

    def aggregate_day(
        self,
        auth_user_id: str,
        participant_code: str,
        day: date,
        events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        day_start = datetime.combine(day, time.min, tzinfo=self.tz)
        day_end = day_start + timedelta(days=1)
        night_end = day_start + timedelta(hours=6)

        screen_events: list[tuple[datetime, str]] = []
        locations: list[tuple[datetime, float, float]] = []
        movements: list[dict[str, Any]] = []
        app_windows: list[tuple[datetime, float, dict[str, float]]] = []
        heartbeats = 0
        calls: dict[str, int] = {}
        sms: dict[str, int] = {}

        for event in events:
            event_type = str(event.get("event_type") or "")
            value = _payload(event)
            event_dt = _parse_timestamp(event["event_time"]).astimezone(self.tz)

            if event_type == "Screen_Event":
                state = str(value.get("state") or "")
                screen_events.append((event_dt, state))
            elif event_type == "Location_Grid_100m":
                if value.get("lat") is not None and value.get("lng") is not None:
                    locations.append((event_dt, _as_float(value["lat"]), _as_float(value["lng"])))
            elif event_type == "Movement_Window_5m":
                movements.append(value)
            elif event_type == "App_Usage_Category_15m":
                raw_categories = value.get("categories_sec")
                if isinstance(raw_categories, dict):
                    window_minutes = min(
                        60.0,
                        max(1.0, _as_float(value.get("window_minutes"), 15.0)),
                    )
                    app_windows.append(
                        (
                            event_dt,
                            window_minutes * 60.0,
                            {str(k): max(0.0, _as_float(v)) for k, v in raw_categories.items()},
                        )
                    )
            elif event_type == "Service_Heartbeat":
                heartbeats += 1
            elif event_type == "Call_Stats_Daily":
                calls = {
                    "incoming": max(_as_int(value.get("incoming")), calls.get("incoming", 0)),
                    "outgoing": max(_as_int(value.get("outgoing")), calls.get("outgoing", 0)),
                    "missed": max(_as_int(value.get("missed")), calls.get("missed", 0)),
                    "rejected": max(_as_int(value.get("rejected")), calls.get("rejected", 0)),
                }
            elif event_type == "SMS_Activity_Daily":
                sms = {
                    "sent": max(_as_int(value.get("sent")), sms.get("sent", 0)),
                    "received": max(_as_int(value.get("received")), sms.get("received", 0)),
                }

        screen_events.sort(key=lambda item: item[0])
        screen_minutes = 0.0
        night_screen_minutes = 0.0
        unlock_count = 0
        screen_on_since: datetime | None = None

        for event_dt, state in screen_events:
            if state == "Screen_Unlocked":
                unlock_count += 1
                if screen_on_since is None:
                    screen_on_since = max(event_dt, day_start)
            elif state == "Screen_On":
                if screen_on_since is None:
                    screen_on_since = max(event_dt, day_start)
            elif state == "Screen_Off" and screen_on_since is not None:
                interval_end = min(event_dt, day_end)
                if interval_end > screen_on_since:
                    screen_minutes += (interval_end - screen_on_since).total_seconds() / 60.0
                    night_screen_minutes += _overlap_minutes(
                        screen_on_since,
                        interval_end,
                        day_start,
                        night_end,
                    )
                screen_on_since = None

        if screen_on_since is not None and screen_on_since < day_end:
            capped_end = min(day_end, screen_on_since + timedelta(minutes=30))
            screen_minutes += (capped_end - screen_on_since).total_seconds() / 60.0
            night_screen_minutes += _overlap_minutes(screen_on_since, capped_end, day_start, night_end)

        locations.sort(key=lambda item: item[0])
        distance_km = 0.0
        location_cells: list[tuple[float, float]] = []
        for _, lat, lng in locations:
            location_cells.append((lat, lng))
        for previous, current in zip(location_cells, location_cells[1:]):
            step = _haversine_km(previous, current)
            if step <= 25.0:
                distance_km += step

        cell_counts = Counter(location_cells)
        primary_count = max(cell_counts.values(), default=0)
        home_minutes = min(1440.0, primary_count * 15.0)
        significant_places = len(cell_counts)
        location_entropy = _normalized_entropy(location_cells)

        movement_mean = _weighted_average(
            [(_as_float(v.get("mean_magnitude")), _as_int(v.get("sample_count"))) for v in movements]
        )
        movement_variability = _weighted_average(
            [(_as_float(v.get("std_magnitude")), _as_int(v.get("sample_count"))) for v in movements]
        )
        high_motion_fraction = _weighted_average(
            [(_as_float(v.get("high_motion_fraction")), _as_int(v.get("sample_count"))) for v in movements]
        )

        # Every App_Usage_Category_15m row is already the total for its own
        # previous-15-minute interval. Sum those independent intervals directly;
        # do not subtract consecutive snapshots as if they were cumulative.
        category_seconds = _aggregate_app_usage_windows(app_windows)

        location_coverage = min(1.0, len(locations) / 96.0)
        movement_coverage = min(1.0, len(movements) / 288.0)
        heartbeat_coverage = min(1.0, heartbeats / 24.0)
        screen_coverage = min(
            1.0,
            heartbeat_coverage + (0.10 if screen_events else 0.0),
        )

        modality_flags = [
            location_coverage >= 0.10,
            movement_coverage >= 0.10,
            len(screen_events) >= 2,
        ]
        usable_day = sum(modality_flags) >= 2 and max(
            location_coverage,
            movement_coverage,
            heartbeat_coverage,
        ) >= 0.50

        return {
            "auth_user_id": auth_user_id,
            "participant_code": participant_code,
            "feature_date": day.isoformat(),
            "screen_minutes": round(screen_minutes, 2),
            "unlock_count": unlock_count,
            "night_screen_minutes": round(night_screen_minutes, 2),
            "distance_km": round(distance_km, 3),
            "home_minutes": round(home_minutes, 1),
            "significant_places": significant_places,
            "location_entropy": None if location_entropy is None else round(location_entropy, 4),
            "movement_mean": None if movement_mean is None else round(movement_mean, 4),
            "movement_variability": None if movement_variability is None else round(movement_variability, 4),
            "high_motion_fraction": None if high_motion_fraction is None else round(high_motion_fraction, 5),
            "social_media_minutes": round(category_seconds.get("Social_Media", 0.0) / 60.0, 2),
            "entertainment_minutes": round(category_seconds.get("Entertainment", 0.0) / 60.0, 2),
            "education_minutes": round(category_seconds.get("Education", 0.0) / 60.0, 2),
            "incoming_calls": calls.get("incoming", 0),
            "outgoing_calls": calls.get("outgoing", 0),
            "missed_calls": calls.get("missed", 0),
            "rejected_calls": calls.get("rejected", 0),
            "sms_sent": sms.get("sent", 0),
            "sms_received": sms.get("received", 0),
            "routine_regularity": None,
            "location_coverage": round(location_coverage, 4),
            "screen_coverage": round(screen_coverage, 4),
            "movement_coverage": round(movement_coverage, 4),
            "usable_day": usable_day,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def process_participant(
        self,
        participant_code: str,
        *,
        include_today: bool = False,
    ) -> dict[str, Any]:
        participant = await self.participant(participant_code)
        if not participant:
            raise ValueError("Participant not found.")
        if participant.get("active") is not True:
            raise ValueError("Participant is inactive.")

        enrolled_date = _parse_timestamp(participant["enrolled_at"]).astimezone(self.tz).date()
        today = datetime.now(self.tz).date()
        end_day = today if include_today else today - timedelta(days=1)
        if end_day < enrolled_date:
            return {
                "participant_code": participant_code,
                "processed_days": 0,
                "observation_written": False,
            }

        events = await self._events_for_range(participant_code, enrolled_date, end_day)
        buckets = self._bucket_events(events)

        feature_rows = [
            self.aggregate_day(
                participant["auth_user_id"],
                participant_code,
                day,
                buckets.get(day, []),
            )
            for day in _date_range(enrolled_date, end_day)
        ]

        await self.db.upsert_rows(
            "daily_behavior_features",
            feature_rows,
            "auth_user_id,feature_date",
        )

        all_features = await self.daily_features(participant_code)
        observation = self.build_observation(participant, all_features)
        await self.db.upsert_rows(
            "behavioral_observations",
            [observation],
            "auth_user_id,window_end",
        )

        return {
            "participant_code": participant_code,
            "processed_days": len(feature_rows),
            "raw_events_seen": len(events),
            "observation_written": True,
            "baseline_ready": observation["baseline_ready"],
            "reportable": observation["reportable"],
        }

    async def process_all(self, *, include_today: bool = False) -> list[dict[str, Any]]:
        participants = await self.db.get_rows(
            "participants",
            {
                "active": "eq.true",
                "select": "participant_code",
                "order": "enrolled_at.asc",
            },
        )
        results: list[dict[str, Any]] = []
        for participant in participants:
            code = str(participant.get("participant_code") or "")
            if not code:
                continue
            try:
                results.append(await self.process_participant(code, include_today=include_today))
            except Exception as exc:
                results.append({"participant_code": code, "error": str(exc)})
        return results

    async def daily_features(self, participant_code: str) -> list[dict[str, Any]]:
        return await self.db.get_rows(
            "daily_behavior_features",
            {
                "participant_code": f"eq.{participant_code}",
                "select": "*",
                "order": "feature_date.asc",
            },
        )

    def _feature_stats(
        self,
        baseline: list[dict[str, Any]],
        key: str,
    ) -> tuple[float, float] | None:
        values = [
            _as_float(row.get(key))
            for row in baseline
            if row.get(key) is not None
        ]
        if len(values) < 5:
            return None
        return mean(values), pstdev(values)

    def _observation_item(
        self,
        baseline: list[dict[str, Any]],
        recent: list[dict[str, Any]],
        *,
        key: str,
        label: str,
        unit: str,
        display_scale: float = 1.0,
    ) -> dict[str, Any] | None:
        stats = self._feature_stats(baseline, key)
        recent_values = [
            _as_float(row.get(key))
            for row in recent
            if row.get(key) is not None
        ]
        if not stats or not recent_values:
            return None

        baseline_mean, baseline_std = stats
        recent_mean = mean(recent_values)
        if baseline_std < 1e-9:
            z = 0.0 if abs(recent_mean - baseline_mean) < 1e-9 else None
        else:
            z = (recent_mean - baseline_mean) / baseline_std

        if z is None:
            direction = "unknown"
        elif z >= 1.0:
            direction = "above"
        elif z <= -1.0:
            direction = "below"
        else:
            direction = "stable"

        return {
            "label": label,
            "value": round(recent_mean * display_scale, 2),
            "unit": unit,
            "z": None if z is None else round(z, 3),
            "direction": direction,
        }

    def _change_detection(
        self,
        baseline: list[dict[str, Any]],
        post_baseline: list[dict[str, Any]],
        days_enrolled: int,
    ) -> dict[str, Any] | None:
        if days_enrolled < CHANGE_DETECTION_START_DAY:
            return None

        feature_map = {
            "screen_minutes": "screen activity",
            "distance_km": "mobility",
            "high_motion_fraction": "movement proxy",
            "social_media_minutes": "social media use",
        }
        strongest: tuple[str, float] | None = None

        for key, label in feature_map.items():
            stats = self._feature_stats(baseline, key)
            if not stats:
                continue
            base_mean, base_std = stats
            if base_std < 1e-9:
                continue

            z_values = [
                (_as_float(row.get(key)) - base_mean) / base_std
                for row in post_baseline
                if row.get("usable_day") is True and row.get(key) is not None
            ]
            if len(z_values) < 7:
                continue

            ewma = z_values[0]
            for z in z_values[1:]:
                ewma = EWMA_ALPHA * z + (1.0 - EWMA_ALPHA) * ewma

            if strongest is None or abs(ewma) > abs(strongest[1]):
                strongest = (label, ewma)

        if strongest is None:
            return {"detected": False}

        label, score = strongest
        if abs(score) < EWMA_THRESHOLD:
            return {"detected": False}

        return {
            "detected": True,
            "feature": label,
            "direction": "above" if score > 0 else "below",
            "ewma_z": round(score, 3),
            "message": (
                f"A sustained change in {label} compared with your usual pattern "
                "was detected."
            ),
        }

    def build_observation(
        self,
        participant: dict[str, Any],
        feature_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        participant_code = str(participant["participant_code"])
        auth_user_id = str(participant["auth_user_id"])
        enrolled_date = _parse_timestamp(participant["enrolled_at"]).astimezone(self.tz).date()
        today = datetime.now(self.tz).date()
        days_enrolled = max(1, (today - enrolled_date).days + 1)
        baseline_end = enrolled_date + timedelta(days=BASELINE_DAYS - 1)

        rows = sorted(feature_rows, key=lambda row: str(row.get("feature_date") or ""))
        baseline = [
            row
            for row in rows
            if row.get("usable_day") is True
            and enrolled_date <= date.fromisoformat(str(row["feature_date"])) <= baseline_end
        ]
        post_baseline = [
            row
            for row in rows
            if date.fromisoformat(str(row["feature_date"])) > baseline_end
        ]
        post_usable = [row for row in post_baseline if row.get("usable_day") is True]
        recent = post_usable[-RECENT_WINDOW_DAYS:]

        baseline_ready = (
            days_enrolled >= BASELINE_DAYS
            and len(baseline) >= MIN_BASELINE_USABLE_DAYS
        )
        reportable = baseline_ready and len(recent) >= 3

        observations: dict[str, Any] = {}
        if reportable:
            configs = [
                ("screen_activity", "screen_minutes", "Screen activity", "hours/day", 1.0 / 60.0),
                ("mobility", "distance_km", "Mobility", "km/day", 1.0),
                (
                    "movement_proxy",
                    "high_motion_fraction",
                    "Movement proxy",
                    "% high-motion samples",
                    100.0,
                ),
                (
                    "social_media_use",
                    "social_media_minutes",
                    "Social media use",
                    "min/day",
                    1.0,
                ),
            ]
            for output_key, feature_key, label, unit, scale in configs:
                item = self._observation_item(
                    baseline,
                    recent,
                    key=feature_key,
                    label=label,
                    unit=unit,
                    display_scale=scale,
                )
                if item is not None:
                    observations[output_key] = item

        recent_14 = rows[-14:]
        usable_last_14 = sum(1 for row in recent_14 if row.get("usable_day") is True)
        baseline_days_available = min(BASELINE_DAYS, days_enrolled)
        baseline_period_rows = [
            row
            for row in rows
            if enrolled_date
            <= date.fromisoformat(str(row["feature_date"]))
            <= baseline_end
        ]
        baseline_days_with_features = sum(
            1
            for row in baseline_period_rows
            if (
                _as_float(row.get("location_coverage")) > 0
                or _as_float(row.get("screen_coverage")) > 0
                or _as_float(row.get("movement_coverage")) > 0
            )
        )

        if recent:
            window_start = str(recent[0]["feature_date"])
            window_end = str(recent[-1]["feature_date"])
        elif rows:
            window_start = str(rows[max(0, len(rows) - 7)]["feature_date"])
            window_end = str(rows[-1]["feature_date"])
        else:
            window_start = enrolled_date.isoformat()
            window_end = min(today, baseline_end).isoformat()

        return {
            "auth_user_id": auth_user_id,
            "participant_code": participant_code,
            "window_start": window_start,
            "window_end": window_end,
            "baseline_ready": baseline_ready,
            "reportable": reportable,
            "observations": observations,
            "data_quality": {
                "days_enrolled": days_enrolled,
                "days_with_data": usable_last_14,
                "baseline_calendar_days_elapsed": min(
                    BASELINE_DAYS,
                    days_enrolled,
                ),
                "baseline_days_with_features": baseline_days_with_features,
                "baseline_days_available": baseline_days_available,
                "baseline_days_required": BASELINE_DAYS,
                "baseline_usable_days": len(baseline),
                "baseline_min_usable_days": MIN_BASELINE_USABLE_DAYS,
                "recent_usable_days": len(recent),
                "ema_received": 0,
                "ema_expected": 0,
            },
            "change_detection": self._change_detection(
                baseline,
                post_baseline,
                days_enrolled,
            ),
            "model_output": None,
            "model_status": "withheld_pending_validation",
        }

    async def behavioral_payload(self, participant_code: str) -> dict[str, Any]:
        participant = await self.participant(participant_code)
        if not participant:
            raise ValueError("Participant not found.")

        feature_rows = await self.daily_features(participant_code)
        observation_rows = await self.db.get_rows(
            "behavioral_observations",
            {
                "participant_code": f"eq.{participant_code}",
                "select": "*",
                "order": "created_at.desc",
                "limit": "1",
            },
        )

        if observation_rows:
            stored = observation_rows[0]
            observation_payload = {
                "participant_id": participant_code,
                "window": {
                    "start": stored["window_start"],
                    "end": stored["window_end"],
                },
                "baseline_ready": stored.get("baseline_ready", False),
                "reportable": stored.get("reportable", False),
                "observations": stored.get("observations") or {},
                "data_quality": stored.get("data_quality") or {},
                "change_detection": stored.get("change_detection"),
                "blocking_issues": [],
            }
            quality = observation_payload["data_quality"]
            if not observation_payload["baseline_ready"]:
                observation_payload["blocking_issues"].append("baseline_building")
            elif _as_int(quality.get("recent_usable_days")) < 3:
                observation_payload["blocking_issues"].append("insufficient_recent_data")
        else:
            built = self.build_observation(participant, feature_rows)
            observation_payload = {
                "participant_id": participant_code,
                "window": {
                    "start": built["window_start"],
                    "end": built["window_end"],
                },
                "baseline_ready": built["baseline_ready"],
                "reportable": built["reportable"],
                "observations": built["observations"],
                "data_quality": built["data_quality"],
                "change_detection": built["change_detection"],
                "blocking_issues": (
                    ["baseline_building"] if not built["baseline_ready"] else []
                ),
            }

        recent = [row for row in feature_rows if row.get("usable_day") is True][-RECENT_WINDOW_DAYS:]
        if recent:
            home_hours = mean(_as_float(row.get("home_minutes")) for row in recent) / 60.0
            significant_places = mean(_as_float(row.get("significant_places")) for row in recent)
            movement_values = [
                _as_float(row.get("high_motion_fraction"))
                for row in recent
                if row.get("high_motion_fraction") is not None
            ]
            passive_metrics = {
                "home_hours": round(home_hours, 2),
                "away_hours": round(max(0.0, 24.0 - home_hours), 2),
                "significant_places": round(significant_places, 1),
                "activity_proxy_score": (
                    round(mean(movement_values), 4) if movement_values else None
                ),
                "activity_data_available": bool(movement_values),
            }
        else:
            passive_metrics = {"activity_data_available": False}

        day_coverage = [
            {
                "date": row["feature_date"],
                "usable": bool(row.get("usable_day")),
                "location_coverage": row.get("location_coverage"),
                "screen_coverage": row.get("screen_coverage"),
                "movement_coverage": row.get("movement_coverage"),
            }
            for row in feature_rows[-14:]
        ]

        return {
            "observation_payload": observation_payload,
            "passive_metrics": passive_metrics,
            "day_coverage": day_coverage,
            "checkin_history": [],
            "model_output": None,
            "model_status": "withheld_pending_validation",
        }
