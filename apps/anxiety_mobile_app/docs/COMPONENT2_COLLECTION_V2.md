# Component 2 Behavioural Collection v2

This branch uses a privacy-preserving, storage-neutral behavioural event pipeline. Google Sheets remains the current transport target, but sensor producers no longer depend on Sheets-specific method names so the uploader can later be replaced with Supabase/PostgreSQL.

## Event schema

Every queued event contains:

- `userId`
- `dataType`
- `value` (JSON serialized by the queue)
- `timestamp`
- transport authentication metadata while Google Apps Script remains active

## Current behavioural events

| Event | Frequency | Purpose |
|---|---:|---|
| `Screen_Event` | event-driven | screen on/off/unlock timing and unlock counts |
| `Location_Grid_100m` | ~15 min | privacy-coarsened mobility/location context |
| `App_Usage_Category_15m` | ~15 min | category-level app usage; package names stay on device |
| `Movement_Window_5m` | 5 min | movement proxy using accelerometer-window summaries |
| `Call_Stats_Daily` | once per complete previous day | daily call-count aggregate only |
| `SMS_Activity_Daily` | once per complete previous day | daily SMS-count aggregate only |
| `Battery_Status` | hourly | data-quality / missingness context |
| `Service_Heartbeat` | hourly | collection availability / gap detection |

EMA, GAD-7, PSS-10, consent and physiological records continue through their existing paths.

## Privacy changes

### Location

The exact GPS fix is not uploaded. Latitude/longitude are rounded on the participant device to three decimal places before queueing. This is approximately neighbourhood-scale resolution rather than the former server-side ~1 km grid. The event is named `Location_Grid_100m` so the legacy Apps Script does not apply the old `Location` sanitizer a second time.

The future Component 2 backend should derive mobility features such as distance travelled, time at primary location, significant-place count and location regularity. Long-term retention of coordinates should be minimized once those features are available.

### App usage

Package names are categorized on-device. Only category totals such as `Social_Media`, `Education`, `Entertainment`, `Browser`, and `Other` are uploaded.

### Communication

No phone numbers, SMS bodies, contact names or call content are collected. Only complete previous-day counts are stored. The collector also avoids collecting days before the recorded enrollment date.

## Movement interpretation

`Movement_Window_5m` is a movement proxy, not a validated physical-activity classifier. It includes:

- sample count
- mean acceleration magnitude
- standard deviation of magnitude
- fraction of samples above the high-motion threshold

The participant-facing product should not label this as clinical physical activity until an activity-recognition method is validated.

## Offline queue

`BackgroundServiceHelper.enqueueResearchEvent(...)` is now the preferred producer API. It persists events locally before upload. `sendToSheet(...)` remains only as a compatibility alias for older call sites.

Current transport:

`sensor -> offline queue -> Google Apps Script -> participant spreadsheet`

Planned transport:

`sensor -> offline queue -> authenticated ingest API/Supabase -> PostgreSQL`

## Recommended Supabase mapping

Raw/short-retention table:

`behavioral_sensor_events(id, participant_id, event_time, event_type, value_json, received_at)`

Processed long-retention table:

`daily_behavior_features(participant_id, feature_date, screen_minutes, unlock_count, distance_km, primary_location_minutes, significant_places, app_category_minutes, movement_index, call_counts, sms_counts, routine_regularity, usable_day, quality_json)`

Participant-facing output table:

`behavioral_observations(participant_id, window_start, window_end, baseline_ready, reportable, observations_json, data_quality_json, change_detection_json, created_at)`

## Component 2 inference policy

The current behavioural model probability must not be displayed as an anxiety risk score or used as an active fusion input until stronger validation is available. Participant-facing outputs remain descriptive personal-baseline observations and change-detection summaries.
