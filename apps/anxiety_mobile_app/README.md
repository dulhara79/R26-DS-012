# 🌿 Aura — Anxiety Research Mobile App

Flutter Android application used by **R26-DS-012** for longitudinal anxiety
research. The app integrates wearable physiological monitoring, passive
smartphone sensing, study questionnaires, offline-first collection, and
participant-facing behavioural context.

> **Research use only.** Participant-facing behavioural observations are not an
> anxiety diagnosis or a calibrated clinical risk probability.

## Current architecture

```text
ESP32-C3 chest strap
        ↓
Flutter physiological dashboard
        ↓
Component 1 physiological service

Android passive sensing
        ↓
local offline queue
        ↓
Supabase sensor_events
        ↓
Component 2 daily processor
        ↓
daily_behavior_features
        ↓
behavioral_observations
        ↓
Component 2 API
        ↓
Behavioural Context UI
```

## Component 2 collection

| Event | Collection pattern |
|---|---|
| `Screen_Event` | Event-driven screen on / unlock / off transitions |
| `Location_Grid_100m` | Approximately every 15 minutes; coordinates rounded to 3 decimals on-device |
| `App_Usage_Category_15m` | Approximately every 15 minutes; category totals only |
| `Movement_Window_5m` | Five-minute accelerometer summary |
| `Call_Stats_Daily` | Previous-day aggregate counts |
| `SMS_Activity_Daily` | Previous-day aggregate counts |
| `Battery_Status` | Hourly |
| `Service_Heartbeat` | Hourly |

Screen events are event-driven. They may reach Supabase later in a batch, but
their original `event_time` is preserved and is what the Component 2 processor
uses.

## Privacy

- Pseudonymous participant codes are used instead of names.
- Location is coarsened on-device before queueing.
- App package names are converted to broad categories on-device.
- Call/SMS content, phone numbers and contact names are not uploaded by the
  Component 2 collector.
- Mobile Supabase access uses a publishable key and anonymous authenticated
  participant session. **Never place a Supabase service-role/secret key in the
  APK.**
- `ChestStrap_Vitals` is blocked from the Component 2/general
  `sensor_events` stream because it belongs to the physiological component.

## Component 2 inference policy

The final v8 behavioural GATv2 study did not demonstrate a validated clinical
risk signal. Therefore the mobile handoff is explicit:

```json
{
  "modality": "c2_behavioral",
  "score": null,
  "status": "not_validated",
  "fusion_eligible": false,
  "behavioral_score": null,
  "behavioral_weight": 0.0,
  "recommended_weight": 0.0
}
```

Descriptive personal-baseline observations and data-quality information may be
shown to participants. The experimental behavioural model probability is not
presented as a clinical anxiety probability.

## Build

```powershell
flutter pub get

flutter build apk --release `
  --obfuscate `
  --split-debug-info=./debug-info `
  --dart-define=SUPABASE_URL="YOUR_SUPABASE_URL" `
  --dart-define=SUPABASE_PUBLISHABLE_KEY="YOUR_PUBLISHABLE_KEY" `
  --dart-define=COMPONENT2_API_URL="https://YOUR_COMPONENT2_BACKEND/api"
```

## Component 2 daily processing

The scheduled backend workflow processes completed behavioural days and writes
the derived tables used by the app. For a manual backfill/test, run the
**Component 2 Daily Processing** GitHub Action against `main`.

## PP2 demo fallback

The Component 2 web preview contains an explicitly labelled **synthetic,
development-only** 60-day fixture so the Day-57+ EWMA change-detection UI can
be demonstrated without contaminating research results. It is unavailable in a
release APK.

```bash
flutter config --enable-web
flutter run -d chrome -t lib/pages/c2_preview_main.dart
```

Synthetic/demo values must never be presented as participant data or model
validation evidence.
