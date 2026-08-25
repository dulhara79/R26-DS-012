# Component 2 — Behavioural observations

This branch intentionally treats Component 2 as **behavioural monitoring**, not
as a validated anxiety-risk predictor.

The current deployment contract is:

- participant-facing app: behavioural descriptives, personal-baseline
  deviations and data-quality information only;
- GATv2 probability: research-only and never displayed in the participant UI;
- multimodal fusion: `score: null`, `behavioral_score: null`,
  `behavioral_weight: 0.0`, `recommended_weight: 0.0`,
  `fusion_eligible: false` until incremental fusion value is validated.

A numeric score of `0` is deliberately avoided because it would mean "very low
risk". `null` means that Component 2 does not currently provide a validated
risk estimate.

## Production mobile integration

The Activity tab now opens `Component2BootstrapPage`. Before rendering the
existing `DigitalPhenotypingPage`, it calls `Component2DataService.sync()`.
The service fetches Component 2 output, allow-lists participant-safe fields and
stores them under the existing SharedPreferences keys consumed by the page.

Configure the backend URL with a Dart define:

```bash
flutter run \
  --dart-define=COMPONENT2_API_URL=https://YOUR-COMPONENT2-BACKEND
```

Expected endpoint:

```text
GET /behavioral/{participant_id}
```

The endpoint can return either an observation payload directly or this envelope:

```json
{
  "observation_payload": {
    "participant_id": "613",
    "window": {
      "start": "2026-07-20",
      "end": "2026-08-16"
    },
    "baseline_ready": true,
    "reportable": true,
    "observations": {
      "screen_time": {
        "label": "Screen time",
        "value": 6.2,
        "unit": "hours/day",
        "z": 1.42,
        "direction": "above",
        "confidence": "high"
      },
      "distance_travelled": {
        "label": "Distance travelled",
        "value": 3.8,
        "unit": "km/day",
        "z": -1.64,
        "direction": "below",
        "confidence": "high"
      }
    },
    "data_quality": {
      "days_with_data": 27,
      "baseline_days_available": 28,
      "baseline_days_required": 28,
      "ema_received": 19,
      "ema_expected": 21
    },
    "blocking_issues": []
  },
  "passive_metrics": {
    "home_hours": 15.1,
    "away_hours": 8.9,
    "significant_places": 3,
    "sleep_proxy_window": "11:42 PM – 7:05 AM",
    "overnight_screen_off_hours": 7.4,
    "activity_proxy_score": 0.61,
    "activity_data_available": true
  },
  "day_coverage": [
    {"date": "2026-08-15", "usable": true},
    {"date": "2026-08-16", "usable": true}
  ],
  "checkin_history": []
}
```

`Component2DataService` deliberately drops research-only model probabilities,
risk labels and other inferential fields before caching data for the UI.

If the API is not configured or cannot be reached, the page continues using the
last cached payload. If no cached payload exists, it retains the existing honest
"building your baseline" state rather than inventing values.

## Fusion handoff

Every successful Component 2 sync also stores an explicit fusion contract in
`c2_fusion_handoff`:

```json
{
  "component": "behavioral",
  "modality": "c2_behavioral",
  "participant_id": "613",
  "model_status": "withheld_pending_validation",
  "status": "not_validated",
  "fusion_eligible": false,
  "score": null,
  "behavioral_score": null,
  "behavioral_weight": 0.0,
  "recommended_weight": 0.0,
  "display_permitted": false
}
```

This preserves explicit evidence for the Component 4 fusion/orchestration layer
without allowing the current GATv2 probability to influence the production
fusion result. The central backend independently enforces the same exclusion.

## Chrome preview

The preview fixture is development-only and intentionally provides a 60-day
synthetic history with a labelled Day-57+ change-detection example. It is a
fallback for demonstrating the UI when a live participant has not accumulated
enough study history. It is not evidence of model accuracy and cannot activate
inside a release APK.

```bash
flutter config --enable-web
flutter run -d chrome -t lib/pages/c2_preview_main.dart
```

Preview/synthetic fixtures must remain development-only and must not be used as
research results or participant data.

## Current backend status

The Flutter side and Component 2 backend are wired for production transport.
The backend exposes the behavioural endpoint used by `Component2DataService`,
derives `daily_behavior_features`, writes `behavioral_observations`, and returns
descriptive observations, coverage, and change-detection data.

The app should only surface EWMA/change-detection messages when the backend has
actually computed and validated that signal. A baseline z-score by itself must
not be relabelled as an anxiety-risk score.
