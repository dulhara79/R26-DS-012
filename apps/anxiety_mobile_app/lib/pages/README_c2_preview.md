# Component 2 — Behavioural observations

This branch intentionally treats Component 2 as **behavioural monitoring**, not
as a validated anxiety-risk predictor.

The current deployment contract is:

- participant-facing app: behavioural descriptives, personal-baseline
  deviations and data-quality information only;
- GATv2 probability: research-only and never displayed in the participant UI;
- multimodal fusion: `behavioral_score: null`, `recommended_weight: 0.0`,
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
  "participant_id": "613",
  "model_status": "withheld_pending_validation",
  "fusion_eligible": false,
  "behavioral_score": null,
  "recommended_weight": 0.0,
  "display_permitted": false
}
```

This preserves the interface needed by Component 3 without allowing the current
GATv2 probability to influence the production fusion result.

## Chrome preview

The original preview assets can still be used for UI development where present:

```bash
flutter config --enable-web
flutter run -d chrome -t lib/pages/c2_preview_main.dart
```

Preview/synthetic fixtures must remain development-only and must not be used as
research results or participant data.

## Remaining backend work

The Flutter side is now wired for production transport, but the research
pipeline/backend still has to expose the `/behavioral/{participant_id}` endpoint
and generate the descriptive fields above from real participant data.

The app should only surface EWMA/change-detection messages when the backend has
actually computed and validated that signal. A baseline z-score by itself must
not be relabelled as an anxiety-risk score.
