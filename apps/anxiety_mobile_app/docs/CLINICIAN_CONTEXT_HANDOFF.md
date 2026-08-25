# Clinician Longitudinal Context Handoff v2

This document defines the privacy-safe longitudinal context that the Aura mobile
application can prepare for a clinician-facing backend. It is intentionally
separate from multimodal risk fusion.

## Purpose

The clinician view should not depend on one numerical score alone. A more useful
review combines four different streams while keeping their meanings separate:

```text
Self-report trend
        +
Physiological event confirmations
        +
Intervention response
        +
Component 2 behavioural changes
        ↓
Clinician longitudinal context
```

This context is intended to support clinical conversation and longitudinal
review. It is not a new risk score and must not silently change the multimodal
composite.

## Current transport status

The mobile app currently **builds and caches** the combined payload under:

`clinician_longitudinal_context_v2`

The older check-in/C2-only summary is still available internally as
`clinician_insight_handoff_v1` and is used as an input to the v2 builder.

The combined payload is **not automatically transmitted yet**. The central
backend does not currently expose a dedicated participant longitudinal-context
ingestion endpoint. The integration team should add a specific authenticated
endpoint before enabling transport. Do not overload the fusion endpoint with
this contextual payload.

## 1. Self-report trend

New EMA, GAD-7 and PSS-10 submissions are retained locally in a privacy-safe
summary history in addition to following their existing research-event upload
path.

### EMA

The trend may contain:

- number of EMA responses in the last 7 and 30 days;
- average reported anxiety/worry;
- average reported stress;
- average fatigue;
- average social connection;
- most common participant-selected context.

### GAD-7

The trend may contain:

- latest total score;
- latest severity label already produced by the questionnaire screen;
- previous total score, when available;
- numerical delta and descriptive direction compared with the previous result.

### PSS-10

The trend may contain:

- latest total score;
- previous total score, when available;
- numerical delta and descriptive direction compared with the previous result.

Only summary fields required for longitudinal review are copied into this local
history. Full questionnaire item responses are not duplicated into the
clinician-context cache.

**History limitation:** the local trend starts with submissions recorded after
this feature is installed. There is currently no historical Supabase backfill
into the patient app.

## 2. Physiological event confirmations

Aura's physiological alert/check-in workflow can be summarized over 7-day and
30-day windows using:

- number of alert-triggered check-ins;
- number answered by the participant;
- number where the participant reported feeling anxious;
- number where the participant did not report feeling anxious;
- response rate;
- confirmation rate among answered check-ins;
- common participant-reported context;
- recent event summaries.

A participant confirmation means only that the participant reported anxiety at
that check-in. It does **not** establish that every physiological alert was a
clinical anxiety episode or validate the forecasting model by itself.

## 3. Intervention response

For check-ins where an action was attempted, the context may summarize:

- number of intervention/action attempts;
- number of five-minute follow-ups answered;
- number reporting that they felt better;
- participant-reported improvement rate;
- the most frequently occurring action among follow-ups where the participant
  reported feeling better.

These outcomes are observational self-reports. They must not be described as
proof that an intervention caused improvement or as treatment-effect estimates.

## 4. Component 2 behavioural changes

The Component 2 section may include only descriptive within-person context:

- whether a personal baseline is ready;
- reportable/data-quality state;
- behavioural pattern direction relative to the participant's own baseline;
- sustained Day-57+ EWMA change detection when the backend actually detects it;
- recent usable sensing days and baseline coverage metadata.

The final Component 2 deployment decision remains explicit:

```json
{
  "status": "not_validated",
  "fusion_eligible": false,
  "score": null
}
```

The experimental Component 2 probability is not included in this clinician
context and does not enter the multimodal composite.

## Combined payload shape

Illustrative structure:

```json
{
  "schema_version": "clinician_longitudinal_context_v2",
  "app_user_id": "P_...",
  "generated_at": "2026-08-25T10:00:00Z",
  "self_report_trend": {
    "seven_day": {
      "ema": {
        "count": 12,
        "mean_stress": 2.1,
        "mean_anxiety": 2.8,
        "mean_fatigue": 2.4,
        "mean_social_connection": 3.2,
        "common_context": "Studying / Working"
      }
    },
    "gad7": {
      "available": true,
      "latest_score": 9,
      "previous_score": 7,
      "delta": 2,
      "direction": "higher_than_previous"
    },
    "pss10": {
      "available": true,
      "latest_score": 19,
      "previous_score": 16,
      "delta": 3,
      "direction": "higher_than_previous"
    }
  },
  "physiological_event_confirmations": {
    "thirty_day": {
      "events": 8,
      "answered": 7,
      "confirmed_anxiety": 5,
      "not_confirmed": 2,
      "confirmation_rate": 0.714,
      "common_context": "Studying or working"
    }
  },
  "intervention_response": {
    "thirty_day": {
      "intervention_attempts": 4,
      "followups_answered": 3,
      "felt_better_count": 2,
      "felt_better_rate": 0.667,
      "most_helpful_action": "2-minute paced breathing"
    }
  },
  "c2_behavioral_changes": {
    "status": "not_validated",
    "fusion_eligible": false,
    "score": null,
    "baseline_ready": true,
    "patterns": [
      {
        "label": "Screen activity",
        "direction": "above",
        "within_person_z": 1.4
      }
    ],
    "change_detection": {
      "detected": true,
      "feature": "screen activity",
      "direction": "above",
      "ewma_z": 2.1
    },
    "data_quality": {
      "baseline_days_required": 28,
      "baseline_usable_days": 26,
      "recent_usable_days": 7
    }
  },
  "fusion_policy": {
    "c2_status": "not_validated",
    "c2_fusion_eligible": false,
    "c2_score": null,
    "context_payload_affects_composite": false
  }
}
```

Values above are illustrative schema examples only and are not participant or
research-result values.

## Clinician presentation principles

The clinician interface should present the four streams as distinct sections or
aligned timelines rather than collapsing them into a new score. Useful questions
include:

- Did self-reported anxiety/stress change over time?
- Were physiological alert check-ins usually confirmed by the participant?
- In what situations were confirmed events commonly reported?
- What actions were attempted, and what did the participant report five minutes
  later?
- Did passive behavioural patterns show a sustained within-person change around
  the same period?
- Was sensing coverage adequate enough to interpret the behavioural context?

Temporal co-occurrence can be highlighted descriptively, but the system must not
claim causation from these observational streams.

## Explicit exclusions

The handoff must not contain:

- exact GPS coordinates or raw location trails;
- individual app/package names;
- SMS/call content, phone numbers or contact identifiers;
- a Component 2 clinical risk probability;
- a fabricated zero score for Component 2;
- synthetic PP2 fixture values in production;
- a new combined 'clinician risk score' calculated from this context payload.

## Recommended central-backend integration

Add a dedicated authenticated endpoint such as:

```text
POST /v1/subjects/{subject_id}/longitudinal-context
```

The backend should resolve/pair the participant identity, validate the payload
schema, persist the latest context and/or append time-stamped summaries, and
return it only through the clinician-authorized egress path.

This endpoint should remain separate from fusion ingestion. The contextual
payload may support clinician interpretation but should not alter fusion weights
or the composite unless a future validated protocol explicitly changes that
decision.
