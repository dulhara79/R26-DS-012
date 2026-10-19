# ClinAnx P5A — Signals, Contributions & Data Quality Design

**Status:** Approved by user in chat on 2026-09-16  
**Base branch:** `main`  
**Base commit:** `02cff6a5048e89362e34fd6426adb2527c2ae1d6`  
**Implementation branch:** `integration/clinanx-p5a-signals-data-quality`

## 1. Purpose

P5A deepens the existing Patient Overview without creating another assessment authority.

This phase adds two dedicated clinician-facing deep views:

1. **Signals & Contributions** — explains which modality signals the backend reported for the current assessment and what contribution values the backend supplied.
2. **Data Quality** — explains whether each modality was available, fresh/valid, sufficiently covered, and included in fusion when the current assessment was produced.

P5A preserves the system invariant established in P3/P4:

> One patient -> one backend-owned assessment -> multiple read-only clinician projections of that same assessment.

P5A does not create a second risk engine, does not recompute fusion, and does not invent backend fields or routes.

## 2. Source of Truth

`PatientOverviewScreen` remains the owner of the current canonical `AssessmentSummary` context.

The two P5A screens receive the exact same immutable `AssessmentSummary` instance already being displayed by Patient Overview:

```text
Patient Overview
      |
      | same subject_id
      | same fusion_result_id
      | same AssessmentSummary
      |
      +------> Signals & Contributions
      |
      +------> Data Quality
```

Required identity invariant:

```text
PatientOverview.assessment.fusionResultId
    ==
SignalsContributions.assessment.fusionResultId
    ==
DataQuality.assessment.fusionResultId
```

Neither deep screen performs its own assessment fetch on initial navigation. This prevents a clinician from opening a detail view and unintentionally seeing a different assessment revision than the overview they just selected.

Refresh remains owned by Patient Overview. A later refreshed canonical assessment may be passed into newly opened deep views, but P5A does not invent independent refresh endpoints.

## 3. Current Contract Reuse

P5A uses the existing `AssessmentSummary` / `ModalityStatus` contract introduced for the target clinician architecture.

Relevant server-derived fields already available per modality:

- `componentId`
- `score`
- `available`
- `includedInFusion`
- `state`
- `confidence`
- `coverage`
- `capturedAt`
- `contribution`

Relevant assessment-level fields:

- `subjectId`
- `fusionResultId`
- `assessmentStatus`
- `computedAt`
- `modelVersion`
- current assessment object
- optional forecast object
- assessment confidence / uncertainty

P5A does **not** introduce a second domain representation for the same assessment.

The older `FusionResult` / `FusionDetailScreen` code may inform presentation, but it is not an authority for P5A and must not be mixed into the new screens to fill gaps in `AssessmentSummary`.

## 4. Contract Honesty Rules

P5A follows these hard rules:

- Show only backend-provided contribution values.
- Do not reconstruct contribution from score, confidence, or any other field.
- Do not reconstruct or infer fusion weights.
- Do not calculate a risk tier, current score, forecast score, or fusion result locally.
- Do not convert missing values to zero.
- Do not convert missing/stale/unavailable data to low risk.
- Do not infer an exclusion reason that the backend did not report.
- Do not infer causality from a contribution value.
- Do not use legacy/local alert logic to decorate these screens.
- Do not call TC-WPN directly.
- Do not guess new backend routes.

If a value required by the target design is not present in the verified current contract, render a neutral state such as `Not reported` rather than fabricating it.

## 5. Screen A — Signals & Contributions

### 5.1 Purpose

Answer:

> What signals formed this assessment, and what contribution information did the backend report for them?

### 5.2 Header / provenance

Display:

- participant display identifier when available from navigation context;
- canonical `subject_id`;
- `fusion_result_id` when present;
- assessment computed time when present;
- model/fusion version when present;
- assessment status.

The page must make clear that it is explaining one specific backend assessment.

### 5.3 Modality order

Always display modalities in this canonical order:

1. C1 Physiological
2. C2 Behavioural
3. C3 Clinical NLP / TC-WPN
4. C4 Contextual

A missing modality record still gets a visible placeholder card so absence is explicit rather than silently omitted.

### 5.4 Per-modality content

Each modality card may show, when reported:

- signal label;
- score;
- availability/state;
- fusion inclusion/exclusion state;
- backend-provided contribution;
- captured time.

If a contribution is missing:

```text
Contribution: Not reported
```

Do not calculate a replacement.

### 5.5 C3 semantics

C3 must always be labelled:

```text
Clinical NLP / TC-WPN
```

It must remain visibly subordinate to the multimodal assessment.

Do not label it `Overall risk`, `Patient risk`, `TC-WPN Risk`, or anything that implies the C3 score is the patient's overall multimodal risk.

### 5.6 C2 semantics

When the backend contract reports:

- `state == notValidated`, and
- `includedInFusion == false`

render:

```text
Experimental — not included in fusion
```

Do not imply that C2 contributed to the composite when it was excluded.

### 5.7 Contribution interpretation notice

Show a short research-safe explanation such as:

> Contribution reflects the value reported by the fusion backend for this assessment. It does not establish that a modality caused the patient's state.

This is explanatory only and must not add unsupported causal claims.

## 6. Screen B — Data Quality

### 6.1 Purpose

Answer:

> Was the expected data present, usable, and appropriately represented when this assessment was produced?

### 6.2 Assessment-level summary

Display:

- assessment status: Complete / Partial / Unavailable / Unknown;
- `fusion_result_id` when present;
- computed time when present;
- model version when present.

Assessment status remains a server-derived/read-model value. P5A does not recalculate completeness from local heuristics.

### 6.3 Per-modality content

For each C1-C4 modality, display the server-reported values when available:

- Available / Unavailable / Unknown
- modality state
- confidence
- coverage
- captured time
- Included in fusion / Not included in fusion / Inclusion not reported
- score, when useful for context, without treating its absence as zero

### 6.4 Missing values

Render neutral explicit values:

```text
Score: —
Confidence: Not reported
Coverage: Not reported
Captured: Not reported
Reason: Not reported by backend
```

The UI must not invent `buffering`, `insufficient data`, `service unavailable`, or another reason unless the current verified contract actually provides that reason.

### 6.5 State semantics

Use the existing typed modality state and render it faithfully:

- `ok` -> Available, subject to inclusion field
- `stale` -> Stale
- `notValidated` -> Experimental / not validated
- `unavailable` -> Unavailable
- `error` -> Service error
- `unknown` -> Status unknown

If `includedInFusion == false`, render that independently from availability because a signal can be available yet excluded.

## 7. Patient Overview Integration

P5A adds explicit navigation affordances to Patient Overview.

### 7.1 Signals section

Below or after the existing C1-C4 signal summary, add:

```text
View signals & contributions ->
```

This opens `SignalsContributionsScreen` with the exact `AssessmentSummary` currently rendered by Patient Overview.

### 7.2 Data quality summary

Add a compact Data Quality summary after the signals section. It may show:

- assessment status;
- a non-authoritative presentation count such as number of modality records currently present, if phrased as UI inventory rather than a clinical completeness decision;
- a link:

```text
View data quality ->
```

The authoritative assessment status continues to come from `assessment.assessmentStatus`.

### 7.3 Navigation identity

The production navigation path must preserve:

- canonical `subject_id`;
- the same `AssessmentSummary` object;
- optional participant display label.

No MRN lookup, local roster lookup, or secondary assessment request is allowed solely to open these deep screens.

## 8. State Ownership

P5A screens are presentation-only views over immutable assessment data.

They do not need independent controllers for the initial implementation.

This means:

```text
PatientOverviewController
        |
        +--> AssessmentSummary
                |
                +--> Patient Overview
                +--> Signals & Contributions
                +--> Data Quality
```

The P5A screens therefore do not introduce repository dependencies or lifecycle mutations.

## 9. Error and Partial-Data Behavior

Because deep screens receive an already-loaded assessment, their main concern is partial/missing fields rather than network state.

Rules:

- Missing modality record -> explicit `Unavailable / not reported` card.
- Null score -> `—`, never `0`.
- Null confidence -> `Not reported`.
- Null coverage -> `Not reported`.
- Null captured time -> `Not reported`.
- Null contribution -> `Not reported`.
- Unknown typed state -> `Status unknown`.
- Unknown assessment status -> `Assessment status unknown`.
- Missing model version -> `Not reported` or omitted with no fabricated fallback.
- Missing fusion result ID -> state that no fusion record ID was reported; do not synthesize one.

These rules also apply to tests and fixtures.

## 10. Security / Authority Boundaries

P5A does not change authentication or authorization behavior.

It relies on Patient Overview already having received an assignment-scoped authoritative assessment from the central backend path.

P5A must not:

- broaden patient access;
- bypass assignment scoping;
- expose a hidden local cache as current truth;
- call component services directly;
- create or modify attention events;
- submit clinician judgement;
- make backend mutations.

## 11. Files Expected to Change

Primary implementation surface:

```text
lib/features/patients/patient_overview_screen.dart
lib/features/patients/signals_contributions_screen.dart
lib/features/patients/data_quality_screen.dart

test/features/patients/patient_overview_screen_test.dart
test/features/patients/signals_contributions_screen_test.dart
test/features/patients/data_quality_screen_test.dart
test/p5a_assessment_authority_test.dart
```

A small shared presentation helper may be extracted if required to avoid duplicating modality labels/status formatting, but P5A will not introduce a new assessment domain model.

## 12. Test-First Verification Plan

Implementation will follow TDD.

### 12.1 Signals & Contributions tests

Prove that:

- all C1-C4 cards render in canonical order;
- the same `fusion_result_id` appears as the source assessment;
- server score values render unchanged;
- backend contribution renders unchanged;
- missing contribution is `Not reported`;
- no weight is inferred;
- C3 is labelled `Clinical NLP / TC-WPN`;
- C3 is never labelled overall risk;
- C2 experimental/excluded state is explicit;
- stale/unavailable/error/unknown remain explicit;
- null score is not rendered as zero;
- provenance fields render only when reported.

### 12.2 Data Quality tests

Prove that:

- Complete / Partial / Unavailable / Unknown assessment states remain distinct;
- availability and fusion inclusion remain separate concepts;
- confidence/coverage/captured time show exact server values;
- missing confidence/coverage/time render `Not reported`;
- stale remains stale;
- unavailable remains unavailable;
- error remains service error;
- unknown remains unknown;
- missing reason is not fabricated;
- C2 experimental/excluded semantics remain explicit.

### 12.3 Patient Overview navigation tests

Prove that:

- both P5A navigation actions exist when an assessment is loaded;
- `SignalsContributionsScreen` receives the exact loaded `AssessmentSummary` object;
- `DataQualityScreen` receives the exact loaded `AssessmentSummary` object;
- subject identity is preserved;
- navigation does not trigger a second repository load merely to enter a deep view.

### 12.4 Authority regression test

Add static/structural checks ensuring P5A does not introduce:

- direct `/predict` calls;
- guessed `/contributions` or `/data-quality` routes;
- local fusion/risk/contribution arithmetic;
- legacy `FusionResult` as P5A screen authority;
- local attention-event creation;
- a local fallback that maps missing values to zero or low risk.

## 13. Acceptance Criteria

P5A is complete when all of the following are true:

1. Patient Overview has working links to two distinct deep screens.
2. Both screens are projections of the same loaded `AssessmentSummary`.
3. Signals & Contributions shows C1-C4 with backend values and backend contribution only.
4. Data Quality shows availability, typed state, confidence, coverage, capture time, and fusion inclusion honestly.
5. C2 experimental/excluded semantics are preserved.
6. C3 remains subordinate to the multimodal assessment.
7. Missing values never become zero, low risk, or fabricated reasons.
8. No new backend route is guessed.
9. No local fusion, contribution, or risk computation is added.
10. Focused P5A tests pass.
11. Existing P3/P4 authority tests remain green.
12. Flutter analyzer is clean for the implementation surface.
13. Any pre-existing unrelated full-suite baseline failure is reported explicitly rather than hidden.

## 14. Non-Goals

P5A does not implement:

- Timeline / 24-hour / 7-day history;
- Clinical Notes changes;
- CARE-AnxRAG / Supporting Evidence changes;
- Clinician Assessment changes;
- push notifications;
- P6 realtime behavior;
- server forecast changes;
- attention-event policy changes;
- assignment/auth backend changes;
- new backend contribution/data-quality endpoints;
- backend fusion/model logic changes.

Those belong to later roadmap phases or backend work.

## 15. Relationship to Later P5 Work

After P5A:

- **P5B** will handle Timeline + Clinical Notes.
- **P5C** will handle Supporting Evidence + Clinician Assessment.

P5A intentionally establishes a clean, canonical assessment context first so those later deep views can attach to the same patient/assessment identity without reopening the source-of-truth question.
