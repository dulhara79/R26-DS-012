# ClinAnx P5B — Timeline & Clinical Notes Design

**Status:** Approved by user in chat on 2026-09-16  
**Base branch:** `main`  
**Base commit:** `1be42a9c2260601a449b06ff1dca777a3b00a24f`  
**Implementation branch:** `integration/clinanx-p5b-timeline-clinical-notes`

## 1. Purpose

P5B adds two clinician-facing deep views from Patient Overview without creating any new clinical authority:

1. **Timeline** — historical temporal context from the Central Backend's verified clinician timeline route.
2. **Clinical Notes** — device-local draft/history management plus submission through the Central Backend clinical-note workflow.

The existing `AssessmentSummary` remains the authoritative current assessment. Timeline is historical context only. Clinical NLP / TC-WPN remains one contributing modality and must never be presented as the patient's overall multimodal risk.

## 2. Verified Current Backend Boundary

The current Central Backend exposes:

- `GET /v1/doctor/patients/{subject_id}/timeline?limit=N`
- `POST /v1/clinical-notes`

The timeline response contains a server-provided `trend` array with fields including `composite`, `tier`, `band`, `assessment_status`, `missing_modalities`, `computed_at`, and `trigger`.

The current backend does **not** expose a verified clinician clinical-note history read endpoint. Therefore P5B must not invent `GET /clinical-notes`, `/notes/history`, or any equivalent route.

The target specification wants fused-assessment and forecast history plus attention-event markers, but P5B renders only history fields actually present in the verified current contract. Forecast-history and attention-event-history remain target gaps until the backend exposes exact contracts.

## 3. Source-of-Truth Boundaries

### 3.1 Current assessment

`PatientOverviewController -> AssessmentSummary` remains authoritative for the current multimodal assessment and optional current forecast.

### 3.2 Timeline

Timeline uses a dedicated read model and repository boundary. It does not reuse legacy `FusionResult` as the current assessment authority and does not reconstruct values from local caches.

```text
Patient Overview
   |
   +--> current AssessmentSummary  (current authority)
   |
   +--> TimelineScreen
           |
           +--> TimelineRepository.history(subjectId)
                   |
                   +--> GET /v1/doctor/patients/{subject_id}/timeline
                           |
                           +--> server `trend` rows only
```

### 3.3 Clinical notes

The device-local note list is explicitly a local ClinAnx record until a server-authoritative note-history read contract exists.

```text
ClinAnx local draft/history
        |
        +--> clinician edits/saves locally
        |
        +--> POST /v1/clinical-notes
                |
                +--> Central Backend -> C3 / TC-WPN -> fusion policy
        |
        +--> refresh canonical Patient Overview assessment
```

A failed submission must not delete note text. A successful submission updates the local note state and then requests a canonical assessment refresh.

## 4. Timeline Contract

Introduce a small P5B-specific immutable history contract rather than expanding `AssessmentSummary` or reusing legacy `FusionResult`.

Each `TimelineEntry` contains only backend-reported fields:

- `double? composite`
- `String? tier`
- `String? band`
- `String? assessmentStatus`
- `List<String> missingModalities`
- `DateTime? computedAt`
- `String? trigger`

Rules:

- null composite stays null and renders `—` / unavailable, never `0`;
- missing timestamps stay unknown rather than `DateTime.now()`;
- no risk tier is derived from composite;
- no interpolation or synthetic points;
- no client-side forecast history;
- no client-side attention-event markers;
- 24-hour and 7-day views are filters over returned server rows only.

The repository requests enough recent rows for the view but does not pretend the backend guarantees continuous temporal coverage.

## 5. Timeline UI

`TimelineScreen` is opened from Patient Overview and receives canonical `subjectId` plus optional participant display label.

The screen provides:

- `24 hours` and `7 days` selectors;
- chronological history rows, newest/oldest ordering chosen consistently and tested;
- exact server composite, tier/band text, assessment status, timestamp, and trigger when reported;
- explicit empty state when no rows exist in the selected window;
- explicit partial-data state when row values are missing;
- a notice that gaps represent no returned assessment record, not low risk.

A simple list is preferred over a line chart for P5B because the current contract does not guarantee regular sampling. A chart that visually connects missing intervals would imply interpolation.

## 6. Clinical Notes Contract

P5B keeps the existing `ClinicalNote` persistence model for backward compatibility with already stored device records, but introduces a dedicated P5B controller/repository boundary so the new screen is not governed by legacy `ChartController` or `FusionResult` state.

The local notes repository supports:

- load local notes by local patient record key;
- save/update local notes;
- load effective support set from existing local storage;
- submit a note using the verified Central Backend `POST /v1/clinical-notes` path.

P5B does not introduce a server note-history reader.

## 7. Clinical Notes UI

The screen must clearly state that displayed history is device-local unless/until a backend note-history contract exists.

Capabilities:

- view local notes with recorded time, authorship, note type and lifecycle state;
- create a note;
- edit an existing local note;
- save as draft;
- submit/retry analysis through the backend;
- preserve the note on network/model failure;
- show returned C3 result as `Clinical NLP / TC-WPN signal`, not `TC-WPN Risk` or overall patient risk;
- show `Not reported` where the backend omitted a score/result;
- after a successful backend submission, call the Patient Overview canonical refresh callback/controller.

No local fusion, no local tier derivation from the TC-WPN score, and no fixed-weight wording such as “largest weight in the composite”.

## 8. Navigation Identity

`PatientsScreen` already resolves a canonical `subjectId` before opening Patient Overview. P5B also needs the local record key used by `RecordStore` so existing local drafts remain reachable.

Production navigation will therefore preserve both concepts explicitly:

- `subjectId` — canonical backend identity for Timeline and note submission;
- `localRecordId` — local ClinAnx persistence key for existing notes/support records.

These values must never be conflated. The backend is called with `subjectId`; `RecordStore` is called with `localRecordId`.

## 9. State Ownership

Add focused controllers rather than extending legacy `ChartController`:

- `TimelineController` owns Timeline loading/filter state.
- `ClinicalNotesController` owns local note list, draft mutations, submission state and backend result state.
- `PatientOverviewController` remains owner of current `AssessmentSummary`.

After successful note submission, `ClinicalNotesController` invokes an injected `Future<void> Function()` refresh callback owned by Patient Overview. It does not fetch or compute the current multimodal assessment itself.

## 10. Error / Partial Behavior

Timeline:

- 404 / no history -> empty/unavailable history state;
- offline/timeout/server -> explicit unavailable/offline state;
- missing composite -> `—`, never Low/0;
- missing timestamp -> `Time not reported`;
- unknown assessment status -> `Status not reported`.

Clinical Notes:

- local save succeeds independently of backend availability;
- backend failure keeps the text and marks the analysis attempt failed;
- a backend response without a C3 result remains an explicit `analysisFailed`/no-result state;
- successful backend submission refreshes canonical Patient Overview;
- note text is never written to diagnostic logs by P5B code.

## 11. Security / Authority Rules

P5B must not:

- call TC-WPN `/predict` directly;
- compute fusion, contribution, current risk, forecast or tier in Flutter;
- infer missing timeline periods as low risk;
- invent note-history or forecast-history endpoints;
- make `FusionResult` the P5B current-state authority;
- replace backend `subjectId` with local MRN/participant ID for network calls;
- create attention events locally;
- present a clinical-note score as overall patient risk.

## 12. Expected Files

Primary new files:

```text
lib/domain/contracts/timeline_entry.dart
lib/domain/repositories/timeline_repository.dart
lib/domain/repositories/clinical_notes_repository.dart
lib/data/repositories/p5b_repositories.dart
lib/state/timeline_controller.dart
lib/state/clinical_notes_controller.dart
lib/features/patients/timeline_screen.dart
lib/features/patients/clinical_notes_screen.dart
```

Primary modified files:

```text
lib/features/patients/patient_overview_screen.dart
lib/features/patients/patients_screen.dart
.github/workflows/p1_p2_dashboard_ci.yml
```

Tests:

```text
test/contracts/timeline_entry_test.dart
test/repositories/p5b_repositories_test.dart
test/timeline_controller_test.dart
test/clinical_notes_controller_test.dart
test/features/patients/timeline_screen_test.dart
test/features/patients/clinical_notes_screen_test.dart
test/features/patients/patient_overview_screen_test.dart
test/p5b_timeline_clinical_notes_authority_test.dart
```

## 13. TDD Acceptance Criteria

P5B is complete when fresh verification proves:

1. Timeline loads only the verified clinician timeline route and parses only server `trend` rows.
2. 24h/7d filters never synthesize/interpolate points.
3. Null/missing history values never become zero/low/current values.
4. Patient Overview exposes working Timeline and Clinical Notes links.
5. Canonical `subjectId` and local record identity stay distinct.
6. Local notes remain available and clearly labelled device-local.
7. Draft creation/editing and failure preservation work.
8. Submission uses the Central Backend note route, never TC-WPN directly.
9. C3 is labelled Clinical NLP / TC-WPN signal and never overall patient risk.
10. Successful submission triggers canonical Patient Overview refresh.
11. No unverified note-history, forecast-history or event-history endpoint is introduced.
12. P0-P5A authority tests remain green.
13. Focused analyzer is clean.
14. Full-suite status is reported truthfully, including any pre-existing unrelated failure.

## 14. Non-Goals

P5B does not implement:

- server-side clinical-note history persistence/read API;
- forecast-history contract changes;
- attention-event markers in Timeline unless the backend later supplies them;
- CARE-AnxRAG / Supporting Evidence changes;
- Clinician Assessment;
- push notifications/realtime delivery;
- backend assignment/auth redesign;
- backend model/fusion logic changes.

Those remain backend work or later ClinAnx phases.
