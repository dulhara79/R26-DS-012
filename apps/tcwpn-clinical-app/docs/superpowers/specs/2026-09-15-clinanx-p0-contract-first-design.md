# ClinAnx P0 Contract-First Integration Design

Date: 2026-09-15
Branch: `integration/clinanx-p0-contract-first`
Repository: `dulhara79/tcwpn_mobile_app`
Status: Design approved in chat; implementation not yet started

## 1. Purpose

This design defines the first implementation slice required to move ClinAnx toward the target architecture in the project documents without guessing backend APIs or rewriting working functionality.

The immediate objective is to establish a stable, typed P0 contract boundary in ClinAnx that can be implemented against canonical JSON fixtures while the central backend owner implements the matching server contracts.

This is an integration sprint, not a rewrite.

## 2. Source basis

This design is based on:

- `ClinAnx_Clinician_Mobile_App_Technical_Architecture_and_UX_Specification(1).pdf`
- `R26-DS-012_System_Integration_Implementation_Handbook(1).pdf`
- current `main` of `dulhara79/tcwpn_mobile_app`
- current `main` of `UVINDUSEN/component4final`
- current `main` of `DewduSendanayake/anxiety_mobile_app`
- current `main` of `UVINDUSEN/Care-AnxRAG`

Verified repository snapshot used during design review:

- ClinAnx: `357f4e8078eb190769f5fced1894788f1f921264`
- Central backend: `a1ceef8daba268dac24d81aedd76c89e7c9ccc6a`
- Patient app: `2d7cb5936706c37ccbf462005b4e4b238bf3c1dd`
- CARE-AnxRAG: `fa83dd6a03e6478ac983e73dd07f5af0244f8874`

The handbook distinguishes CURRENT, TARGET and PROPOSED behavior. This design preserves that distinction.

## 3. Non-negotiable invariants

The implementation must preserve these rules:

1. One canonical `subject_id` identifies a participant across systems.
2. The central backend owns the authoritative current `FusionResult`.
3. Patient App and ClinAnx must resolve the same underlying `fusion_result_id` for the same assessment instance.
4. Current multimodal assessment and near-term forecast are different concepts and different fields.
5. Missing, stale, unavailable or insufficient data must never be converted into a reassuring low/zero state.
6. C2 remains experimental/excluded from active fusion until research governance changes that rule.
7. TC-WPN/C3 is a Clinical NLP contributing signal, not the overall patient-risk engine.
8. Urgent attention state is eventually server-owned through one persistent `AttentionEvent` lifecycle.
9. Backend authorization, not Flutter visibility, must enforce clinician-patient assignment.
10. ClinAnx local cache is non-authoritative and must be visibly stale/offline when used.

## 4. Verified current behavior to preserve

### 4.1 Clinical note flow

ClinAnx currently submits notes to the central backend via `POST /v1/clinical-notes` rather than using direct TC-WPN inference as the authoritative path.

The central backend then performs C3 analysis, persists the C3 reading and can trigger fusion. This is aligned with the target architecture and must not be replaced by a direct mobile-to-TC-WPN inference path.

### 4.2 Current fusion

The central backend already persists append-only `FusionResult` records and exposes a clinician timeline. ClinAnx already parses and uses `fusionResultId` in multiple paths, including clinician verdict submission.

This implementation must extend around that working path rather than introduce a second fusion implementation.

### 4.3 Patient current-risk projection

The patient app currently consumes `GET /v1/patients/{subject_id}/risk` for a reduced current-risk view and treats missing/GREY as unavailable rather than low. This behavior is directionally correct, although the target shared assessment contract needs to become richer and authenticated.

### 4.4 CARE-AnxRAG

CARE-AnxRAG already supports explicit abstention/failure semantics. P0 ClinAnx work should preserve the existing downstream evidence role and display unavailable/abstained states honestly rather than changing RAG internals.

## 5. Verified gaps that block the target architecture

The following target capabilities are not yet present as stable shared contracts:

- individual clinician principal accepted by the central backend
- `clinician_subject_assignments`
- assignment-scoped dashboard and roster APIs
- canonical latest `AssessmentSummary`
- persisted `ForecastResult`
- persistent `AttentionEvent`
- server-side event deduplication/episode handling
- event list/detail APIs
- atomic acknowledge/resolve lifecycle
- shared event delivery to patient and clinician apps

These are dependencies for final ClinAnx dashboard and attention workflows.

## 6. Existing contract mismatch to resolve

ClinAnx currently calls:

```text
POST /v1/subjects/attach
```

The inspected central backend does not expose this route.

The backend does expose canonical enrolment/pairing/resolve mechanisms, including:

```text
POST /v1/subjects/self
POST /v1/subjects
POST /v1/subjects/pair
GET  /v1/subjects/resolve
```

P0 must freeze one canonical identity flow before further coding. Preferred direction: remove the stale ClinAnx `/v1/subjects/attach` assumption and use the existing canonical alias/resolve flow unless the backend owner identifies a real identity case that requires a dedicated attach endpoint.

No route will be invented inside ClinAnx to hide this mismatch.

## 7. Scientific forecast boundary

The current assessment is multimodal fusion. The currently available near-term future signal is C1/physiological-led.

The P0 contract must therefore carry forecast scope explicitly, for example:

```json
{
  "scope": "physiological",
  "horizon_minutes": 10,
  "score": 0.84,
  "tier": "High",
  "escalation_predicted": true,
  "generated_at": "...",
  "valid_until": "..."
}
```

ClinAnx must not describe this as a validated multimodal forecast unless a later approved research method justifies that wording.

Recommended research-safe wording:

> Potential escalation predicted within the near-term forecast horizon.

The UI must not promise an anxiety attack at an exact future time.

## 8. P0 ClinAnx contract model

The first implementation slice will add typed models sufficient to parse the frozen target contract without requiring the final UI.

### 8.1 `AssessmentSummary`

Required semantics:

```text
subject_id
fusion_result_id
current_assessment
forecast
confidence / uncertainty
assessment_status
modalities[]
computed_at
model_version
```

### 8.2 `CurrentAssessment`

```text
score
tier
band
```

Score may be absent when assessment status is unavailable.

### 8.3 `ForecastResult`

```text
forecast_result_id?
scope
horizon_minutes
score?
tier?
escalation_probability?
escalation_predicted
generated_at
valid_until
```

### 8.4 `ModalityStatus`

```text
component_id
score?
available
included_in_fusion
status
confidence?
coverage?
captured_at?
contribution?
```

C2 must be representable as experimental/excluded with no active fusion contribution.

### 8.5 `PatientSummary`

Minimum dashboard/roster projection:

```text
subject_id
display_id
fusion_result_id?
current_assessment?
forecast?
assessment_status
last_updated
open_event_count
```

### 8.6 `AttentionEvent`

```text
id
subject_id
fusion_result_id?
forecast_result_id?
event_type
severity
reason?
forecast_horizon?
status
created_at
acknowledged_at?
acknowledged_by?
resolved_at?
resolved_by?
policy_version?
```

Supported lifecycle:

```text
OPEN -> ACKNOWLEDGED -> RESOLVED
```

Client code must treat server event state as authoritative once these APIs exist.

## 9. Typed enums

P0 will introduce explicit typed enums with safe unknown handling for at least:

- `RiskTier`
- `AssessmentStatus`
- `ForecastScope`
- `ModalityState`
- `AttentionEventStatus`
- `AttentionSeverity`

Unknown enum values must not crash the app or silently map to a reassuring clinical state.

## 10. Canonical JSON fixtures

ClinAnx will not wait for all backend endpoints before development. The contract will be exercised using frozen representative fixtures.

Required fixtures:

1. complete assessment
2. partial assessment
3. unavailable assessment
4. assessment with stale C1
5. assessment with C2 experimental/excluded
6. assessment with C3 unavailable
7. physiological escalation forecast
8. `AttentionEvent` OPEN
9. `AttentionEvent` ACKNOWLEDGED
10. `AttentionEvent` RESOLVED
11. unknown-enum compatibility case

Fixtures are contract artifacts, not demo guesses. They must stay aligned with the agreed central-backend schema.

## 11. Client boundary design

ClinAnx should continue moving toward one integration boundary:

```text
Flutter UI
  -> controllers/view state
  -> repositories/use cases
  -> CentralBackendApiClient
  -> central backend
```

Widgets should not know HTTP endpoint details or derive clinical thresholds.

The first P0 slice may adapt the existing project structure rather than perform a wholesale folder rewrite. Existing working auth, chart, note, fusion and evidence components should be preserved where possible.

## 12. Error and data-quality behavior

P0 parsing and state representation must support explicit states for:

- initial/loading
- partial assessment
- unavailable assessment
- C1 stale
- C2 experimental/excluded
- C3 unavailable
- backend unavailable
- RAG unavailable/abstained
- expired session
- unknown response enum
- action conflict / already-acknowledged event
- offline cached state

Rules:

- unavailable is not low
- stale cache is not current
- no fabricated score
- no fabricated evidence
- unknown server values remain explicit/unknown

## 13. Local alert transition strategy

The current ClinAnx implementation still creates local `ClinicalAlert` records from RED/DARK RED current fusion state.

This must not remain the final authoritative alert path.

However, the local path will not be deleted in the first contract-only slice because the server `AttentionEvent` subsystem does not yet exist. Deletion will occur only when the server event API is implemented and ClinAnx can consume persistent events safely.

Target state:

```text
server AttentionEvent
  -> ClinAnx event list/detail
  -> acknowledge/resolve on server
  -> Activity/history from server state
```

No final implementation should create a new urgent event purely because a Flutter widget sees a red color/band.

## 14. Backend dependencies owned outside this repository

The central backend owner must freeze/implement these P0 dependencies:

1. clinician principal handling
2. clinician table/identity mapping as required
3. `clinician_subject_assignments`
4. assignment-scoped dashboard/roster access
5. canonical latest `AssessmentSummary`
6. persisted `ForecastResult`
7. persistent `AttentionEvent`
8. deduplication/episode behavior
9. event list/detail
10. acknowledge/resolve with actor/timestamp
11. assignment enforcement for event/patient access
12. same underlying `fusion_result_id` projected to patient and clinician views

ClinAnx will not simulate these as if they already existed.

## 15. Patient-app dependency

The patient app should later migrate from an independent local alert authority to consumption of the same server-created event episode used by ClinAnx.

P0 cross-app acceptance condition:

```text
same subject
same assessment instance
same fusion_result_id
same server event_id for the escalation episode
```

Patient-specific details may remain reduced compared with the clinician projection.

## 16. First implementation slice in this repository

After this design is reviewed, the first code PR slice will contain only the P0 contract foundation:

- typed contract models/enums
- frozen JSON fixtures
- parser/contract tests
- gateway/repository interface additions required to consume the frozen contract
- explicit safe error/unavailable mapping
- resolution of the `/v1/subjects/attach` mismatch once the identity contract is confirmed

It will not yet implement the full dashboard, final patient overview, notifications or server attention lifecycle.

## 17. Explicit non-goals for the first slice

The first slice will not:

- change fusion mathematics
- change TC-WPN model behavior
- call TC-WPN directly for authoritative inference
- make C2 eligible for fusion
- build WebSocket/FCM infrastructure
- rewrite CARE-AnxRAG
- add a client-side authoritative fusion score
- claim a validated multimodal future forecast
- redesign the entire ClinAnx UI
- invent backend routes that have not been agreed
- modify teammate-owned repositories

## 18. Test strategy for the first slice

Minimum verification before the first implementation PR is considered complete:

- unit tests parse complete/partial/unavailable assessment fixtures
- unit tests parse OPEN/ACKNOWLEDGED/RESOLVED events
- unknown enum values degrade safely
- missing C3 is not parsed as zero
- C2 excluded state stays explicitly excluded
- stale C1 remains stale/unavailable according to the fixture
- forecast and current assessment remain separate domain objects
- `fusion_result_id` is preserved
- local model parsing does not compute an authoritative composite
- existing working ClinAnx tests continue to pass
- Flutter static analysis/build checks pass where available

## 19. Acceptance criteria for this design slice

The contract foundation is successful when:

1. ClinAnx can parse the same frozen `AssessmentSummary` that the central backend owner has agreed to implement.
2. ClinAnx has typed representation for current assessment, forecast, modality state and attention lifecycle.
3. No UI or parser conflates current assessment with forecast.
4. No missing/stale modality becomes an implicit low score.
5. `fusion_result_id` is preserved end-to-end in ClinAnx state.
6. C3 remains subordinate to the primary fusion assessment.
7. C2 remains explicitly experimental/excluded.
8. Event state is ready to become server-authoritative without inventing local server semantics.
9. The `/v1/subjects/attach` mismatch has an agreed resolution before final implementation relies on it.

## 20. Follow-on implementation order

Once P0 contracts are frozen and backend dependencies become available:

1. mobile foundation and repositories
2. assignment-scoped Dashboard + Patients
3. fusion-first Patient Overview
4. persistent Attention Event inbox/detail/actions
5. Activity from server events
6. deep clinical views: contributions, timeline, notes, evidence, data quality
7. patient app shared assessment/event migration
8. polling/local notification fallback, then optional push/realtime
9. failure injection and security hardening
10. research release verification

## 21. Coordination rule

One owner per code area.

- Dulhara: ClinAnx, C3 workflow, integration contract review
- Uvindu: central backend, persistence, assignments, fusion integration, attention engine/APIs
- Dewdu: patient app, C1 forecast path, patient assessment/notification behavior
- Senuvi: notification/data-quality integration support, contract/failure QA

Changes required in teammate-owned repositories should be communicated as explicit contract requirements rather than silently reimplemented in ClinAnx.
