# Phase 5 - ClinAnx Central Backend Integration

## Scope

This change integrates the clinician-facing P0 workflow with the verified Phase 1-4
Central Backend contracts. It does not change Fusion, C4/DCAR, TC-WPN scientific
logic, CARE-AnxRAG scientific logic, the Patient App, or notification
infrastructure.

## CURRENT verified implementation before this branch

- The Central Backend `main` already exposes authenticated `GET /v1/me`.
- `GET /v1/clinicians/me/patients` is clinician-assignment scoped.
- `GET /v1/patients/{subject_id}/assessment/latest` returns the canonical
  persisted `AssessmentSummary` backed by one authoritative `FusionResult`.
- `GET /v1/attention-events`, event detail, acknowledge and resolve are
  assignment scoped and server authoritative.
- ClinAnx already had typed assessment/event contracts, a fusion-first Patient
  Overview, server AttentionEvent views/actions, explicit unavailable states,
  C3 wording as Clinical NLP / TC-WPN, and C2 experimental/excluded wording.
- This branch removes the former production contract gates now that the
  Central Backend routes and response semantics are frozen and implemented.
- The Patients tab still used a clinician-local roster and legacy FusionResult
  presentation rather than the server assignment roster.

## TARGET

The System Integration Handbook and ClinAnx specification require ClinAnx to be
an assignment-aware presentation client over Central Backend state:

- Central Backend is authoritative for clinician identity, assignments,
  current assessment, forecast persistence and AttentionEvent lifecycle.
- Fusion is the primary current assessment.
- C3 / TC-WPN is a contributing Clinical NLP signal, not overall patient risk.
- C2 remains Experimental / not included in fusion.
- Current assessment and near-term forecast are separate concepts.
- Missing, stale or unavailable data must not become Low/Green.
- A C1-led future forecast must remain labelled physiological unless a
  separately specified and validated multimodal forecasting method exists.
- AttentionEvent state is server state. Local cache/notifications are only
  projections or delivery mechanisms.
- Assignment enforcement is server-side; Flutter hiding a row is not
  authorization.

## IMPLEMENTED

### Verified API adapters

- `CentralBackendAuthRepository` validates the active token with `GET /v1/me`
  and fails closed when the principal is not a clinician or conflicts with the
  local session identity.
- `CentralBackendAssessmentRepository` loads
  `/v1/patients/{subject_id}/assessment/latest` and preserves
  `fusion_result_id`, forecast identity, timestamps and model provenance.
- A canonical 200 response with `assessment_status=unavailable` and null
  assessment identity is represented as unavailable; no Low score is fabricated.
- `CentralBackendPatientRepository` consumes only
  `/v1/clinicians/me/patients`. It does not discover or resolve arbitrary
  patient identifiers.
- `CentralBackendDashboardRepository` consumes the frozen
  `/v1/clinicians/me/dashboard` aggregate directly, including clinician,
  authoritative assigned count, OPEN AttentionEvents and patient summaries.
- `CentralBackendPatientRepository` consumes the summary rows returned by the
  assignment roster directly; it does not perform per-patient N+1 requests.
- Existing AttentionEvent ACK/RESOLVE calls continue to send the frozen empty
  body `{}`; actor and timestamp remain server-derived.

### ClinAnx workflow

- Dashboard continues to show Needs Attention before assigned patients.
- Patients now renders the same assignment-scoped server roster used by the
  dashboard rather than the legacy local roster.
- Patient rows show the server current multimodal assessment and the forecast
  separately, preserve `fusion_result_id`, and show unavailable explicitly.
- Search/filtering only narrows the already assigned server list. It cannot
  widen authorization.
- Patient Overview remains fusion-first with forecast separate from current
  assessment and C3 subordinate under Signals.
- Activity and Attention Event Detail continue to render the persistent server
  lifecycle and reconcile server conflicts.

### Contract alignment

The mobile modality parser now recognizes the verified frozen backend status
vocabulary:

- `poor_signal` -> stale presentation
- `warming_up`, `insufficient_data`, `no_support_set` -> unavailable
- `not_validated` -> experimental/not validated
- unknown values remain unknown and do not become OK

The attention severity parser accepts the Phase 4 policy outputs
`elevated` and `high` without deriving severity locally.

## DEFERRED

- Patient App integration is Phase 6 and is not changed here.
- Device-token registry, FCM/APNs and realtime/push design are later-phase work.
  Pre-existing notification/polling code is preserved but not expanded here.
- No new Fusion, C4/DCAR, CARE-AnxRAG or TC-WPN scientific logic is introduced.
- No client-side multimodal forecast is introduced.
- No production JWT issuer, audience, algorithm, key or credential value is
  invented.

## KNOWN LIMITATIONS / DEPLOYMENT DEPENDENCIES

1. A real ClinAnx session token must be verifiable by the Central Backend's
   configured JWT verifier. Demo/local authentication tokens are not a
   substitute for a production/research clinician JWT.
2. Server-side clinical-note history remains a separate contract issue. The
   authoritative note-analysis submission stays Central Backend -> C3, followed
   by canonical assessment refresh; local note draft/history must not be
   interpreted as model authority.
3. Existing push/polling code predates this Phase 5 branch. Server
   AttentionEvent remains authoritative regardless of notification delivery.

## Security / authority guarantees preserved

- no privileged shared backend token is added to the app;
- no client-provided lifecycle actor or timestamp;
- no client Fusion calculation;
- no local red-band urgent-event creation;
- no C3-as-overall-risk presentation;
- no C2 promotion into Fusion;
- no unassigned subject discovery path in the Patients screen;
- no silent missing/stale -> Low conversion;
- no fabricated RAG fallback.
