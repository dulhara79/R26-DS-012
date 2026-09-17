# ClinAnx — architecture

ClinAnx is the clinician-facing Flutter client for R26-DS-012. It is a **presentation, interaction and clinician-workflow layer**. It does not own model orchestration, multimodal fusion, forecast policy or the authoritative AttentionEvent lifecycle.

> **Research prototype — not a diagnostic device.** This architecture describes a research implementation boundary, not clinical deployment approval.

## 1. System boundary

```text
ClinAnx Flutter client
        │
        │ clinician session bearer over HTTPS
        ▼
Central Backend
        ├── identity / assignment scope
        ├── C1-C4 orchestration
        ├── current multimodal assessment
        ├── forecast persistence
        ├── AttentionEvent lifecycle
        └── CARE-AnxRAG evidence

Push path
Central Backend → FCM → ClinAnx
                     └→ event identity only

Recovery path
ClinAnx → server AttentionEvent polling every 30 s while resumed
```

The mobile app can warm the TC-WPN service with a non-clinical `/health` request when configured, but authoritative clinical-note inference is orchestrated through the Central Backend.

## 2. Mobile responsibilities

ClinAnx owns:

- clinician sign-in/session handling;
- secure local session storage;
- clinician-scoped local cache/drafts;
- Dashboard, Patients, Patient Overview and deep clinical views;
- safe presentation of current assessment versus forecast;
- server AttentionEvent Activity/detail UI;
- acknowledgement/resolution requests without inventing actor/time locally;
- generic notification presentation and push-open routing;
- device-token registration lifecycle;
- Primary/Secondary Firebase build-slot selection;
- research build identity and configuration visibility;
- offline/stale/unavailable presentation.

ClinAnx does **not** own:

- authoritative multimodal composite calculation;
- component model inference policy;
- assignment authorization rules;
- forecast generation or episode confirmation policy;
- creation/deduplication of authoritative AttentionEvents;
- server audit persistence;
- patient-facing notification policy;
- backend/service/model version assignment.

## 3. Layering

```text
Screen (lib/features/...)
        │
Controller / state (lib/state/...)
        │
Repository / gateway (lib/data/...)
        │
ApiClient + authenticated Session
        │
Central Backend
```

Rules:

- screens do not calculate authoritative risk;
- typed contracts preserve server vocabulary and null/unavailable states;
- network failures remain explicit errors/offline states, never synthetic clinical results;
- idempotent reads may be retried, but write/lifecycle authority remains server-canonical;
- participant/clinical traffic uses HTTPS outside loopback development.

## 4. Authentication and authorization boundary

Central clinical requests use the signed-in clinician session bearer. Reusable privileged service credentials are not part of the mobile authority model.

Session material is stored in platform secure storage. The mobile client can display `401` as expired/invalid session and `403` as forbidden assignment/access, but enforcement of clinician-to-patient assignment is a server responsibility.

The app never treats a clinician identifier supplied by the client as authoritative actor identity for AttentionEvent state changes. Canonical actor/time must come back from the server response.

## 5. Current assessment versus forecast

The app keeps these concepts separate:

- **Current multimodal assessment** — what the latest eligible multimodal evidence indicates now.
- **Near-term forecast** — future-horizon information. Until a validated multimodal forecast exists, ClinAnx labels the forecast scope as physiological when that is what the server contract represents.

A missing/stale/unavailable signal never becomes `0`, Low or Green. Unknown vocabulary fails closed to an explicit unknown/unavailable presentation.

C2 remains visible as an experimental behavioural signal and is not silently treated as active fused evidence while the registered exclusion rule remains in force.

C3/TC-WPN is displayed as a Clinical NLP signal, not the overall patient risk authority.

## 6. AttentionEvent lifecycle

The mobile contract treats AttentionEvents as persistent server records.

```text
OPEN
  │ acknowledge request
  ▼
ACKNOWLEDGED
  │ resolve request
  ▼
RESOLVED
```

ClinAnx never converts receipt of a notification into acknowledgement or resolution. Notification delivery and clinical lifecycle are separate concerns.

The client-facing event operations include the frozen AttentionEvent routes under `/v1/attention-events`, including list/detail and acknowledge/resolve operations. The backend owner is responsible for implementing the matching server contract and assignment enforcement.

## 7. Push, device tokens and polling fallback

Phase 7 added FCM-based push acceleration and device-token registration through the Central Backend contract.

Push routing payload accepted by ClinAnx is intentionally minimal:

```json
{
  "type": "attention_event",
  "event_id": "evt_..."
}
```

The parser uses a strict routing-field allowlist; extra fields are rejected. Clinical detail is fetched after authenticated open.

Two Firebase project configurations are supported as **build-time slots**:

- Primary — normal research build
- Secondary — disaster-recovery build

FCM registration tokens are project-specific, so this is not a runtime credential swap. If push initialization, permission or provider delivery fails, persistent server events remain discoverable through the independent 30-second foreground polling path.

## 8. Local persistence

Clinical local stores are clinician-scoped so one signed-in clinician does not inherit another clinician's cached roster/dashboard/notification state.

Cached information must retain provenance and be labelled stale/offline where appropriate. A cached assessment is never silently promoted to current state.

Local clinical-note drafts are a client workflow aid. Authoritative server analysis/history remains a backend concern.

## 9. Error semantics

Important mappings are deliberately distinct:

| Condition | Mobile behavior |
|---|---|
| `401` | session expired/invalid; require re-authentication |
| `403` | forbidden/assignment failure; do not imply expired identity |
| `404` | requested canonical resource unavailable |
| `409` | server-state conflict; refresh/reconcile canonical state |
| `422` | request validation failure |
| timeout/offline | explicit unavailable/offline state; preserve safe local work |
| RAG abstention | explicit abstained state |
| RAG unavailable | explicit service-unavailable state; no invented guidance |

## 10. Research build identity and reproducibility

Research/study builds expose non-secret configuration identity such as:

- app version;
- build environment;
- source/build revision;
- backend host;
- authentication mode;
- active Firebase slot/project identifier;
- demo-data state.

The committed `pubspec.lock` pins the Dart package graph. Phase 9 CI is pinned to Flutter 3.47.4, matching the SDK used to generate the recorded Phase 8/9 verification evidence.

Service/model versions not owned by this repository must be read from the integrated environment and recorded in the release manifest rather than guessed in mobile source.

## 11. Contract-gated backend integration

This repository intentionally distinguishes **implemented mobile contracts** from **verified running backend routes**. Some target adapters remain guarded/unavailable until the backend owner exposes the matching authenticated endpoints. ClinAnx must fail explicitly rather than silently substitute a local implementation.

That boundary is deliberate: this repository can prove mobile behavior and contract handling, but it cannot independently prove server assignment enforcement, cross-client result identity, server event persistence/concurrency or complete end-to-end delivery.

## 12. Release documentation

Phase 9 release material is maintained under `docs/release/`:

- `PHASE9_RESEARCH_RELEASE.md`
- `SOP.md`
- `DEPLOYMENT_RUNBOOK.md`
- `TEST_EVIDENCE.md`
- `KNOWN_LIMITATIONS.md`

`APK_BUILD.md` contains the reproducible mobile build commands. Manual usability evidence remains under `docs/qa/` and must not be marked complete without a real human run.
