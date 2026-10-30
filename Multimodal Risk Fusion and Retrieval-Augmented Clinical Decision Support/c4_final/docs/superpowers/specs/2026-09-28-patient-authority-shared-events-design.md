# Patient Authority and Shared Attention Events Design

## Intent

Close the remaining R26-DS-012 P0 gaps identified by the System Integration
Handbook and 14-Day Sprint Plan: patient-scoped authorization, removal of the
shared privileged mobile backend token, patient consumption of the same
persistent `AttentionEvent` used by ClinAnx, and optional resolution notes.

## Invariants

- One canonical `subject_id` remains the identity source.
- A patient credential can authorize only its bound `subject_id`.
- Mobile applications never receive or embed `BACKEND_API_TOKEN`.
- The Central Backend is the only authority that creates escalation events.
- Aura and ClinAnx refer to the same persistent `event_id`.
- Current assessment and physiological forecast remain separate.
- Missing data remains unavailable, never zero or Low.
- Existing clinician routes and `POST /v1/clinical-notes` remain compatible.

## Patient session design

Aura generates a high-entropy installation secret and stores it in secure
storage. `POST /v1/subjects/self` accepts the existing `app_user_id` plus that
secret. On first enrolment the backend stores only a PBKDF2 hash bound to the
canonical subject. Repeated enrolment is idempotent only when the same secret
is presented; an identifier without its proof cannot mint a patient session.

The endpoint returns a short-lived HS256 patient JWT containing `sub`,
`subject_id`, `role=patient`, `iss`, `aud`, `iat`, `exp`, and `jti`. Patient JWT
settings are separate from clinician JWT settings. Aura stores the JWT in
secure storage and renews it through the proof-bearing self-enrolment flow.

Patient-originated risk and ingestion requests send the patient JWT. The
backend resolves request aliases and rejects any request whose resolved subject
differs from the patient principal. Service credentials remain supported for
trusted service-to-service callers but are removed from mobile Dart.

## Patient event projection

`GET /v1/patients/me/attention-events?status=OPEN` returns a privacy-minimized
projection of events for the authenticated patient. It includes the canonical
event identifier, lifecycle status, event type, severity, forecast horizon,
creation time, and policy version. It excludes clinician identifiers, clinical
notes, and unrestricted subject lookup.

Aura polls this endpoint while active. Each previously unseen server event is
stored locally for patient feedback and displayed as a local OS notification
using the exact server `event_id`. Notification dismissal never changes server
state. The existing direct C1 forecast path may display forecast information,
but it no longer creates an authoritative local escalation event.

## Resolution note compatibility

The clinician resolve request accepts `{}` or `{ "note": "..." }`. The server
stores the optional note atomically with `resolved_at` and `resolved_by` and
returns it as `resolution_note`. Empty-body ClinAnx clients remain compatible.
ClinAnx adds an optional note prompt before resolving.

## Failure semantics

- Missing, expired, malformed, wrong-audience, or wrong-signature patient JWT:
  `401`.
- Authenticated patient requesting another subject: `403`.
- Wrong installation secret during session renewal: `403`.
- Missing event or subject: `404`.
- Invalid payload: `422`.
- Event lifecycle conflict: existing `409` behavior.

## Verification

- Backend tests prove proof-bound enrolment, JWT validation, self-only risk and
  ingestion access, patient event projection, shared event identity, and
  optional resolution-note persistence.
- Aura tests prove secure session parsing/storage, authenticated request
  headers, server-event polling/deduplication, same event ID in notification
  payloads, and removal of local escalation authority.
- ClinAnx tests prove empty and note-bearing resolve requests and display of the
  canonical persisted note.
- Existing backend and Flutter suites remain green in GitHub Actions.
