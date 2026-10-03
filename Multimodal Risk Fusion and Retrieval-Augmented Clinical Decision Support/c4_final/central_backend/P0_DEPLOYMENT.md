# ClinAnx P0 deployment

The API contract is published by the running service at `/docs` and `/openapi.json`.
There is intentionally no `/v1/subjects/attach`; clients use
`/v1/subjects/resolve?app_user_id=...`.

1. Back up the database and retain the previous application image.
2. Set `DATABASE_URL`, `MRN_PEPPER`, `BACKEND_API_TOKEN`,
   `CLINICIAN_JWT_SECRET`, `CLINICIAN_JWT_ISSUER`,
   `CLINICIAN_JWT_AUDIENCE`, `PATIENT_JWT_SECRET`, `PATIENT_JWT_ISSUER`,
   and `PATIENT_JWT_AUDIENCE`. Keep each random secret server-only and distinct.
   Supply the actual reviewed C1/C3/C4 and RAG endpoints, not placeholders.
3. Before starting the new application image, run
   `PYTHONPATH=central_backend python -m central_backend.migrate_p0` from the
   repository root. It records schema revisions `0001_p0_tables` and
   `0002_assignment_invites_and_forecast_link`, adds `patient_credentials`,
   assignment invites, and the forecast-to-fusion link, and preserves existing
   patient rows. Run it twice on a staging copy to verify idempotence. Run only
   one migration process at a time; this revision runner has no distributed DDL
   lock. The `postgres-contract` CI job checks a disposable PostgreSQL 16
   upgrade, concurrent event transitions and episode creation, and a logical
   dump/restore. Repeat the upgrade and restore with the actual previous
   staging schema and deployment database settings before release.
4. Seed a clinician with `python central_backend/seed_clinician.py DR001 "Dr X"`.
5. A clinician may enrol a new subject and receives the initial assignment.
   Patient self-enrolment of an **existing** subject without a patient credential
   (including migrated records) now requires `pairing_code` in the
   `POST /v1/subjects/self` JSON beside `app_user_id` and
   `installation_secret`. Obtain the one-use code from an assigned clinician's
   `POST /v1/subjects` enrolment. The same installation secret can then renew
   sessions without another code. A known ID alone must never claim a record.
   The old `/v1/subjects/pair` can attach an alias but does not issue a patient
   JWT or substitute for this first-credential proof. Clinician-first mobile
   enrolment must supply the code to `/self` before using patient-only endpoints.
   For an existing patient-first subject, the authenticated patient creates a
   short-lived single-use invite with `POST /v1/patients/me/assignment-invites`
   and gives only its `invite_code` to the clinician. The clinician submits
   `POST /v1/clinicians/me/assignments` with `{"invite_code":"..."}`; the
   response is `{"clinician_id":"DR001","subject_id":"...","active":true}`.
   Unknown/used/expired codes return 404/409/410. Both Aura and ClinAnx now
   use this patient-issued invite flow. For an administrative recovery case,
   an operator can assign using
   `python central_backend/manage_assignment.py assign DR001 <subject_id>`.
   Knowing an MRN or app ID alone does not grant access to an existing subject.
6. Start the service and verify `/health` (liveness), `/ready` (database,
   migration and authentication configuration), `/openapi.json`, login, dashboard,
   assessment, and event lifecycle with an assigned test subject.
7. Run the local synthetic API rehearsal and the staging/device acceptance steps
   in [SYNTHETIC_REHEARSAL.md](SYNTHETIC_REHEARSAL.md). A passing local test
   exercises deterministic component doubles; it does not verify live Spaces,
   PostgreSQL concurrency, Android builds, or device notification delivery.

Clinician JWTs now authorize the retained doctor timeline/evidence/explanation,
subject resolve/external-ID, manual fusion, and verdict routes with assignment
checks. Internal service tokens remain supported for existing ingestion and
`POST /v1/clinical-notes -> C3 -> fusion`; missing service-token configuration
never permits anonymous access. The patient risk response adds the backend tier
and a separate, time-limited physiological forecast. No patient clinical-note
content or per-modality score is included in that projection.

The C1 escalation policy `escalation-v1` requires ten valid forecast points,
two predictions 20–120 seconds apart **from distinct C1 source-stamped windows**,
a source fusion result for event creation,
recovery below 0.40, and a ten-minute re-arm cooldown. Event audit records
capture create/ACK/RESOLVE identifiers and actors without copying note text.
Recovery still closes an episode if the forecast array is unavailable. Newly
computed fusion rows preserve their source reading IDs; legacy pre-upgrade rows
without this snapshot retain best-effort historical modality evidence.

Polling remains the P0 notification delivery mode. `/v1/device-tokens` and
FCM delivery are **not** provided by this deployment; disable best-effort device
registration in the clinician app until a separate push contract is agreed.

Deploy the database additions before the application. Roll back by restoring the
database backup and previous application image; do not drop P0 tables in-place.
