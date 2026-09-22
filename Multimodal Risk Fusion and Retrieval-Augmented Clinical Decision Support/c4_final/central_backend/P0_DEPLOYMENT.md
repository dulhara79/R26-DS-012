# ClinAnx P0 deployment

The API contract is published by the running service at `/docs` and `/openapi.json`.
There is intentionally no `/v1/subjects/attach`; clients use
`/v1/subjects/resolve?app_user_id=...`.

1. Back up the database and retain the previous application image.
2. Set `DATABASE_URL`, `BACKEND_API_TOKEN`, `CLINICIAN_JWT_SECRET`,
   `CLINICIAN_JWT_ISSUER`, and `CLINICIAN_JWT_AUDIENCE`. Secrets belong only on
   the server.
3. Run `python -m central_backend.migrate_p0` from the repository root with
   `PYTHONPATH=central_backend`.
4. Seed a clinician with `python central_backend/seed_clinician.py DR001 "Dr X"`.
5. Assign subjects using `python central_backend/manage_assignment.py assign DR001 <subject_id>`.
6. Start the service and verify `/health`, `/openapi.json`, login, dashboard,
   assessment, and event lifecycle with an assigned test subject.

Deploy the database additions before the application. Roll back by restoring the
database backup and previous application image; do not drop P0 tables in-place.
