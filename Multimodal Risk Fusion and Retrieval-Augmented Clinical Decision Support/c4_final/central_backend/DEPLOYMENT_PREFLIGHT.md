# Central Backend deployment and clinician login preflight

Build **from the repository root** to include in-process fusion modules,
numpy and frozen reference distributions.

```sh
docker build -f central_backend/Dockerfile -t r26ds012-central:staging .
# Provide DATABASE_URL and backend-only secrets using a protected environment.
# SQLite is for development only: use a persistent mounted path if applicable.
# Example DATABASE_URL=sqlite:////data/central_backend.db
docker run --rm --env-file /private/central.env -v central-db:/data \
  r26ds012-central:staging python migrate_p0.py
docker run --rm -it --env-file /private/central.env -v central-db:/data \
  r26ds012-central:staging python seed_clinician.py DR001 "Dr X"
docker run -d --name r26-central --env-file /private/central.env \
  -v central-db:/data -p 8000:8000 r26ds012-central:staging
```

For PostgreSQL, use the actual persistent study database and run the migration
as one controlled job *before* starting new replicas. Do not run migrations in
each backend container: the migration tool has no distributed DDL lock.
Back up the database and retain the previous image before updating.

## Diagnose clinician login with synthetic credentials

- `GET /ready` HTTP 503: configuration, schema migration or database setup
  is not ready. `GET /health` is liveness only.
- `POST /auth/login` HTTP 401: clinician ID not found, disabled, or password
  incorrect. IDs are case-sensitive; verify the account in the deployed DB.
- HTTP 403: authenticated account role is not permitted.
- HTTP 200 on login and on authenticated `GET /v1/me`: backend login works.
  Verify the ClinAnx APK's `BACKEND_BASE` and installed build revision.

The clinician account is never populated from `AUTH_LOCAL`. An operator must
run `seed_clinician.py` using the *same* `DATABASE_URL` as the API.

Use HTTPS externally and configure **distributed rate limiting** at the trusted
ingress for `/auth/login`. A per-process counter is not sufficient with
multiple workers/replicas. Do not log raw passwords, JWTs, private DB URLs,
participant data or password hashes. Use only synthetic test identities.

## Optional phase-7 notifications

Revision `0003_push_registry_outbox` is required for this build's `/ready`.
See [PUSH_DELIVERY.md](PUSH_DELIVERY.md). Push is disabled by default; do not
claim device delivery until Firebase credentials, both app token registrations,
worker scheduling and physical-device tests succeed.
