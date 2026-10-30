# R26-DS-012 attention notification delivery (handbook phase 7)

This branch provides authenticated `POST /v1/device-tokens`, `GET /v1/device-tokens`
and `DELETE /v1/device-tokens/{device_id}`, encrypted FCM tokens, and a
persistent outbox generated in the **same transaction** as the server event.

## Deployment safeguards

1. Apply schema revision `0003_push_registry_outbox` on staging with the
   existing migration runner. Back up the database first.
2. Provision one shared, protected `DEVICE_TOKEN_ENCRYPTION_KEY` (Fernet
   symmetric key); never put it in source or mobile builds.
3. Set `FCM_PROJECT_ID` and workload identity / Google ADC for the service
   running the worker. Enable `ENABLE_PUSH_NOTIFICATIONS=1` only after setup.
4. Schedule `python notification_worker.py` as a separate job; the backend
   HTTP server does not own background worker scheduling. Multiple workers
   use PostgreSQL `SKIP LOCKED` and renewable delivery claims.
5. Update **both** Android apps to register their FCM token after authenticated
   login, revoke it on logout and update it on token refresh. A backend-only
   deployment cannot cause mobile push notifications to arrive.
6. Verify an assigned clinician and enrolled patient receive only
   `type`, `event_id`, `severity`, `environment` in push data.
   On notification open, fetch event detail through the authenticated API.
7. Repeat synthetic duplicate-delivery, revoked token, multiple-clinician,
   replay, expired-token, worker retry and app-background/terminated tests on
   a staging Firebase project before release.

Push is disabled by default. Existing foreground polling remains the
P0 fallback. No claim of functioning device push is made until staging
credentials and app changes are verified.

For request rate limiting, apply `deploy/nginx_rate_limits.conf.example` at
the *actual trusted HTTPS ingress*, and verify HTTP 429 on abusive traffic.
The API itself is not a distributed rate limiter.
