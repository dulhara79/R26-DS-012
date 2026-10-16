# Handbook Phase 8 Hardening Implementation Plan

**Handbook scope:** Offline/stale behavior, security review, failure injection, usability, and reproducible configuration.

**Branch:** `integration/clinanx-handbook-p8-hardening`

## Constraints

- ClinAnx follows the handbook contract even if the Central Backend implementation lands later.
- Central Backend remains the only clinical authority; mobile never computes authoritative fusion or alert state.
- No reusable privileged backend/service token may be compiled into the mobile release.
- Central Backend clinician calls use the authenticated clinician `Session` bearer.
- Non-local clinical endpoints must use HTTPS.
- Push payload consumption remains routing-only (`type=attention_event`, `event_id`); no PHI/model detail is trusted from push.
- Push failure must never lose an AttentionEvent; server persistence + polling remains the fallback.
- Missing/stale/unavailable data never becomes Low/Green/0.
- Do not fabricate completion of human usability or live backend security tests.

## Task 1 — Mobile auth/security boundary

1. Add regression tests proving `lib/` contains no `BACKEND_TOKEN` / `Env.backendToken` authority.
2. Remove the shared backend token build define and production overrides.
3. Make Central Backend repositories/gateway use `ApiClient` default clinician session bearer.
4. Add endpoint-policy tests: HTTPS required for non-local clinical endpoints; localhost HTTP allowed only for development/testing.

## Task 2 — Push/failure hardening

1. Add tests for malformed/unknown push payloads, duplicate event IDs, token registration failure, token rotation cleanup failure, and sign-out revocation failure.
2. Keep push registration best-effort so clinical access is not blocked when FCM or `/v1/device-tokens` is unavailable.
3. Keep polling as independent fallback and preserve event-ID deduplication.

## Task 3 — Reproducible configuration

1. Add a typed build/config summary that exposes non-secret research-build identity: app version, backend host, auth mode, active Firebase slot/project ID, demo-data state.
2. Surface that summary in Settings without exposing tokens/API secrets.
3. Add tests ensuring demo data is opt-in, Firebase slot is primary/secondary only, and secrets are not rendered.

## Task 4 — Usability/accessibility evidence

1. Preserve the existing accessibility guards (no global text-scale cap; textual status labels).
2. Add/update the Phase 8 manual usability checklist to include push/fallback and configuration-state checks.
3. Leave manual items unchecked until a human run is actually performed.

## Task 5 — CI/release gate

1. Add Phase 8 tests to the focused workflow.
2. Add blocking forbidden-pattern checks for shared backend token usage and unsafe clinical HTTP configuration patterns.
3. Run focused tests, full `flutter test`, and full `flutter analyze` in PR CI.
4. Create PR to `main` only after implementation is ready; do not add the feature branch to push triggers.

## Completion evidence

Phase 8 mobile code is complete only when PR CI is green and the code-level handbook requirements above are satisfied. Human usability and live backend authorization tests remain separately identified evidence until actually executed.
