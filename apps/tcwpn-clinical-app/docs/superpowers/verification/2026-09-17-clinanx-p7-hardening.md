# ClinAnx Phase 7 Hardening Verification

**Verification date:** 17 Sep 2026 (Sri Lanka project date)

## Branch and base

- Repository: `dulhara79/tcwpn_mobile_app`
- Branch: `integration/clinanx-p7-hardening`
- Base `main`: `fc76fcea3c3276a398f5f4baefeb3cc735991c32`
- Verified implementation head before this evidence-only commit: `f098820887f6c3e0d26f016702289c0a9bdd580c`
- GitHub Actions run: `35138940628`
- Flutter used by CI: stable `3.47.4`

`main` was rechecked before finalization and remained at the same base SHA; no main-branch drift required reconciliation.

## Verification evidence

The branch workflow made focused checks, authority/privacy checks, the full test suite, and full static analysis blocking.

Run `35138940628` completed successfully with:

- Focused Phase 0-7 + repository/controller/widget suite: **275 tests passed**.
- Focused `flutter analyze`: **No issues found**.
- Authority, privacy and forbidden-pattern checks: **passed**.
- Full `flutter test`: **307 tests passed**.
- Full `flutter analyze`: **No issues found**.

No local Flutter execution was used as completion evidence. The active local container could not clone GitHub / did not provide the project Flutter workspace; verification was performed by GitHub Actions on the exact branch head stated above.

## Phase 7 requirements checked

### Local privacy / session isolation

Verified by automated tests:

- DR001 RecordStore data is not returned under DR002.
- Server dashboard cache is isolated by clinician.
- Pending notification-open state is isolated by clinician.
- Legacy unscoped roster key is not used as the new authority.
- Clinical roster initialization moved from the root pre-auth app into the authenticated shell provider lifecycle.

### Offline / stale / failure safety

Existing and Phase 7 tests verify:

- stale C1 remains stale and is not fused as current evidence;
- unavailable C3 remains null, never zero;
- missing/unavailable modalities do not fabricate Low/Green/current score;
- network loss is offline rather than an empty safe result;
- 401 maps to session expiry;
- 403 remains forbidden without clearing a valid session;
- 409 remains a canonical-state conflict and event mutation reloads server state;
- malformed responses fail explicitly;
- RAG unavailable/timeout and RAG abstention remain distinct and do not fabricate guidance;
- duplicate notification delivery is deduplicated per event identity.

### Accessibility

- Removed the global `MediaQuery.withClampedTextScaling(... maxScaleFactor: 1.3)` cap.
- Source/feature tests preserve textual status labels in critical assessment and AttentionEvent views so color is not the sole state carrier.

This is automated accessibility hardening evidence, not a claim that a human accessibility/usability study has already been completed. A manual checklist is provided at `docs/qa/phase7_usability_checklist.md`.

### Configuration / security truthfulness

- `AuthService` consumes centralized `Env` auth configuration rather than declaring a second set of build-time auth defines.
- `DEMO_DATA` now defaults to `false`; demo fixtures require explicit opt-in.
- Settings states that clinical-note text goes to the Central Backend, which orchestrates Clinical NLP / TC-WPN.
- Settings distinguishes explicitly pinned hosts from other HTTPS hosts using platform TLS.
- No new certificate pins were fabricated.

### CI release gate

- Phase 7 tests are in the focused suite.
- Full `flutter test` is blocking.
- Full `flutter analyze` is blocking.
- Existing clinical-authority guards remain blocking.
- New privacy, demo-default, and text-scaling guards are blocking.

## External dependencies intentionally not claimed as complete

This Phase 7 verification proves ClinAnx client behavior and fixture/repository contracts. It does **not** claim live whole-system target architecture is complete.

At Phase 7 start, the external Central Backend still used the shared `BACKEND_API_TOKEN` model and did not expose the full target clinician-principal/assignment/latest-assessment/persistent-AttentionEvent implementation. The patient app also still required alignment to the same authoritative FusionResult/event episode. Those repositories were inspected read-only and were not modified.

Therefore the following still require live cross-repository verification after teammate changes land:

- clinician JWT/principal + server-side assignment enforcement;
- live assignment-scoped Dashboard/latest-assessment transport;
- live persistent AttentionEvent list/detail/ACK/RESOLVE against the frozen Phase 6 contract;
- patient and clinician retrieval proving the same `fusion_result_id` for the same assessment;
- full patient + clinician shared event episode E2E;
- multi-clinician server concurrency with durable persistence.

## Manual evidence still required before research release

Use `docs/qa/phase7_usability_checklist.md` with synthetic/demo participants. Record defects and retest results. Fixture tests must not be presented as a completed human usability study or live backend E2E.