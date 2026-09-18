# ClinAnx Known Limitations

This file records limitations that must remain visible for the R26-DS-012 ClinAnx research prototype.

> **Research prototype — not a diagnostic device.** These limitations are part of the release evidence, not optional footnotes.

## 1. Research / clinical-use boundary

ClinAnx is not a validated diagnostic device and must not be represented as one. It does not prove that an anxiety event will occur at an exact future time and should not be used as a substitute for clinician judgement.

## 2. Whole-system verification is outside this repository

This repository can verify mobile behavior and client contracts, but it cannot independently prove:

- Central Backend assignment enforcement;
- database durability/audit persistence;
- same `fusion_result_id` across patient and clinician projections in a live environment;
- server AttentionEvent persistence/concurrency;
- patient and clinician notification delivery of the same event;
- complete cross-repository end-to-end behavior.

Those require integrated backend/patient-app evidence.

## 3. Some target backend adapters remain contract-gated

The mobile code deliberately fails explicitly when a target backend contract has not been verified. It does not silently substitute local risk logic or guessed endpoints.

This means some screens/flows may remain unavailable until the backend owner exposes the matching authenticated contract defined by the handbook/mobile client.

## 4. Firebase Primary/Secondary is build-time disaster recovery

Two Firebase projects are supported as separately configured build slots.

This is **not** seamless runtime switching. FCM registration tokens are project-specific. A Secondary build must register against the Secondary project.

Both projects still depend on Firebase/FCM infrastructure, so two projects do not provide true independent push-provider redundancy.

## 5. Polling fallback has lifecycle limits

The current polling fallback runs every 30 seconds while ClinAnx is resumed/foregrounded and stops when the app is not active. It is a recovery path for missed/unavailable push while the app is in use; it is not a substitute for OS push waking a terminated/background app.

Persistent server AttentionEvents remain the source of truth.

## 6. Real-device push evidence depends on real configuration

Automated tests can verify token lifecycle, payload privacy, routing and deduplication, but they cannot prove real FCM delivery without actual Firebase project configuration and target devices.

Real Primary/Secondary project IDs and credentials must not be fabricated or committed merely to make tests look complete.

## 7. Android release packaging is not production-ready

The current Android project still uses:

- application ID `lk.sliit.r26ds012.clinanx`;
- debug signing for the `release` build type.

Therefore a generated release-mode APK is a research/development artifact, not a production-signed distribution package.

Before external study distribution, configure an approved unique application ID and protected release signing process. Signing material must remain outside source control.

## 8. iOS/APNs production delivery is not claimed

Flutter/iOS project files exist, but this Phase 9 mobile evidence does not claim a completed production APNs provisioning/distribution setup or physical-device iOS push validation. Any iOS study release needs its own signing/provisioning and push evidence.

## 9. Manual usability evidence is separate

Automated widget/unit tests do not substitute for the human checklist in `docs/qa/phase8_usability_checklist.md`.

Until a human executes and records the checklist, do not claim the manual usability portion of hardening/release readiness is complete.

## 10. Cached data can be stale

ClinAnx can retain clinician-scoped cached information for safe offline presentation. Cached data must remain clearly labelled with last-updated/offline/stale provenance and must never be presented as current authoritative state merely because it is available locally.

## 11. Current forecast scope must remain honest

A near-term C1 forecast is physiological unless a validated/persisted multimodal forecast contract explicitly says otherwise. The UI must not relabel a physiological forecast as a validated overall multimodal forecast.

## 12. C2 behavioural signal remains experimental/excluded

C2 may be displayed for research/data-quality purposes but remains excluded from active fusion under the current registered rule. A future scientific decision to include it requires a deliberate versioned methods/configuration change; the mobile app must not silently change this behavior.

## 13. C3/TC-WPN is not overall patient risk

TC-WPN represents the Clinical NLP signal derived from clinical-note evidence. Its score must not be presented as the entire multimodal patient-risk result.

## 14. CARE-AnxRAG availability is not guaranteed

Evidence generation may abstain, time out or be unavailable. ClinAnx deliberately shows those states rather than inventing advice.

## 15. Environment/service/model versions are externally supplied

The mobile build can record/display versions reported by the integrated environment, but this repository does not own every service/model version. Missing external version metadata must be recorded as a release-evidence gap rather than guessed.

## 16. Security posture is research-prototype scope

Phase 8/9 hardening removes the mobile shared privileged-service authority pattern, uses secure session storage and enforces HTTPS for remote clinical endpoints. That does not by itself constitute a formal penetration test, regulatory security certification or production clinical-security approval.
