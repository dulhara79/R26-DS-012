# ClinAnx — R26-DS-012 clinician mobile app

> **Research prototype — not a diagnostic device.** ClinAnx is an engineering/research client for R26-DS-012. It must not be presented as clinical deployment approval and does not guarantee that an anxiety event will occur at an exact future time.

ClinAnx is the clinician-facing Flutter application for the R26-DS-012 multimodal anxiety platform. The mobile app presents server-owned assessment, forecast, AttentionEvent, evidence, and clinician-workflow data. It does **not** compute authoritative multimodal risk, run component models directly, or create the authoritative escalation state.

```text
ClinAnx (this repo)
        │ authenticated HTTPS
        ▼
Central Backend
        ├── C1 physiological
        ├── C2 behavioural (experimental / excluded from active fusion)
        ├── C3 Clinical NLP / TC-WPN
        ├── C4 contextual-demographic
        ├── fusion / current assessment
        ├── forecast + AttentionEvent lifecycle
        └── CARE-AnxRAG evidence

AttentionEvent delivery
        ├── FCM push when configured
        └── 30-second foreground polling fallback
```

## Handbook implementation status

This repository implements the **ClinAnx/mobile-side contract** through Handbook Phase 8 and contains the Phase 9 research-release package. That statement does not claim that the Central Backend, patient application, or complete cross-repository deployment has been independently verified from this repository.

Phase 9 follows the handbook requirement for **pinned versions, SOP, test evidence, deployment/runbook, and known limitations**.

## Main clinician workflow

| Area | Purpose |
|---|---|
| Dashboard | Assignment-scoped worklist, OPEN AttentionEvents, current assessment and forecast presentation |
| Patients | Patient selection and patient overview using canonical server identity |
| Patient Overview | Current multimodal assessment kept separate from near-term physiological forecast |
| Signals / Data Quality | C1-C4 availability, freshness, inclusion and server-reported contribution/provenance |
| Timeline | Server-reported historical assessment rows; no locally invented history |
| Clinical Notes | Local draft workflow; submitted note analysis is orchestrated through the Central Backend |
| Supporting Evidence / Ask CARE | CARE-AnxRAG evidence with explicit abstained/unavailable states |
| Clinician Assessment | Human judgement recorded separately from model assessment |
| Activity | Persistent server AttentionEvent lifecycle: OPEN → ACKNOWLEDGED → RESOLVED |
| Settings | Research build identity, environment/backend/Firebase routing information and session actions |

## Safety and authority rules

- The Central Backend is the authority for assessment, forecast and AttentionEvent state.
- ClinAnx never computes an authoritative multimodal composite locally.
- Missing, stale or unavailable data is not converted to `0`, Green, Stable or Low risk.
- Current assessment and forecast remain separate objects and labels.
- C2 is shown as **experimental — not included in fusion** while the registered exclusion rule remains active.
- TC-WPN is a Clinical NLP signal, not the overall multimodal risk authority.
- CARE-AnxRAG timeout/unavailability and abstention remain explicit; the app does not invent fallback guidance.
- Clinician session material is stored using platform secure storage.
- Remote participant/clinical traffic requires HTTPS; plaintext HTTP is limited to loopback development.
- Push transport accepts routing identity only (`type=attention_event`, `event_id=...`); clinical detail is fetched after authenticated open.
- Push failure does not remove the server-backed Activity/polling recovery path.

## Quick development verification

```bash
flutter pub get
flutter test
flutter analyze
```

For a local/demo run, provide configuration through `--dart-define` values read by `lib/core/config/env.dart`. Example:

```bash
flutter run \
  --dart-define=BACKEND_BASE=http://127.0.0.1:8000 \
  --dart-define=APP_VERSION=1.0.0+1 \
  --dart-define=BUILD_ENVIRONMENT=demo \
  --dart-define=BUILD_REVISION=$(git rev-parse --short HEAD) \
  --dart-define=DEMO_DATA=true
```

Use local/demo authentication only with synthetic data. Research/study builds must use the configured clinician authentication service, HTTPS remote endpoints and `DEMO_DATA=false`.

See [`APK_BUILD.md`](APK_BUILD.md) for the reproducible Primary/Secondary Firebase release commands.

## Push and Firebase disaster-recovery design

ClinAnx uses Firebase Cloud Messaging only for push transport. It does not require Firebase Realtime Database, Firestore, Firebase Storage or Firebase Functions.

Two Firebase **build slots** are supported:

- `primary` — normal research build
- `secondary` — disaster-recovery build using the second Firebase project

This is intentionally not described as an instant runtime project/API-key switch. FCM registration tokens are project-specific. Runtime resilience is provided by persistent server AttentionEvents plus the independent polling fallback.

## Version and reproducibility policy

- App version is declared in `pubspec.yaml` and injected into research build identity as `APP_VERSION`.
- Source revision is injected as `BUILD_REVISION` and must match the commit used to build the artifact.
- Dart/Flutter package resolution is locked by the committed `pubspec.lock`.
- Phase 9 CI is pinned to the Flutter SDK version recorded in the release documentation.
- Backend/service/model versions are displayed/recorded when supplied by the integrated environment; this mobile repository does not invent versions for services it does not own.

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — current ClinAnx architecture and trust boundaries
- [`APK_BUILD.md`](APK_BUILD.md) — reproducible APK/build instructions
- [`docs/release/PHASE9_RESEARCH_RELEASE.md`](docs/release/PHASE9_RESEARCH_RELEASE.md) — Phase 9 release manifest/checklist
- [`docs/release/SOP.md`](docs/release/SOP.md) — research release standard operating procedure
- [`docs/release/DEPLOYMENT_RUNBOOK.md`](docs/release/DEPLOYMENT_RUNBOOK.md) — configure, build, install, smoke-test and rollback
- [`docs/release/TEST_EVIDENCE.md`](docs/release/TEST_EVIDENCE.md) — automated/manual verification evidence and evidence gaps
- [`docs/release/KNOWN_LIMITATIONS.md`](docs/release/KNOWN_LIMITATIONS.md) — scientific, mobile, push, backend-integration and packaging limitations
- [`docs/qa/phase8_usability_checklist.md`](docs/qa/phase8_usability_checklist.md) — manual Phase 8 usability/hardening evidence checklist
- [`docs/integration/PHASE6_ATTENTION_EVENT_API_CONTRACT.md`](docs/integration/PHASE6_ATTENTION_EVENT_API_CONTRACT.md) — frozen AttentionEvent client contract

## Repository layout

```text
lib/
├── core/        configuration, notifications, common infrastructure
├── data/        API/session/repositories and clinician-scoped local stores
├── domain/      typed contracts and models
├── state/       controllers and view state
└── features/    clinician screens and navigation

docs/
├── integration/ backend-facing client contracts
├── qa/          manual hardening/usability evidence
└── release/     Phase 9 research-release package
```

## Research context

R26-DS-012 is a research project. The mobile UI must preserve the distinction between model output, supporting evidence and clinician judgement. Research/model performance claims belong in the research artifacts and validated service metadata; they are not hard-coded into the clinician UI or treated as clinical-device claims here.
