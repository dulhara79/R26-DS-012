# ClinAnx — R26-DS-012 clinician console

Flutter app for psychiatry teams: enrol a patient, submit a clinical note, and
read the multimodal anxiety risk assessment the **Central Backend** produces
from it.

The app is a presentation layer. Every model call, every fusion, and every piece
of clinical guidance happens server-side.

```
Flutter (this repo)  →  Central Backend  →  C1 · C2 · C3 · C4
                                         →  RAGF fusion
                                         →  CARE-AnxRAG
```

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the contract, the layering rules,
and the known gaps. See [`APK_BUILD.md`](APK_BUILD.md) for release builds.

---

## Quick start

```bash
flutter pub get

flutter run \
  --dart-define=BACKEND_BASE=https://your-central-backend.example \
  --dart-define=BACKEND_TOKEN=your-backend-token
```

On the Android emulator, a backend running on your own machine is
`http://10.0.2.2:8000` — not `localhost`.

Every configuration value is a `--dart-define`; see `.env.example` for the full
list and `lib/core/config/env.dart` for where each one is read. Nothing is
hard-coded, and no token belongs in source.

## Verify

```bash
flutter pub get
flutter analyze
flutter test
flutter build apk --debug
```

## What the app does

| Screen | Purpose |
|---|---|
| Caseload | roster, enrolment state, latest band per patient |
| Patient chart | composite, tier, band, confidence, per-modality breakdown, trend |
| Note analysis | submit a clinical note; TC-WPN runs server-side |
| Support set | manage the labelled notes that form TC-WPN's prototypes |
| Fusion detail | gate decision, weights, contributions, conformal set |
| Evidence | CARE-AnxRAG guidance, with abstention and insufficient-evidence states |
| Alerts, Settings | escalations; service health, pin status, session |

## What the app deliberately does not do

- It does not compute a composite score. It holds no weight table, and there is
  no local or provisional fusion — if the backend is unreachable, the chart says
  so.
- It does not call C1, C2, C3, C4, the fusion service, or CARE-AnxRAG directly.
  It holds none of their URLs.
- It does not call TC-WPN's `/predict`. The only direct call to the Space is an
  unauthenticated `GET /health` warm-up.
- It does not implement `/v3/risk/classify` or `/intervene`. The old
  GBDT/SHAP/DiCE intervention engine is retired.
- It does not turn a missing score into `0`, or a failed request into "Low risk".

## Modality naming

| Wire key | Signal | Paper |
|---|---|---|
| `c1_physiological` | wearable | Component 1 |
| `c2_behavioral` | phone sensing — excluded from the composite | Component 2 |
| `c3_clinical_nlp` | TC-WPN clinical notes | Component 4 |
| `c4_demographic` | DCAR prior | Component 4 · contextual |

`c2_behavioral` is reported but never fused: AUROC 0.5205 against a permutation
null of 0.4991, p = 0.255. Where a behavioural value exists it is shown as
**experimental, not included in composite** — never as a contribution.

## Layout

```
lib/
├── core/        config · design tokens · TLS pinning
├── data/        api/ (ApiClient, gateways, session, auth) · local/ (stores)
├── domain/      typed models — Patient, FusionResult, ModalityReading, …
├── state/       RosterController, ChartController
└── features/    screens; no HTTP, no URLs, no raw JSON
```

## Research context

R26-DS-012, SLIIT. TC-WPN (Temporal-Confidence Weighted Prototypical Networks)
is Component 4: few-shot anxiety detection from clinical notes. Its published
result is a **mechanistic negative result** — the temporal and consistency
weighting mechanisms produce no statistically significant improvement over an
auxiliary-head-only baseline. Nothing in this UI should imply otherwise, and no
performance figure is hard-coded into a screen: Settings renders whatever the
service's own `/health` reports.
