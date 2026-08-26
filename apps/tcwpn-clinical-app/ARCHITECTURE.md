# ClinAnx — architecture

Clinician-facing Flutter client for **R26-DS-012**, a multimodal anxiety risk
framework. This app is a **presentation and interaction layer**. It does not
orchestrate model services and it does not compute risk.

---

## 1. The shape of the system

```
                 ┌─────────────────────┐
                 │   CLINICIAN APP     │
                 │   Flutter (this)    │
                 └──────────┬──────────┘
                            │ HTTPS — one base URL, one token
                            ▼
                 ┌─────────────────────┐
                 │   CENTRAL BACKEND   │
                 │  FastAPI · Postgres │
                 └──────────┬──────────┘
             ┌──────────────┼──────────────┬──────────────┐
             ▼              ▼              ▼              ▼
            C1             C2             C3             C4
      Physiological   Behavioural    Clinical NLP    Demographic
        wearable      phone sensing    TC-WPN        DCAR prior
             │              │              │              │
             └──────────────┴──────┬───────┴──────────────┘
                                   ▼
                            RAGF FUSION
                        gate → harmonise → weight
                                   ▼
                            Composite risk
                                   ▼
                            CARE-AnxRAG
                                   ▼
                   Evidence-grounded clinical guidance
```

The app knows **one** service address. It has never heard of the fusion service,
and it holds no C1/C2/C3/C4 URLs.

The single exception is `GET /health` on the TC-WPN Space, used to wake a
sleeping container so the clinician's first note analysis does not pay the full
cold start. That call carries no credential and no patient data.

## 2. Layering

```
Screen  (lib/features/…)        no HTTP, no URLs, no JSON
   │
Controller (lib/state/…)        one per screen scope; owns loading/error state
   │
Gateway (lib/data/api/…)        one method per backend route; returns models
   │
ApiClient (lib/data/api/…)      base URL, auth, timeout, retry, error mapping
   │
Central Backend
```

Rules that hold across the tree:

- No screen constructs an `http.Client`, a `Uri`, or a `Dio`.
- No screen reads a raw `Map<String, dynamic>`. Gateways return typed models.
- `ApiClient` checks status **before** decoding the body, always.
- Retries apply to idempotent reads only, and never to a TLS pin failure.

## 3. Gateways

Two, and only two.

### `CentralBackendGateway`

| Method | Route |
|---|---|
| `health()` | `GET /health` |
| `enrol()` | `POST /v1/subjects` |
| `resolveMrn()` | `GET /v1/subjects/resolve?mrn=…` |
| `registerExternalId()` | `POST /v1/subjects/{subject_id}/external-ids` |
| `submitNote()` | `POST /v1/clinical-notes` |
| `runFusion()` | `POST /v1/fusion/run` |
| `timeline()` | `GET /v1/doctor/patients/{subject_id}/timeline?limit=n` |
| `evidence()` | `POST /v1/doctor/patients/{subject_id}/evidence` |
| `submitVerdict()` | `POST /v1/verdict` |

### `TcwpnWarmupGateway`

| Method | Route |
|---|---|
| `health()` | `GET {TCWPN_BASE}/health` |

That is the complete list of network calls this application makes.

**Deleted, and not to be revived:** `FusionGateway` (`/contribute`,
`/state/{mrn}` — routes no service ever served), `TcwpnGateway.analyse()`
(direct `POST /predict`, which bypasses the gate, harmonisation, recency
weighting and conformal calibration), and `C3Gateway` (`/v3/risk/classify`, the
retired intervention route). The old intervention engine — GBDT/XGBoost tiering,
SHAP, DiCE, FAISS case retrieval, `/intervene` — is retired outright. Clinical
guidance now comes from CARE-AnxRAG, downstream of fusion.

## 4. Modality naming

The wire keys are the backend's and do not line up with the component numbering
in the dissertation. Both are correct in their own frame; they must never be
mixed.

| Wire key | Service | Paper |
|---|---|---|
| `c1_physiological` | wearable biosensors | Component 1 |
| `c2_behavioral` | phone sensing — **excluded from the composite** | Component 2 |
| `c3_clinical_nlp` | TC-WPN, clinical notes | Component 4 |
| `c4_demographic` | DCAR demographic prior | Component 4 · contextual arm |

**Rule: wire identifiers follow the backend, human-readable labels follow the
paper.** `Modality` in `lib/domain/models.dart` is the only place either is
written down. An earlier build used `c4_clinical_nlp` and `c3_intervention` —
3 and 4 swapped, plus a modality the backend does not have — and against a real
response that produced four contributions with weight `0` and score `null`: the
composite rendered and the breakdown was empty. `test/widget_test.dart` fails
loudly if that regresses.

## 5. Fusion is server-side, and only server-side

`composite_score`, `tier`, `band`, `weights` and `contributions` are read off
the wire and never derived on the device.

- The client holds **no weight table**. It cannot compute a composite even by
  accident.
- Contribution values are the server's, not `weight × score`. The backend
  harmonises the raw score before weighting, so recomputing locally disagrees
  with the composite printed beside it (0.71 × 0.65 = 0.4615; the server says
  0.5428).
- The band is **not** re-derived from the composite. The fusion service bands in
  three tiers at 0.33/0.66; the local display helper splits four ways at
  0.25/0.50/0.75. At 0.8878 those disagree. The server wins.
- When the backend is unreachable the chart shows an explicit unavailable state.
  There is no provisional composite, no fallback fusion, no local renormalisation.

A gate rejection is a normal outcome, not an error: fewer than two usable
modalities yields `composite: null`, `band: "GREY"` and a `reason`. The parser
keeps the null as a null, renders it as an em dash, and shows the reason.
Never `0.000`, never "Stable", never "Low risk".

## 6. Absence is not zero

`status` is passed through verbatim — `ok` · `absent` · `warming_up` ·
`insufficient_data` · `poor_signal` · `no_support_set` · `not_validated` ·
`error` — and only `ok` **with** a non-null score counts as usable evidence,
which is the same test the backend's gate applies.

`c2_behavioral` is a special case worth stating plainly. It is excluded from the
composite by pre-registered rule, not by absence: AUROC 0.5205 against a
permutation null of 0.4991, p = 0.255. When the backend reports an experimental
behavioural value it arrives with `fusion_eligible: false` and is rendered as
**"Behavioural signal · experimental, not included in composite"**. It is never
given a contribution row, a weight, or a percentage.

`c4_demographic: null` likewise means unavailable, never 0 % risk.

## 7. Freshness

The backend is authoritative. `captured_at`, `age_minutes`, `fresh`, `updated_at`
and `computed_at` come from the server and are displayed as received. The device
clock is never substituted for a missing model timestamp.

Two parsing details that are safety properties rather than conveniences:

- A timestamp without an offset is read as **UTC**. SQLite drops `tzinfo` on
  round-trip; Dart's `DateTime.parse` would read the result as local time, which
  on a device at UTC+5:30 puts the composite five and a half hours away from the
  readings in the same payload.
- A reading captured *after* the fusion row being displayed is flagged
  `pendingNextFusion`, not rendered as weight 0.00. Physiological ingests are
  debounced (`AUTO_FUSION_DEBOUNCE_MIN`), so a wearable reading can legitimately
  be `ok` and `fresh` while carrying zero weight — because it has not been
  considered yet, not because it was considered and discounted.

Cached chart data is labelled **Cached · last updated `<timestamp>`**. No
composite is ever computed offline.

## 8. Patient identity

```
MRN  ──►  POST /v1/subjects  ──►  subject_id (opaque UUID) + pairing code
```

The backend HMAC-hashes the MRN on arrival and never persists the raw value. The
app stores the returned `subject_id` and uses it for every subsequent call. The
raw MRN goes over the wire exactly twice — enrolment and `resolve` — because
only the server holds the pepper.

`ChartController` is constructed per patient and holds its identifier as a
`final` field; every store call takes it as a required argument. Switching
patients disposes the previous controller. There is no static "active patient".

## 9. Configuration

Build-time only, via `--dart-define`. `lib/core/config/env.dart` is the single
place any of it is read.

| Define | Required | Purpose |
|---|---|---|
| `BACKEND_BASE` | yes | Central Backend base URL |
| `BACKEND_TOKEN` | yes | shared backend token (`BACKEND_API_TOKEN` server-side) |
| `TCWPN_BASE` | no | TC-WPN Space, `/health` warm-up only |
| `AUTH_BASE` | no | clinician auth service; empty ⇒ local demo mode |
| `AUTH_SALT` | no | local-mode password salt |
| `AUTH_LOCAL` | no | local-mode account table |
| `DEMO_DATA` | no | `false` for any build touching real patients |
| `DISABLE_TLS_PINNING` | no | emulator work behind a debugging proxy only |

`C1_BASE`, `C2_BASE`, `C3_BASE`, `C4_BASE`, `FUSION_BASE`, `CARE_RAG_BASE` and
`HF_TOKEN` are **not** configuration for this app. If you find one in a build
script, that script predates the Central Backend.

## 10. Security posture, stated honestly

- Session material is held in `flutter_secure_storage`, never
  `SharedPreferences` and never a plain JSON file.
- No token is compiled into source. `BACKEND_TOKEN` is injected at build time.
- **`BACKEND_TOKEN` is a single shared app credential, not a per-clinician one.**
  Anything inside an APK is extractable. Clinician attribution therefore travels
  in each request body's `author` field. This is adequate for a research
  prototype and is *not* adequate for clinical deployment, which needs per-user
  authentication and authorisation. `ApiClient` evaluates its bearer through a
  callback per request, so replacing this with a clinician JWT is a change at
  one call site.
- TLS pinning is available (`SecureHttp`) and fails closed: an unconfigured pin
  set raises `PinningNotConfigured` rather than silently falling back to the
  platform trust store. See the outstanding item below.
- Patient note text is never written to a log.

## 11. Known gaps

1. **`kPinnedHosts` lists only the TC-WPN Space.** That host now carries nothing
   but an unauthenticated `/health` ping, while every note, every composite and
   every evidence call goes to the Central Backend — which is *not* pinned and
   therefore uses the platform trust store. The priority is inverted. Add the
   backend host and regenerate with `python tool/pin_certs.py`.
2. **`kPinsReviewBy` is `1970-01-01`**, so `SecureHttp.needsReview` is
   permanently true and Settings shows a stale-pin warning that can never be
   cleared. Set a real review date when the pins are generated.
3. **No per-clinician authentication against the Central Backend** — see §10.
4. `POST /v1/fusion/run` has no dedicated regression test; the fusion path is
   covered through the timeline parser instead.