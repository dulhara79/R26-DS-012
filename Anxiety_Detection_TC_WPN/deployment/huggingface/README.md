---
title: Tc Wpn Demo
emoji: 🏢
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 6.14.0
python_version: '3.11'
app_file: app.py
pinned: false
short_description: TC-WPN clinical anxiety detection — R26-DS-012 component C4
---

# TC-WPN — deployment layer

This Space is the **deployment layer** for TC-WPN, mirrored from
`dulhara79/tcwpn_test` → `deployment/huggingface/`. It is deliberately **not** a
copy of the research repository.

It answers exactly one question:

> Given this note text and this support set, what does TC-WPN predict?

It does **not** answer *"what is the final multimodal anxiety risk?"* — that
belongs to the central backend. There is no fusion logic here, and
`tests/test_contract.py` fails the build if any fusion vocabulary appears in a
prediction.

**Contract version 1.1.0.** v1.1.0 adds the R26-DS-012 §1 common envelope
(`subject_id`, `modality`, `score`, `status`, `captured_at`, `computed_at`,
`latency_ms`) and fixes the temporal axis. See *Migrating from 1.0.0* below.

---

## The support set is the classifier — read this before integrating

This is the single thing most likely to be got wrong, so it is first.

TC-WPN is a **prototypical network**. It stores no decision boundary and no
prototypes. On every request it embeds the K support notes, builds one weighted
centroid ("prototype") per class from them, embeds the query note, and
classifies by cosine distance to those two centroids. The prototypes are
constructed at request time and discarded.

So `support_set` is **a bank of labelled reference notes from other patients**,
curated once per site. It is **not** the queried patient's own note history.

`src/tcwpn/sampler.py` guarantees one support slot per distinct patient and
`support_patients ∩ query_patients = ∅` in every training episode;
`src/tcwpn/collate.py` raises on overlap. **Nothing raises at serving time.**
The backend must enforce the same exclusion when it selects support notes —
otherwise it reproduces the exact leakage the clean pipeline was rebuilt to
remove, and no error will tell you.

### Who does what

| Step | Owner |
|---|---|
| Curate the labelled reference bank (de-identified, other patients) | Clinical lead, once |
| Version the bank and store it | Central backend |
| Select K notes per request, **excluding the queried `subject_id`** | Central backend |
| Compute `days_before_index` per support note | Central backend |
| Build prototypes, score the note | This service |

The frozen benchmark is **K = 5** and only K = 5. Serving at another K is not
wrong, but it is outside what was measured; default the selector to 5.

### There is no `no_support_set` status

With K = 0 there are no prototypes and nothing to classify against. The API
returns `422 MISSING_SUPPORT_SET`. (`tcwpn_full` does carry an `aux_head` over
the query embedding alone, so a support-free score is technically computable —
but `evaluation.py` scores the *prototype* path, so the aux head has no
measured performance and is not served.)

---

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/predict` | bearer | Inference |
| POST | `/api/predict` | bearer | Alias |
| GET | `/health` | — | Provenance, config, research metrics |
| GET | `/docs` | — | Interactive explorer |
| GET | `/ui` | — | Gradio demo |
| POST | `/auth/*` | — | Account management (unchanged) |

Research metrics appear **only** on `/health`, never in a prediction response.

## Provenance

Every identifier in the chain is served at `GET /health`:

```
TC-WPN git commit  →  training config  →  checkpoint  →  evaluation
                   →  HF model repo    →  Space       →  API response
```

`app.py` refuses to start if the checkpoint it downloads does not match
`deployment_config.json` — by sha256 and by architecture. That is what lets the
question *"how do you know the deployed model is the model you evaluated?"* be
answered by pointing at a URL.

---

## Request

```json
{
  "subject_id": "S-7f3a91c2",
  "note_text": "Patient presents with persistent worry...",
  "note_type": "Psychiatry note",
  "note_date": "2026-08-19T09:00:00Z",
  "visit_count": 3,
  "support_set_version": "nhsl-bank-v3",
  "support_set": [
    { "id": "bank-114", "text": "...", "label": "anxiety",
      "note_date": "2025-11-02T00:00:00Z", "days_before_index": 41.0 },
    { "id": "bank-207", "text": "...", "label": "control",
      "note_date": "2026-01-14T00:00:00Z", "days_before_index": 12.5 }
  ],
  "return_attention": true,
  "return_support_contributions": true
}
```

| Field | Rule |
|---|---|
| `subject_id` | Canonical backend ID. `patient_id` is accepted as a deprecated alias; the response always returns `subject_id`. |
| `support_set` | **Required**, ≥1 `anxiety` and ≥1 `control`. No fallback to demo examples in the API. |
| `support_set_version` | Identifier of the bank these notes came from. Echoed on the response; put it in the audit log. |
| `days_before_index` | **Preferred.** See below. |
| `note_date` | On the query: the index time, and `captured_at` on the response. On a support note: fallback only. |

### `days_before_index` — supply it

The model's temporal input drives `w_i^T = exp(−λ · dt_i / 365)`, λ = 0.5.
`scripts/apply_index_time.py` defines what `dt` is:

```
days_before_patient_last_note := days_before_index = (t_index − charttime)
```

where `t_index` is the index time of **that note's own patient**. Because
support notes come from K different patients, a K-shot set carries K
independent reference points. Only the backend knows each note's patient
anchor, so only the backend can compute this.

`temporal_axis` on the response says which path was taken:

| Value | Meaning |
|---|---|
| `backend_supplied` | Every note carried `days_before_index`. Reproduces training semantics. |
| `approximated` | At least one note fell back to `(query note_date − support note_date)`. Treats the query as the index time — right for the prediction point, wrong for the support note's own patient. |
| `unavailable` | No usable dates. All deltas 0, so `w^T = 1` for every note and temporal weighting is inert. |

---

## Response

```json
{
  "subject_id": "S-7f3a91c2",
  "modality": "c4_clinical_nlp",
  "score": 0.68,
  "status": "ok",
  "captured_at": "2026-08-19T09:00:00Z",
  "computed_at": "2026-08-19T09:00:04Z",
  "model_version": "tcwpn-clean-benchmark-36d7413",
  "latency_ms": 840,

  "prediction": "ANXIETY",
  "probability": 0.68,
  "threshold": 0.26931220789750415,
  "confidence": 0.68,
  "entropy": 0.6266,
  "calibration_status": "uncalibrated",

  "prototype_distance_anxiety": 0.42,
  "prototype_distance_control": 0.88,
  "temperature": 10.0,

  "support_count": { "anxiety": 3, "control": 2, "k": 5 },
  "support_set_version": "nhsl-bank-v3",
  "evaluated_k": 5,
  "temporal_axis": "backend_supplied",
  "undated_support_notes": 0,

  "temporal_weighting_used": true,
  "prototype_consistency_weighting_used": true,
  "application_risk_band": "MODERATE",

  "preprocessing_version": "bioclinicalbert-maxlen512-maxchunks1-stride128-v1",
  "contract_version": "1.1.0",
  "tcwpn_git_commit": "d5277314891077840316789e8eab55b23fd8c6e1",
  "analysed_by": "clinician-uuid"
}
```

Five fields need reading carefully.

**`score`** is the envelope field fusion reads. It is the same float as
`probability` — two names, one number, so neither the backend nor the app has
to guess which was meant.

**`probability` is uncalibrated.** It is a softmax over cosine-distance
prototype logits, and no calibrator is fitted anywhere in this pipeline
(`calibrator_fitted: false`; seed-42 clean benchmark **ECE 0.0849, Brier
0.209**). There is no `calibrated_probability` field, because there is no
calibrated probability. Fusion averages this with C1 and C2 as though the three
were comparable probabilities — that is a modelling assumption and belongs in
the paper next to the timebase caveat.

**`temporal_weighting_used` / `prototype_consistency_weighting_used`** report
which mechanisms are *enabled in this build*, and nothing more. Phase 4, five
seeds, paired against `aux_only`: `tcwpn_full` **+0.0006 (p = 0.886)**,
`temporal_aux` +0.0006 (p = 0.906), `pcw_aux` −0.0080 (p = 0.150). Neither w^T
nor w^C shows a detectable effect once the auxiliary CE head is held constant.
Neither this API's documentation, nor the clinician app's copy, nor the paper
may attribute a score to either mechanism. The `mechanism_note` field carries
this text into every response so it cannot be lost in transit.

**`application_risk_band`** is a UI convenience, not a model output. TC-WPN was
evaluated as a binary classifier; LOW/MODERATE/HIGH/VERY HIGH were never
validated as clinical classes.

**`attention_based_highlighted_spans`** (with `return_attention: true`) are
attention-derived textual cues. Attention weights are not automatically
faithful feature attribution and no faithfulness analysis has been performed,
so they must not be presented as the features responsible for the prediction.
They are wordpiece-merged tokens, not multi-word clinical phrases — do not
build a UI that assumes phrase-level spans.

### Statuses and errors

`/predict` returns `status: "ok"` or `status: "error"`. That is the whole
vocabulary for this service.

| `error_code` | HTTP | When |
|---|---|---|
| `MISSING_NOTE_TEXT` | 400 | empty `note_text` |
| `MISSING_SUPPORT_SET` | 422 | 0 support notes, or one-sided |
| `INVALID_SUPPORT_LABEL` | 422 | label not `anxiety`/`control` |
| `INFERENCE_FAILED` | 500 | inference threw |
| `SERVICE_NOT_READY` | 503 | checkpoint not loaded |

Error responses carry `modality`, `score: null`, `model_version` and
`computed_at`, so the backend can store the failure as a typed reading and show
the clinician a **gap** rather than silently dropping a modality that was due.

`warming_up`, `insufficient_data`, `poor_signal` and `not_validated` belong to
C1 and C2. C4 never emits them.

---

## AUROC vs the threshold

Different quantities, reported separately on `/health`:

- **AUROC** is threshold-free discrimination.
- **`threshold`** is an operating point selected on validation
  (`threshold_objective: f1`), never re-tuned on test.

The threshold is not derived from the AUROC.

**Figures that must not be republished:** `0.9635`, `0.8989`, `0.9291`,
`0.4036`. These come from the archived pipeline — keyword gate applied to
positives only, test split filtered by a text-derived score, and support/query
patient overlap (`AUDIT_OF_ARCHIVED_PIPELINE.md`). The clean five-seed
`tcwpn_full` K=5 benchmark is **AUROC 0.7377, SD 0.0031**; seed-42 blinded
(anxiety lexicon masked) is **0.6284**. `test_contract.py` asserts the stale
figures do not appear.

## Clinical status

Clinical decision support only. Not a diagnostic device. Research metrics are
measured on MIMIC-IV and are **not** validated on NHSL data.

---

## Migrating from contract 1.0.0

| 1.0.0 | 1.1.0 |
|---|---|
| `patient_id` (request + response) | `subject_id`; `patient_id` still accepted on the request |
| — | `modality`, `score`, `captured_at`, `computed_at`, `latency_ms` added |
| — | `prototype_distance_anxiety` / `_control`, `temperature` restored |
| — | `support_set_version`, `evaluated_k`, `calibration_status`, `mechanism_note` added |
| support note: `note_date` only | `days_before_index` added and preferred |
| `temporal_reference_note_date` | `temporal_axis` + `temporal_reference` + supplied/approximated/undated counts |
| `support_contributions[].days_before_last_note` | `.days_before_index`, and `.id` echoed |
| `threshold: 0.2693` | `0.26931220789750415` (full locked precision) |

Nothing in the auth layer changed. `/auth/*`, `accounts.py`, `mailer.py`, the
bearer requirement on `/predict` and `analysed_by` are all as they were.

---

## Space configuration notes

- `sdk: gradio` is retained. A Docker Space needs a paid plan on a personal
  account, and the Gradio SDK Space already serves the FastAPI app through
  `gr.mount_gradio_app` — that is how `/predict`, `/health` and `/auth/*` are
  reachable. `Dockerfile` stays in GitHub only, for a future move to Render or
  a PRO Docker Space.
- `python_version` is pinned to `3.11`. The pinned CPU torch wheel has no cp313
  build, so leaving this at `3.13` breaks the install.
- Secrets go in Settings → Variables and secrets, never in git:
  `HF_TOKEN`, `REQUIRE_AUTH=true`, `STRICT_STARTUP=true`,
  `ALLOWED_ORIGINS=<clinician app origin>`.
- `.gitignore` covers `create_account.py` and `.env`. Keep it that way; the
  Space repo is public.
- Before pushing: delete `tc_wpn/models/core.py`, `core_v3.py`, `embedder.py`,
  `embedder_old.py` and `patient_level_eval.py` from the Space. With two
  `core*.py` files present it is not determinable which one the checkpoint
  matches.