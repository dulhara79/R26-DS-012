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
belongs to the central backend. There is no fusion logic here.

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

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/predict` | bearer | Inference |
| POST | `/api/predict` | bearer | Alias |
| GET | `/health` | — | Provenance, config, research metrics |
| GET | `/docs` | — | Interactive explorer |
| GET | `/ui` | — | Gradio demo |
| POST | `/auth/*` | — | Account management |

Research metrics appear **only** on `/health`, never in a prediction response.

## Request

```json
{
  "patient_id": "P001",
  "note_text": "...",
  "note_date": "2026-08-22T10:00:00Z",
  "visit_count": 4,
  "support_set": [
    {"id": "S1", "text": "...", "label": "anxiety", "note_date": "2026-08-18T10:00:00Z"},
    {"id": "S2", "text": "...", "label": "anxiety", "note_date": "2026-08-20T10:00:00Z"},
    {"id": "S3", "text": "...", "label": "control", "note_date": "2026-08-19T10:00:00Z"}
  ]
}
```

`support_set` is **required**, with at least one `anxiety` and one `control`
note. There is no fallback to demo examples in the API. A missing or one-sided
support set returns:

```json
{
  "status": "error",
  "error_code": "MISSING_SUPPORT_SET",
  "message": "Both anxiety and control support examples are required for TC-WPN inference."
}
```

The Gradio UI at `/ui` does keep demo examples, and flags when it is using them.

## Response

```json
{
  "model": "TC-WPN",
  "model_version": "tcwpn-clean-benchmark-<commit>",
  "prediction": "ANXIETY",
  "probability": 0.72,
  "threshold": 0.2693,
  "confidence": 0.72,
  "entropy": 0.59,
  "support_count": {"anxiety": 2, "control": 1},
  "temporal_weighting_used": true,
  "used_default_support_set": false,
  "preprocessing_version": "...",
  "inference_configuration": {"projection_dim": 256, "lambda_decay": 0.5, "beta": 2.0, "aux_weight": 0.3},
  "application_risk_band": "HIGH"
}
```

Two fields need reading carefully:

- **`probability`** is uncalibrated — a softmax over cosine-distance prototype
  logits. No calibrator is fitted in this deployment.
- **`application_risk_band`** is a UI convenience, not a model output. TC-WPN
  was evaluated as a binary classifier; LOW/MODERATE/HIGH/VERY HIGH were never
  validated as clinical classes.

With `return_attention: true` the response carries
`attention_based_highlighted_spans`. These are attention-derived textual cues.
Attention weights are not automatically faithful feature attribution and no
faithfulness analysis has been performed, so they must not be presented as the
features responsible for the prediction.

## AUROC vs the threshold

These are different quantities and `/health` reports them separately:

- **AUROC** is a threshold-free discrimination metric.
- **`threshold`** is an operating point selected on the validation set
  (`threshold_objective: f1`), never re-tuned on test.

The threshold is not derived from the AUROC.

## Clinical status

Clinical decision support only. Not a diagnostic device. Research metrics are
measured on MIMIC-IV and are not validated on NHSL data.

## Space configuration notes

- `sdk: gradio` is retained. A Docker Space needs a paid plan on a personal
  account, and the Gradio SDK Space already serves the FastAPI app through
  `gr.mount_gradio_app` — that is how `/predict`, `/health` and `/auth/*` are
  reachable today. `Dockerfile` stays in GitHub only, for a future move to
  Render or a PRO Docker Space.
- `python_version` is pinned to `3.11`. The pinned CPU torch wheel has no cp313
  build, so leaving this at `3.13` breaks the install.
- Secrets go in Settings -> Variables and secrets, never in git:
  `HF_TOKEN`, `REQUIRE_AUTH=true`, `STRICT_STARTUP=true`,
  `ALLOWED_ORIGINS=<clinician app origin>`.
- `.gitignore` covers `create_account.py` and `.env`. Keep it that way; the
  Space repo is public.