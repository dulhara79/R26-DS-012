---
title: DCAR Demographic Anxiety Risk
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# DCAR — Demographic & Contextual Anxiety Risk

Component 4 of **R26-DS-012**. Maps five demographic fields to a calibrated
`P(GAD-7 total >= 10)` and returns the reliability metadata the fusion layer needs.

**This is a population prior, not a diagnostic instrument.** It is scored once per
patient at first login and never changes.

## Deploy

```bash
huggingface-cli repo create dcar-demographic-risk --type space --space_sdk docker
git clone https://huggingface.co/spaces/<user>/dcar-demographic-risk && cd dcar-demographic-risk
cp -r <repo>/hf_space/{Dockerfile,app.py,requirements.txt,README.md,artefacts} .
git lfs track "*.joblib" && git add -A && git commit -m "DCAR v1.0" && git push
```

Set `DCAR_API_TOKEN` as a Space secret. Without it the endpoints are open — dev only.

## Endpoints

| Route | Purpose |
|---|---|
| `GET /health` | model version, feature list, operating threshold. The clinician app's Settings screen renders this rather than hard-coded metrics. |
| `POST /predict` | full result: score, severity distribution, expected GAD-7, confidence, coverage |
| `POST /fusion_component` | the `c4_demographic` block, ready to drop into `/fuse` |
| `GET /docs` | OpenAPI |

```bash
curl -X POST https://<user>-dcar-demographic-risk.hf.space/predict \
  -H "Authorization: Bearer $DCAR_API_TOKEN" -H "Content-Type: application/json" \
  -d '{"patient_id":"NHSL-0142","gender":"female","age":21,
       "edu":"bachelor'\''s degree","smoke":"never smokes","drink":"never drinks"}'
```

## Notes

- `available` is false below 3 of 5 fields supplied — the fusion layer excludes the stream rather than scoring a mostly-empty profile.
- `percentile` is the score's rank in the frozen reference distribution. The fusion service uses it for cross-modality harmonisation; without it you are averaging incommensurable numbers.
- The Space is CPU-only and cold-starts in a few seconds. Because DCAR runs once per patient, cold start is not on any critical path.
