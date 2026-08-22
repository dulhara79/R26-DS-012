# What's new in this build — auto-trigger, conformal prediction, CARE-AnxRAG

Test suite: **121 passed, 0 failed** (was 80). CARE-AnxRAG (built separately) is now wired in as a called HTTP service, not imported code. Run it yourself:
`cd central_backend && python3 test_backend.py`
(new deps: none — everything added is stdlib + what you already installed)

## 1 · Auto-trigger (diagram: "FUSION — triggered by the event")
- Contextual and clinical-note ingests now fuse IMMEDIATELY after storing.
- Physiological ticks are DEBOUNCED: fusion fires only if >= AUTO_FUSION_DEBOUNCE_MIN
  (env var, default 5) minutes since this subject's last fusion of any kind.
- Behavioural never triggers — it is excluded from the composite, so a fusion
  after it could not change the answer.
- Ingest responses now include `fusion_triggered` (+ `fusion` summary or
  `fusion_skipped_reason`). A fusion failure never fails the ingest.
- Manual `POST /v1/fusion/run` unchanged.

## 2 · Conformal prediction + clinician verdicts
- `POST /v1/verdict {fusion_result_id, tier_label, author}` — the clinician's
  HITL judgement. This is BOTH your label-collection mechanism for the paper
  AND the calibration source. UI rule: record the verdict BEFORE showing the
  conformal set, or the label is contaminated.
- Every fusion response now carries `conformal_set`, `conformal_alpha` (0.10),
  `conformal_calibrated`, `conformal_n`, `conformal_note` alongside the
  unchanged `tier` field.
- Honest by construction: with < 20 verdicts the set is ALL tiers and
  `calibrated: false` with a stated reason. It tightens only when labels earn it.
- `conformal.coverage_report()` gives the paper's two numbers together:
  empirical coverage AND mean set size.
- Storage note: conformal lives inside the fusion row's harmonisation JSON —
  no schema migration needed on your existing dev database. (New `verdicts`
  table is created automatically.)

## 3 · CARE-AnxRAG integration (separate HTTP service)
`POST /v1/doctor/patients/{subject_id}/evidence  {"question": "..."}`

- CARE-AnxRAG is called over HTTP — it is NOT imported into central_backend.
  Start it separately (`care-anxrag serve --host 127.0.0.1 --port 8000`), set
  `RAG_URL` in `.env`, done. This keeps central_backend small and lets the RAG
  service be redeployed/restarted independently — see `rag_client.py`.
- Current scope, matching CARE-AnxRAG's own documented contract: question in,
  structured answer out. No patient data (subject_id, notes, scores) is
  forwarded — subject_id is used only for auth/audit on the backend side.
- CARE-AnxRAG does its OWN retrieval, evidence scoring, and abstention. The
  backend's job is narrow: call it, and never fabricate an answer if the call
  fails. A failed/unreachable RAG service returns `available: false` with an
  error — same rule every other component in this system follows.
- TWO safety layers, deliberately not one: CARE-AnxRAG's own `safety_level` /
  `safety_message` are surfaced as-is, AND a local, dependency-free crisis
  pre-screen runs in central_backend BEFORE the network call — verified in
  tests to genuinely skip the network entirely when it fires. This is a
  backstop independent of CARE-AnxRAG's reachability or configuration, not a
  replacement for its own handling.
- `GET /health` now includes a `rag` block: `configured` + `reachable` +
  whatever CARE-AnxRAG's own `/health` reports.
- ⚠️ Not yet run against the REAL CARE-AnxRAG service — tests here stub the
  HTTP layer (`rag_client.call_rag`), since the actual service (~2GB, local
  Ollama) isn't available in this environment. Before treating this as
  integration-ready, run `./scripts/validate.sh` in CARE-AnxRAG itself (its
  author flagged a recent `generation.py` fix as pending a full validation
  re-run), then start it locally and re-run `test_backend.py` — or better,
  add a live smoke test that only runs when `RAG_URL` is actually reachable.

## ⚠️ Do this now: rotate your pepper
Your uploaded zip contained your real `.env` (MRN_PEPPER inside). It has now
been shared, so treat it as burned. Generate a new one and update `.env`:
    python3 -c "import secrets; print('MRN_PEPPER=' + secrets.token_urlsafe(48))"
Never include `.env` in a zip again — add it to `.gitignore` too.

## Still pending (unchanged)
Real reference distributions from teammates · real guideline texts ·
deployment (Render/Oracle; backend NOT on HF) · Flutter wiring · Ollama for
real generation · fine-tuning DCAR.
