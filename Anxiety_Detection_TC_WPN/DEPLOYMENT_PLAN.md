# TC-WPN — deployment plan

Implements the review's Phases 1–8. This document is the checklist; the files
under `deployment/huggingface/` are the implementation.

**Verdict being acted on:** the Space is not thrown away. Its FastAPI structure,
authentication, structured `support_set`, temporal metadata, strict checkpoint
loading and Gradio demo are kept. What changes is that it becomes a small
deterministic serving layer around the *final validated* TC-WPN implementation,
with the checkpoint pinned to that implementation.

---

## What was created

```
tcwpn_test/
├── src/tcwpn/                      research code — unchanged, source of truth
├── configs/  scripts/  data/  tests/  archives/  notebooks/  docs/
│
└── deployment/
    └── huggingface/                <- the deployment boundary (§14)
        ├── README.md               Space front matter + API contract
        ├── Dockerfile
        ├── app.py                  the wrapper
        ├── deployment_config.json  the ONE versioned unit (§3, §17)
        ├── requirements.txt        CPU torch wheel
        ├── vendor.sh               copies src/tcwpn/model.py in, stamps commit
        ├── tc_wpn/
        │   ├── model.py            VENDORED by vendor.sh — do not edit here
        │   └── preprocessing.py    verbatim chunk_tokenize from the repo
        ├── scripts/
        │   └── export_checkpoint.py    Phase 4
        └── tests/
            ├── test_contract.py            Phase 6
            └── test_local_vs_hf_parity.py  Phase 7
```

`accounts.py` and `mailer.py` come across from the existing Space unchanged.

---

## Two things the review flagged that the repo settles

The review was explicit that it could not read the repository source files or
the integration PDF, and did not invent conclusions about them. Two of its open
questions are answerable from the repo, and both **support** the plan rather
than change it.

**1. Phase 3 removes a live bug, not just a versioning risk.** The old Space
wrapper's `build_prototype` omitted `model.temporal_encoder(...)`, which
`tc_wpn/models/core.py::TCWPN.build_prototypes` applies to support embeddings
before weighting. The query still went through `query_proj`. Prototypes and
queries were therefore in different learned spaces; both are 256-d, so nothing
raised. The final `src/tcwpn/model.py` **removed the BiGRU encoder entirely**
(correction 1 in its header: a K-shot support set is K notes from K *different*
patients, so a recurrent pass over it models an ordering that does not exist).
Building the wrapper on the final model.py therefore removes the failure mode
structurally — there is no encoder left to forget to call. `Phase 7`'s
`test_support_order_does_not_change_the_result` guards against it returning.

**2. The `0.9635` figure cannot travel to the new deployment.** The review
treated `/health`'s `test_auroc_patient_level: 0.9635` as an acceptable
improvement over a naked AUROC — reasonable given only the Space was visible.
But §17 requires that a published metric be traceable to the checkpoint serving
it, and `AUDIT_OF_ARCHIVED_PIPELINE.md` documents that the pipeline behind that
number had a keyword gate applied to positives only (V1), a test split filtered
by a text-derived score (V2), and support/query patient overlap (V3). The clean
five-seed benchmark for `tcwpn_full` K=5 is **AUROC 0.7377, SD 0.0031**
(`data/TC-WPN — Phase 3B .../tcwpn_full_seed_results.csv`).

So `deployment_config.json` carries the clean figures, `research_metrics.metrics_verified`
starts `false`, and `/health` publishes nothing until it is set `true` by hand.
This is §15.7 and §17 applied, not a departure from them.

Likewise `THRESHOLD = 0.4036` does not carry over. It was locked on the archived
validation split; the clean `tcwpn_full_k5_seed42` run locks **0.2693**
(`eval_test.json`). `export_checkpoint.py` reads `locked_threshold` from the run
manifest rather than letting anyone retype it.

---

## Phase 1 — Freeze TC-WPN

Determine the exact final TC-WPN model. Not the old model, not the Space model,
not whichever checkpoint happens to be in HF.

- [ ] Confirm `src/tcwpn/model.py` at `main` is the final implementation.
      It is `PrototypicalModel` + `build_model`, with `ABLATION_PRESETS`.
- [ ] Confirm the preset. `tcwpn_full` = `(temporal=True, pcw=True,
      learn_temperature=True, aux=0.3)`, per `configs/tcwpn_full.yaml`.
- [ ] Commit everything and note the SHA. `vendor.sh` refuses to run on a dirty
      tree — vendoring from uncommitted code makes the recorded hash a lie.

**Decision to record here:** which preset is deployed. `tcwpn_full` is the
configuration named in the paper. `aux_only` is statistically indistinguishable
from it (Phase 4: mean Δ +0.0006, paired *t* p = 0.886). Deploying `tcwpn_full`
is defensible; the service must simply not attribute its score to w^T or w^C.

## Phase 2 — Verify the checkpoint

Confirm checkpoint ↔ architecture ↔ tokenizer ↔ preprocessing ↔ evaluation config.

- [ ] `scripts/train.py` saves `{"model", "cfg", "k", "seed", "step"}`. The
      `cfg` block is the training config, so the checkpoint carries its own
      architecture and can be compared rather than assumed.
- [ ] `app.py::_verify_architecture` does that comparison at startup and raises
      `DeploymentIntegrityError` on mismatch (§15.6).
- [ ] `tc_wpn/preprocessing.py` holds a verbatim `chunk_tokenize`. Confirm the
      run used `--max-len 512 --max-chunks 1 --stride 128` (the repo defaults,
      and what the frozen benchmark used) and that
      `deployment_config.json → preprocessing` matches.

## Phase 3 — Build the deployment wrapper

```bash
cd tcwpn_test
bash deployment/huggingface/vendor.sh
```

Copies `src/tcwpn/model.py` → `deployment/huggingface/tc_wpn/model.py`, writes
`__init__.py` with the vendored commit, and stamps
`provenance.tcwpn_git_commit` and `model_version` (`tcwpn-clean-benchmark-<short>`)
into `deployment_config.json`.

`app.py` calls the model's own `build_prototype()` and `classify()` rather than
re-implementing them. That is not stylistic: a re-implementation is exactly what
diverges silently, and it is what Phase 7 exists to catch.

## Phase 4 — Upload the exact checkpoint

```bash
python deployment/huggingface/scripts/export_checkpoint.py \
    --run-dir results/psych_mimic4idx/tcwpn_full_k5_seed42 \
    --out-name tcwpn_clean_v1_full_k5_seed42.pt \
    --dry-run          # inspect first
```

Then without `--dry-run`. It verifies architecture, computes sha256, uploads to
`dulharakaushalya/tc-wpn-clinical` (private), and writes
`checkpoint_filename`, `checkpoint_sha256`, `hf_model_repo_revision` and
`operating_point.threshold` back into `deployment_config.json`.

> **Blocker.** `find tcwpn_test -name "*.pt"` returns nothing. No checkpoint
> from the clean pipeline is committed anywhere in the repo. Phase 4 cannot run
> until the `tcwpn_full k=5 seed=42` run is re-executed or its `best.pt`
> retrieved from the Kaggle session that produced it.

## Phase 5 — Update the Space

Push only `deployment/huggingface/` contents to the Space root, plus the
existing `accounts.py` and `mailer.py`.

- [ ] Delete `tc_wpn/models/core.py`, `core_v3.py`, `embedder.py`,
      `embedder_old.py`, `patient_level_eval.py` from the Space. With two
      `core*.py` files present it is not determinable from the repo which one
      the checkpoint matches.
- [ ] Space secrets: `HF_TOKEN`, `REQUIRE_AUTH=true`, `ALLOWED_ORIGINS=<clinician app origin>`,
      `STRICT_STARTUP=true`.
- [ ] Confirm `GET /health` returns `"status": "ok"` and an empty
      `startup_errors`. If provenance is incomplete the Space will not start —
      that is intended.

## Phase 6 — Contract test

```bash
REQUIRE_AUTH=false pytest deployment/huggingface/tests/test_contract.py -v
```

Asserts the §11 response contract, the `MISSING_SUPPORT_SET` envelope, the
one-sided-support refusal, `model_version` / `preprocessing_version` /
`inference_configuration` presence, that no research metric leaks into a
prediction, that AUROC is not filed under the operating point, that the four
bands are named `application_risk_band`, that attention output is qualified,
and that no fusion vocabulary appears in a TC-WPN response.

## Phase 7 — Compare local vs HF

```bash
REQUIRE_AUTH=false pytest deployment/huggingface/tests/test_local_vs_hf_parity.py -v

TCWPN_SPACE_URL=https://dulharakaushalya-tc-wpn-demo.hf.space \
TCWPN_SPACE_TOKEN=... \
pytest deployment/huggingface/tests/test_local_vs_hf_parity.py -v
```

Two comparisons:

**A. research path vs deployment path**, same process, same weights. Runs
`PrototypicalModel.forward` over the same episode as `app.run_inference` and
requires the probability to agree to 1e-5, the support weights to agree
elementwise, and the result to be deterministic and order-invariant. This is the
CI test.

**B. deployment path vs live Space.** Checks `model_version`,
`preprocessing_version`, `tcwpn_git_commit` and the observed checkpoint sha256
match *before* comparing numbers — a probability match between two different
builds proves nothing — then requires the probability to agree to 1e-4.

## Phase 8 — Integrate the other models

Only after 1–7 pass. TC-WPN exposes one endpoint; the central backend calls it
alongside the other component services and performs fusion itself.

```
TC-WPN ──► text prediction ──┐
speech ──► speech prediction ─┤
physio ──► physio prediction ─┼──► CENTRAL BACKEND ──► fusion ──► final output
phone  ──► behaviour prediction┘
```

Nothing in `deployment/huggingface/` computes a composite. `test_contract.py`
enforces it: a `/predict` response containing `composite`, `fusion`,
`multimodal` or `final_risk` fails the suite.

---

## §17 — release record

Fill this in at release and keep it with the paper. It is the answer to *"how do
you know the deployed model is the model you evaluated?"*

| Step | Value |
|---|---|
| TC-WPN git commit | |
| Training config | `configs/tcwpn_full.yaml` |
| Training command | `python scripts/train.py --config configs/tcwpn_full.yaml --k 5 --seed 42 --stem psych_mimic4idx` |
| Run name | `tcwpn_full_k5_seed42` |
| Checkpoint file | |
| Checkpoint sha256 | |
| Locked threshold (validation, objective f1) | |
| Test AUROC (threshold-free) | |
| HF model repo @ revision | `dulharakaushalya/tc-wpn-clinical@` |
| Space commit | |
| `model_version` | `tcwpn-clean-benchmark-` |
| `preprocessing_version` | |
| Phase 6 pass | |
| Phase 7A / 7B pass | |

Every row is also served live at `GET /health`.

---

## One claim the service must not make

Phase 4 of the experimental record, five seeds, paired against `aux_only`
(0.7371): `temporal_aux` +0.0006 (p = 0.906), `pcw_aux` −0.0080 (p = 0.150),
`tcwpn_full` +0.0006 (p = 0.886). Neither w^T nor w^C shows a detectable effect
once the auxiliary CE head is held constant.

`temporal_weighting_used: true` in the response therefore means *the mechanism
is switched on in this build*, and nothing more. Neither the API documentation,
the clinician app copy, nor the paper may attribute the score to the temporal or
consistency mechanism.
