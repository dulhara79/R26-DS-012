# Paper alignment — what changes, what's confirmed, what needs a decision

Source: *"A Multimodal Digital Biomarker Framework for Personalized Vulnerability
Mapping and Acute Escalation Forecasting in Young Adults with Anxiety Disorders"*
(Kaushalya, Layathma, [3rd author], Seneviratne, Weerasinghe, Thelijjagoda).

Read start to end against everything built (fusion_service, central_backend,
121 passing tests). Organized so you can act on it in the right order: the one
decision that must come first, then safe fixes already applied, then new work
that's needed regardless of that decision, then everything else confirmed
correct or paper-only.

---

## 0 · THE ONE THING TO DECIDE BEFORE ANYTHING ELSE

**§III.D says:** *"A missing modality is imputed with that modality's
population mean, computed once from the training cohort and frozen prior to
evaluation, with a fixed 0.05 confidence penalty per imputed value... the
behavioral score is not supplied to the fusion model and is instead treated as
a permanent instance of this same missing-modality case rather than a
special-cased exclusion."*

**What's built does the opposite, on purpose, and it's currently one of your
system's non-negotiable rules:** a missing or failed component is
`status="error"`, `score=null` — **never a number, never imputed.** Rule 2 of
that same list: demographics-only (day one, before physio or notes exist)
returns `tier=null` with a stated reason, **never** a tier computed from
whatever's available plus filled-in guesses.

These two designs are not reconcilable by a small patch. Population-mean
imputation means the pipeline can ALWAYS produce a tier — there's no state
where evidence is "insufficient," because a missing modality just becomes the
average patient's value with a small confidence haircut. That erases the
day-one guard entirely: under the paper's literal wording, a patient with
nothing but a submitted demographic form would get physio and text imputed at
the population mean and the classifier would output a tier anyway.

**I have not implemented this.** Not because it's a bad idea in the
abstract — population-mean imputation with a confidence penalty is a
legitimate, common technique — but because switching to it silently would
reverse a decision you made deliberately, for a stated clinical reason: a
system that always answers, even from nothing, will eventually tell a
newly-admitted patient "Low risk" based on population averages, and that's the
exact failure `DECISIONS_LOCKED.md` and the day-one guard exist to prevent.

**Three ways this resolves, and they lead to different code:**

1. **The paper's wording is imprecise** — written before the deployed system's
   stricter rule was finalized, or written to describe the *offline evaluation
   protocol* (where imputation is a normal, defensible way to handle missing
   features in a fixed dataset) rather than the *live clinical deployment*
   (where "insufficient evidence" is a real and important state). If this is
   it, the paper's methods section needs a sentence distinguishing offline
   training/evaluation (imputed) from live deployment (gated, no imputation) —
   and no backend code changes.
2. **The team wants imputation in deployment too** — then the day-one guard,
   the hard age cutoffs, the effective-weight floor, and the masked
   renormalisation all get replaced by: always compute, apply the 0.05
   penalty per imputed value, never return `tier=null`. This is a real rewrite
   of `gate.py` and `fusion.py`, not a small patch, and it needs to be a
   decision made with the same weight as the original one, not something that
   happens because a paper draft said so.
3. **Split the difference** — imputation is fine for modalities that have
   *some* population prior worth using (arguably C2, since it's permanently
   absent by design), but the physiological/text streams still gate on
   presence for a genuinely new patient. This is a middle path but needs
   someone to specify exactly where the line is, or it's just option 2 with
   extra steps.

I'm not picking one of these for you. Tell me which, and I'll build it.

---

## 1 · Already fixed — safe, no architecture risk (121/121 still passing)

### 1a. C1 feature count: 11 → 10
The paper lists exactly ten features by name (§III.A): mean HR, mean RR
interval, SDNN, RMSSD, mean breathing rate, breathing-rate variability, mean
temperature, temperature variability, mean acceleration magnitude,
acceleration variability. `main.py`'s `PhysiologicalWindow.features` docstring
said 11 — a leftover from before the paper's exact list was available. Fixed
to list all ten by name.

### 1b. C3's "confidence" field flagged as unconfirmed, not silently trusted
The paper states its analogous quantity — the per-support-note weight used to
build a class prototype — is **"referred to as prototype consistency rather
than confidence because it is neither calibrated nor an estimate of label
uncertainty"** (§III.C). `modality_clients.py`'s `call_c3` currently reads the
live API's `confidence` field and feeds it into fusion's reliability weighting
as if it *were* calibrated. I did not change that number — I don't know what
the right replacement is, and guessing would be worse than leaving it
unconfirmed. I added a docstring flag (item **C3-2** below) and left the
number itself untouched pending an answer from Dulhara.

---

## 2 · New work needed regardless of how §0 resolves

These are things the paper describes that genuinely don't exist yet, and
building them doesn't depend on the imputation decision — they slot in either
way.

### 2a. XGBoost + Platt scaling + SHAP as the tier classifier
**§III.D:** *"The fused risk score, together with demographic covariates and
interaction history, is passed to a gradient-boosted tree classifier
(XGBoost) producing a three-level vulnerability tier (Low/Medium/High), post
hoc calibrated via Platt scaling and explained per-prediction via SHAP."*

This is a real pivot for your component specifically. DCAR — the standalone
cumulative-ordinal demographic model with its own isotonic calibration and
GAD-7 population-prior framing — does not appear anywhere in this paper.
Demographics enter as **raw covariate features into the classifier**, not as
an independently-scored, independently-weighted fourth modality in a linear
fusion. Practically: `fusion.py`'s AUROC-weighted percentile combination
becomes **stage 1** (combining C1 + C3 into one "fused risk score" — this part
can stay close to what's built, restricted to two modalities instead of
four), and a **new stage 2** takes that score plus demographic covariates plus
interaction history into an XGBoost classifier.

Needed to build stage 2: a training pipeline, labelled data (fused score +
demographics + interaction history + tier label — likely from the `verdicts`
table once you have enough), Platt scaling calibration, and a SHAP explainer
wired to return per-prediction feature attributions. None of this exists yet.

**Question worth asking before building it:** does DCAR's work — the Zenodo
training, the cumulative ordinal modeling, the isotonic calibration — become
throwaway, or does it get repurposed as the *feature engineering* step that
produces "demographic covariates" for the classifier (i.e., DCAR's population
prior becomes one input feature among several, rather than the fusion peer it
currently is)? Worth confirming rather than assuming either way.

### 2b. Case-based retrieval (separate from CARE-AnxRAG — CARE-AnxRAG doesn't cover this)
**§III.D:** *"Three retrievers run in parallel over raw features, an
autoencoder latent representation, and a SHAP-weighted feature space, fused
via Reciprocal Rank Fusion. Retrieved cases are reranked and split into
supporting and contrasting evidence... An evidence-sufficiency gate evaluates
candidate count, similarity, inter-retriever agreement, and classifier
confidence before generation proceeds."*

Worth being precise about scope here: **CARE-AnxRAG, per its own documented
contract, takes only a question and returns an answer grounded in its
WHO/PubMed-sourced guideline corpus — it does not do comparable-patient-case
retrieval at all.** The case-retrieval piece described here is a *different*
capability than what CARE-AnxRAG provides. This was actually built once
already (in the `rag.py` that got removed when CARE-AnxRAG was integrated) —
but that removal was correct at the time, because at that point it looked like
CARE-AnxRAG was replacing the whole RAG layer. It wasn't; it only replaces the
guideline half. **The case-retrieval half still needs to exist, and needs to
be rebuilt to match the paper's exact retriever definitions:**

- raw feature-space distance (this part is close to what the old `rag.py` had)
- **autoencoder latent representation** — presumably reusing C1's own LSTM
  autoencoder's latent space, though that's not exposed via C1's current API
  at all. Needs clarification: does this mean asking Dewdu to expose an
  internal representation, or is there a different autoencoder meant here?
- **SHAP-weighted feature space** — depends on 2a existing first, since you
  need a trained classifier with a SHAP explainer before you can weight a
  feature space by SHAP values.

So the practical order is: 2a before 2b, since 2b's third retriever needs 2a's
output to exist.

Synthetic/oversampled cases must stay excluded from the case base — the paper
independently confirms why this matters (§IV): *"an earlier index evaluated on
the balanced set reported 95.1% top-1 accuracy; recomputing on the
natural-distribution cohort yielded 76.2%, reflecting synthetic
near-duplicates inflating apparent performance."* The old `rag.py`'s
`SYNTHETIC_TRIGGERS` exclusion was the right instinct — keep it when this gets
rebuilt.

---

## 3 · Confirmed correct — no change needed, just noting the paper agrees

Worth knowing these numbers now trace to a citable source, not just something
said earlier in a chat:

| What | Already in the build | Paper confirms |
|---|---|---|
| C1 weighting AUROC | 0.6191 (AffectiveROAD, recalibrated) | §IV: *"recalibration using the first 20% of each drive improved performance to AUROC 0.6191"* — exact match |
| C2 exclusion justification | AUROC 0.5205 vs null 0.4991, p=0.255 | §IV: *"AUROC of 0.5205 (95% CI 0.485–0.560) against a 50-permutation null of 0.4991 (p = 0.255)"* — exact match, and now with a CI and six pre-registered candidate explanations tested and excluded |
| C3 weighting AUROC | 0.7380 | §IV: *"mean AUROC of 0.7377 (SD = 0.0031)"* across 5 seeds — matches to 3 decimal places |
| Tier always re-derived server-side | built this way from the start | §III.D: *"The tier is always re-derived server-side rather than accepted from the client, preventing a compromised client state from bypassing downstream checks"* — exact match |
| Synthetic cases excluded from case evidence | `SYNTHETIC_TRIGGERS` in the old `rag.py` | §III.D + §IV, independently justified by the 95.1%→76.2% finding above |
| C2's retrained SHAP importance | excluded/weight 0 | §IV: *"behavioral risk's importance was confirmed at 0.000"* after retraining with confirmed disjoint SMOTENC-balanced cohort |

---

## 4 · Open items requiring team clarification (not blocking, but flag them)

- **C3-2** (from §1b above): is the live API's `confidence` field the same
  quantity the paper calls prototype consistency, and if so, should it still
  feed fusion's reliability weighting given the paper's own disclaimer that
  it's uncalibrated? Ask Dulhara directly.
- **Autoencoder latent representation for case retrieval** (§2b): whose
  autoencoder, and how does central_backend get access to its latent space
  given C1's current API only returns a final risk index?
- **"Interaction history" as a classifier feature** (§2a): not defined
  anywhere in what's built. Needs a concrete schema before stage 2 can be
  built — what counts as an interaction, where is it stored, over what
  window?
- **C1 horizon precision**: earlier documentation said "5–10 minutes ahead"
  (from an older draft); the paper is more precise — 19 minutes of history
  forecasting a 10-minute-ahead trajectory (§III.A, §III.E). Worth updating
  comments for precision, low priority, not blocking.

---

## What I'd suggest doing next, in order

1. **Answer §0.** Everything else is genuinely independent of it, but it's the
   thing most likely to change other decisions once it's settled, so resolve
   it first.
2. **Send Dulhara the C3-2 question** — quick to ask, unblocks a real accuracy
   concern in the fusion weighting.
3. Once §0 has an answer, I can either (a) leave the current gate/fusion logic
   untouched and start on 2a (XGBoost + Platt + SHAP) as a genuinely additive
   stage, or (b) rebuild `gate.py`/`fusion.py` around imputation first, then
   layer 2a on top. Which one depends entirely on §0's answer.
