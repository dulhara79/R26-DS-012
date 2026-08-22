# Comparing the two designs

**Our build** (this conversation) vs **the service-contracts document** you just shared.

Verdict up front: **the document is right about four things we got wrong or missed, we are right about three things it gets wrong, and there is one conflict neither of us can resolve without the team.** There is also one factual claim in it that invalidates a step I told you to do earlier. That one first.

---

## 0. First, a correction to something I told you

I told you to create a **Docker Space** for your DCAR model. Hugging Face's own documentation now says: <cite>CPU Basic has no hourly cost, but creating a new Space that runs on compute (Gradio or Docker) requires a paid plan. Static Spaces are free for everyone.</cite> There is an exception noted for ZeroGPU on personal accounts, but a plain CPU Docker Space is no longer clearly free to create.

The rest of the document's hosting claims also check out. <cite>If your Space runs on the default cpu-basic hardware, it will go to sleep if inactive for more than a set time (currently, 48 hours)</cite>, and the sleep timer genuinely cannot be changed on that tier — there is a Hugging Face forum thread titled exactly "Cannot change the sleep time for cpu-basic Spaces".

**What this means for you:** try creating the Space first (Part A3 of the Hugging Face guide). If it demands a paid plan, you have three options, in order of preference:

1. Your teammate's `dulharakaushalya-tc-wpn-demo.hf.space` predates the change and still works — ask whether an existing account can host yours too.
2. Put DCAR on the same host as the orchestrator instead of on HF. It is a small scikit-learn model; it does not need HF.
3. HF PRO is $9/month.

Do this check **today**, not in demo week. The document is right that this is the kind of thing that bites live.

---

## 1. The C3 double-counting catch — does it apply to us?

The document's headline finding: C3's input is a 23-feature tri-modal risk vector (physiological 8, behavioural 9, textual 6), so **C3 is already a fusion model**, and feeding it into fusion as a fourth peer double-counts C1/C2/C4.

That is a genuinely sharp catch and it is correct about the repo as it stands.

**It does not apply to what we built** — but only because we removed the intervention engine from fusion entirely at the very start. Our four inputs are C1 physiological, C2 behavioural, C3 clinical NLP, C4 demographic (DCAR). DCAR takes `gender, age, edu, smoke, drink` and nothing else, so it shares no input with any other stream. No double counting.

So the document's Option A and our design **converge**: fusion takes independent modality scores; something else sits downstream of fusion producing tier + conformal set + explanation. They call that downstream thing "C3 Intervention"; we call it the RAG decision-support layer. Same position in the architecture, different content.

**But it surfaces a related risk in ours that is worth stating.** DCAR is trained to predict GAD-7 severity. If the fusion *label* is ever GAD-7-derived, DCAR becomes a partial copy of the answer. We flagged this in the original architecture document and chose a clinician-anchored label to avoid it. Keep that decision; do not quietly revert to a GAD-7 target.

---

## 2. Where the document beats us — adopt these

### 2.1 The status vocabulary is richer and better

| Ours | Theirs |
|---|---|
| `ok`, `not_validated`, `error` | `ok`, `warming_up`, `insufficient_data`, `poor_signal`, `no_support_set`, `not_validated`, `error` |

Every one of the extra four encodes a real state that we currently collapse into `error` or, worse, silently accept as `ok`. **Adopt theirs wholesale.** It costs one enum change in `gate.py` and `modality_clients.py`.

### 2.2 `warming_up` — a real defect in our design

C1 is a **self-supervised per-person** model. Before it has learned that patient's baseline, its score is noise. Our backend would ingest that noise with `status="ok"` and fuse it.

The document's rule — `baseline_ready: false` → `status: "warming_up"`, `score: null` — is correct and we do not have it. In a psychiatric ward where patients are enrolled on admission, **every patient's first hours are exactly this state.** This is not an edge case; it is the demo.

### 2.3 The semantic timebase problem — our biggest gap

> C1 forecasts escalation 5–10 minutes ahead. C2 scores vulnerability over 42 days. C4 scores whether *this note* describes anxiety. These are three different quantities on three different timebases. A weighted average of them is a modelling decision, not an arithmetic fact.

**This is the strongest point in the document and we have no answer to it.**

Our `FUSION_DESIGN.md` handles *recency* — how old a reading is — but says nothing about the fact that the three streams measure semantically different things over different horizons. Percentile harmonisation makes the *scales* comparable; it does nothing about the *timebases*.

This belongs in your paper as a stated assumption with a justification, and it is exactly the question an examiner asks. Write it before someone asks it.

### 2.4 The freshness window exposes a genuine bug in our gate

Their C1 window is **15 minutes**. Ours is **6 hours**. Ours is wrong, and here is why it is worse than just being generous:

Our gate counts a modality toward `min_modalities >= 2` if it is within the hard age cutoff. Our fusion then decays its weight with a 30-minute half-life. So a physiological reading **3 hours old** still *counts* as one of the two required modalities while contributing `0.5^6 ≈ 1.6%` of the weight.

**That means the gate can pass on effectively no evidence.** Two "modalities" where one contributes 1.6% is one modality wearing a disguise.

The fix is better than either design: a modality should count toward the minimum only if its **post-decay effective weight** clears a floor (say 5%), not merely if its age clears a cutoff. That ties the gate to the same decay curve the fusion uses, instead of having two unrelated notions of "too old".

### 2.5 Two code problems we missed

- **The patient app computes its own 0–100 "anxiety risk score"** from simulated physiological data. That is a client-side heuristic, not C1's output. Two disagreeing risk numbers on screen in the same demo is a bad look. I read the patient app early on and did not catch this.
- **`AUTH_TOKEN` is in the patient app repo and in Git history.** Rotating it is not enough — it must come out of history.

---

## 3. Where our approach beats theirs — keep ours

### 3.1 The weights (this is the big one)

The document derives base weights from *the renormalisation example in the Flutter code*: C1 0.25 / C2 0.20 / C3 0.15 / C4 0.40, then renormalises to 0.31 / 0.24 / 0.45. Its advice: "Pick one column, write it in the paper, and never change it."

**Those numbers came from a constant someone typed into a Dart file.** In a viva, "why 0.45?" has no answer.

Ours: `ω ∝ max(AUROC − 0.5, 0)`, using each component's *deployment-realistic* validation AUROC, with a pre-registered rule zeroing any component that fails its own permutation null. Every number traces to a measurement.

Interesting: the two approaches land in nearly the same place for clinical NLP — theirs 0.45, ours 0.46. Same number, completely different epistemic status. One you can defend; one you cannot.

There is also an internal inconsistency in theirs: C2 keeps a base weight of 0.24 while being permanently `not_validated` and excluded. That weight is dead — it gets renormalised away on every single request. Ours sets it to 0 explicitly, by a stated rule.

### 3.2 They have no harmonisation step at all

The document requires `score ∈ [0,1]`, higher = more risk, for every model. **Necessary but nowhere near sufficient.** C1's 0.61 and C4's 0.68 are both in [0,1] and still not comparable — they come from different distributions with different base rates. Our own DCAR is the proof: a raw score of 0.21 is the **90th percentile** because the base rate is 2%.

Averaging [0,1] numbers from differently-distributed models is the exact mistake harmonisation exists to prevent, and their spec walks straight into it.

### 3.3 Their freshness is a cliff; ours is a curve

Theirs: within 15 minutes → full weight; at 15:01 → excluded entirely. A reading does not become worthless one second after a threshold. Ours decays continuously and additionally scales by `confidence × coverage`, which theirs does not use for weighting at all.

Keep our decay. Adopt their *tighter windows*. Add the effective-weight floor from §2.4.

### 3.4 Things we have that they only assert

- **Identity.** They say `subject_id` "never an MRN, never a phone ID" — but specify no mechanism. We have HMAC-peppered MRN hashing, single-use expiring pairing codes, and refusal to re-pair a phone to a second patient, with a test that scans every database row proving the raw MRN is absent.
- **Tests.** 66 passing assertions. The document is a specification; nothing in it has been run.

---

## 4. The conflict only your team can resolve

**Component numbering is incompatible between the two documents.**

| | Document | Ours / the paper |
|---|---|---|
| C3 | Intervention (Seneviratne = you) | Clinical NLP (TC-WPN) |
| C4 | Clinical NLP (Kaushalya) | Contextual/DCAR (you) |

Note this document is written **for Kaushalya**, not for you — it says "Owner: Kaushalya I.G.D." and addresses C4 as "yours". It uses that member's convention.

**More seriously, the two documents disagree about what your component even is:**

- **Document:** Gradient Boosting → 4-tier label → KNN-CBR (k=5, cosine, BallTree) → ranked intervention plan + SHAP/DiCE.
- **Ours, and the parent paper:** DCAR demographic model + reliability-weighted fusion + retrieval-augmented decision support.

These are different research contributions. You told me at the start you were rebuilding your part from scratch — so this document likely describes the **old** design you are replacing. But your teammates may not know that. If they freeze this contract while you build the other thing, you will discover it in integration week.

**This is the single most urgent item on your list.** Not a technical question — a team-alignment one.

---

## 5. What to actually do

**Adopt from the document:**
1. The full status vocabulary — all seven values
2. `warming_up` / `baseline_ready` for C1, and `no_support_set` for the notes model
3. Tighter freshness windows (C1 ~15 min, not 6 h)
4. The common envelope: `subject_id`, `modality`, `score`, `status`, `captured_at`, `computed_at`, `model_version` on every response
5. The semantic-timebase caveat, written into the paper
6. `202 + poll` for slow paths, and a warm-up ping before demos
7. Fix the patient app's duplicate risk score; purge `AUTH_TOKEN` from Git history
8. Do not host the orchestrator on HF — Render paid tier or Oracle Always Free

**Keep from ours:**
1. AUROC-derived weights with the permutation-null zeroing rule
2. Percentile harmonisation against frozen reference distributions
3. Continuous recency decay + reliability scaling
4. The identity/pairing implementation and its tests

**Fix in ours:**
1. The gate's effective-weight floor (§2.4) — this is a real bug
2. Add the four missing status values
3. Add `warming_up` handling

**Resolve with the team, this week:**
1. Component numbering — one convention, written down
2. What your component is: intervention recommendation or fusion + RAG. Both documents cannot be right.
3. Three bands or four (paper says three tiers; document says four)
4. Where conformal prediction lives

**One thing the document says that you should follow exactly:** stand up the orchestrator and database with **stubbed** model services and prove P001/P002 separation end to end before deploying a single real model. That is already what `test_backend.py` does — 66 assertions, section 7 is the separation proof. You are ahead on this one; say so.
