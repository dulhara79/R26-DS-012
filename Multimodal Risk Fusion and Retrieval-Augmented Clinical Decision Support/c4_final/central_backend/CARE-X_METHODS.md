# RAGF-Audit / CARE-X
### Sufficiency-grounded explanation for reliability-gated multimodal fusion

Explanation layer for the RAGF composite.
Component 4 · R26-DS-012 · `explain.py` · `care-x-v1.1`

---

## 1. The problem

Reliability-weighted late fusion is now the standard approach for combining
heterogeneous clinical modalities. ACE (arXiv 2607.20742, 2026) reweights each
modality by reliability before fusion and emits a global trust score. QMF, PDF,
Cred-MF and TMC do variants of the same. This project's RAGF engine belongs to
that family.

Every one of those systems treats **the weight vector itself as the
explanation**. A clinician is shown "clinical notes 0.64, physiological 0.36"
and left to interpret it.

That is not an explanation, for a reason TIER-MoE (arXiv 2607.27289, 2026)
states precisely. Conventional handling

> conflates two distinct cases: a modality that is unreliable for the current
> subject and one that is weak in isolation but complementary in fusion […]
> global modality strength does not describe subject-level reliability, and
> subject-level reliability alone does not measure the complementary value of a
> modality.

A weight of 0.64 means something entirely different when three other modalities
were present than when two were missing. Current systems render both
identically. TIER-MoE addresses this at the *modelling* layer, by learning
subject-specific risk. **No published work addresses it at the explanation
layer** — nobody tells the clinician which of the two situations they are
looking at.

Meanwhile, established clinical XAI is aimed elsewhere. SHAP, LIME, Grad-CAM
and DiCE explain **feature → prediction inside one model**. None of them explain
**reliability → weight → tier across models**. Applying SHAP to a fusion layer
would approximate quantities the fusion layer already records exactly.

---

## 2. Contribution

CARE-X is a **provenance-complete, counterfactually-grounded explanation layer
for reliability-gated late fusion**. Five layers, each targeting a documented
gap.

### L1 · Attribution *(infrastructure)*
Each modality's share of the composite, read from the stored contribution.
Reported only for modalities inside the composite: a modality with zero weight
has *no share*, which is not the same as a share of zero.

### L2 · Weight provenance *(infrastructure)*
Decomposes each weight into its causes — measured discrimination, this
reading's confidence and coverage, and renormalisation after gating — so a
clinician is not left to assume the first cause alone. Where the fusion service
applies `c = 0.5 + 0.5·confidence·coverage`, that multiplier is surfaced.

### L3 · Exact counterfactuals *(supporting, not claimed as novel)*

Solving a linear equation for a threshold crossing is arithmetic, not a method.
It is included because it is the machinery L4b needs, and because the exactness
matters in use — but the comparison against DiCE is not a fair contest: DiCE
searches because trees are non-differentiable, so "we did not have to search" is
a property of the model class, not of this work.

Because the fusion is closed-form —

```
composite = Σ_m  w_m · h_m ,   Σ w_m = 1 over the usable set
```

— counterfactuals are **solved, not searched**:

- **Drop-one**: removing modality *m* and renormalising the survivors is exact
  arithmetic on the stored weights, `w'_k = w_k / (1 − w_m)`.
- **Tier flip**: the composite is linear in each `h_m`, so the value at which it
  crosses cut-point *T* solves in one step:

  ```
  h_m* = ( T − Σ_{k≠m} w_k · h_k ) / w_m
  ```

  If `h_m* ∉ [0,1]`, that modality **cannot** flip the tier at any achievable
  value. This is a guarantee, not an observation.

This is a categorical advantage over DiCE, which searches a space and returns a
possibly-local optimum. Here the answer is unique and derived in closed form.

### L4a · Quantified weight inflation — *primary contribution*

RAGF sets `w_m ∝ base_m · rel_m`, renormalised over the usable set *U*.
Comparing that to the weight the modality would carry in a full panel at ideal
reliability gives a ratio that **factors exactly**:

```
inflation(m) = w_m^obs / w_m^ideal
             = ( Σ_ALL base / Σ_U base )     ← scarcity, panel-level
               × ( rel_m / mean_rel_U )      ← relative reliability, per-modality
```

where `mean_rel_U` is the base-weighted mean reliability of the survivors. This
separates *how much of a weight was earned* from *how much was inherited from
absent modalities* — the TIER-MoE conflation, made numeric.

**This is not a cosmetic upgrade over a threshold rule; the two disagree on real
data.** An earlier heuristic version (`weight ≥ 0.5 AND confidence < 0.25`)
flagged clinical notes as scarcity-inflated. The measured ratios say the
opposite:

| Modality | observed w | full-panel ideal | inflation | = scarcity × rel. reliability |
|---|---|---|---|---|
| physiological | 0.3649 | 0.2751 | **1.33×** | 1.204 × 1.101 |
| clinical notes | 0.6351 | 0.5551 | **1.14×** | 1.204 × 0.950 |

Clinical notes dominates because its discrimination is genuinely higher
(AUROC 0.8989 vs 0.6977); its weight is *suppressed* relative to peers by weak
reliability. The physiological modality is the more inflated of the two. A
hand-picked cut-off encodes an intuition; this encodes the arithmetic, and the
intuition was wrong.

Permanently-excluded modalities are omitted from `Σ_ALL`: a modality that can
never enter the panel is not a missing one, and counting it would inflate the
scarcity term forever.

When base weights for the full eligible panel are unavailable, the scarcity term
is **not recoverable** from a stored fusion result — it depends on the base
weight of modalities that are absent. The layer reports `quantified: false` with
a stated reason rather than substituting a guess.

### L4b · Necessity, sufficiency, decision relevance — *primary contribution*

**Attribution magnitude and decision relevance are orthogonal, and only the
second is actionable.** In the worked example the physiological modality
produces 42.5% of the composite — substantial by any attribution measure. A
SHAP ranking would call it important. Four questions separate what that share
actually means:

| Property | Definition |
|---|---|
| **necessary** | removing it changes the tier (others at actual values) |
| **alone-sufficient** | if it were the *only* modality, the tier is unchanged |
| **decision-relevant** | some `h_m ∈ [0,1]` changes the tier, others held fixed |
| **minimal sufficient set** | smallest *S* whose actual values pin the tier regardless of what the modalities outside *S* report |

The last is the **sufficient reason** of abductive XAI (Marques-Silva, Ignatiev,
Darwiche), which carries a logical guarantee that a SHAP ranking does not.
Because the composite is linear in each `h` given a fixed usable set, its
extremes over the free variables are attained at `h = 0` and `h = 1`, so each
candidate set is checked in one step rather than searched.

On the worked example: neither modality is necessary, both are alone-sufficient,
and the minimal sufficient set is **both together** — no single one pins the
tier while the other is free to vary. That is a materially stronger clinical
statement than any weight vector.

**Scope.** This holds the usable *set* fixed. Sound for score-valued
counterfactuals, because the gate keys on status, freshness and coverage rather
than on the score. **Not** sound for counterfactuals over confidence, coverage
or freshness: those can eject a modality, triggering renormalisation that moves
every other weight discontinuously. The true input→output map is piecewise, not
linear, and reasoning across that gate boundary is stated as out of scope rather
than silently assumed away. It is the natural next contribution.

### L5 · Epistemic honesty ledger *(engineering practice, not a novel concept)*
Every explanation carries what the system does **not** know: placeholder
reference distributions, uncalibrated component probabilities, unreported
coverage, distribution drift, conformal calibration status, unrecorded model
versions. Each entry names the limitation, what it affects, and what would
resolve it.

Model Cards (Mitchell et al. 2019), FactSheets and Datasheets already cover
this territory; the contribution here is per-assessment granularity and
machine-readable resolution paths, not the concept.

---

## 3. Design rules

1. **Deterministic.** No language model, no sampling. The same fusion result
   must always produce identical text — a clinician reopening yesterday's
   assessment and seeing different reasoning would stop trusting the number.
   Mirrors the rule `support_bank.py` applies to support selection.

2. **Stored data only.** No component is re-called. An explanation must be
   reproducible months later, when those Spaces may no longer exist.

3. **A gap is stated, never filled.** Where a value is unknown the explanation
   says so. Same rule `modality_clients.py` applies to a timeout: absent is
   absent, never zero.

4. **Two audiences, one source.** The clinician view carries full detail; the
   patient view carries band-level reassurance only, and is tested to leak no
   score, weight, or modality name.

---

## 4. Validation

`validate_explain.py` — **95 assertions**, independent of the module's own
helpers (a shared helper would let one bug pass both).

| Section | Checks |
|---|---|
| Real deployed fusion result | 9 |
| Drop-one vs hand arithmetic | 5 |
| Tier-flip vs hand arithmetic | 6 |
| Reliability audit (measured verdicts) | 7 |
| Honesty ledger | 6 |
| Determinism (5 identical runs) | 2 |
| Edge cases | 11 |
| Safety — refusal ≠ low risk | 4 |
| Patient-safe leakage | 2 |
| Inline summary | 13 |
| Monotonicity | 1 |
| **Quantified inflation vs hand algebra** | **13** |
| **Sufficiency / necessity / decision relevance** | **15** |

Counterfactuals are verified against independently hand-computed values:

```
c3 High-flip:  (0.67 − 0.3649×1.0) / 0.6351 = 0.4804     ✓
c1 High-flip:  (0.67 − 0.6351×0.7783) / 0.3649 = 0.4815  ✓
drop c1:       renormalised → h_c3 = 0.7783, Δ = −0.0809 ✓
scarcity:      (0.1977+0.3989+0.1220)/(0.1977+0.3989) = 1.204  ✓
c1 inflation:  0.3649 / 0.2751 = 1.326×                  ✓
c3 inflation:  0.6351 / 0.5551 = 1.144×                  ✓
decomposition: scarcity × rel_factor == inflation, both modalities  ✓
```

The decomposition also reconstructs the deployed weights from first principles
(0.3650 / 0.6350 against the live 0.3649 / 0.6351), independently confirming
that base weights are `(AUROC − 0.5)` and reliability is `0.5 + 0.5·conf·cov`.

**Adversarial fuzzing.** 5,000 randomised malformed inputs × 4 seeds
(20,000 total), re-run after the v1.1 layers were added — NaN, ±infinity, strings where floats belong, nulls where
objects belong, unknown modality keys, empty and absent fields. **Zero
crashes.** Two real defects were found and fixed during this process:

1. A chained `.get()` crashed when a key existed with an explicit `null`
   (`.get(k, {})` does not fire its default when the key is present).
2. `x or {}` failed to guard against a non-empty *string* where a dict was
   expected — truthy, so it passed the guard and raised on `.get()`.

Both are now handled by a type-checking `_d()` coercion. `_f()` additionally
rejects NaN and ±inf, because NaN fails every ordering comparison silently and
would make the explainer report a tier as unreachable when the true problem is a
corrupt input.

---

## 5. Endpoints

| Endpoint | Contents |
|---|---|
| `GET /v1/doctor/patients/{id}/explanation` | Full five-layer explanation |
| `GET /v1/doctor/patients/{id}/timeline` | Compact `explanation` block inline |

The timeline carries only a headline (~110 chars) plus flags, so the full
object is not re-serialised on every poll and does not bury the clinical data
it annotates.

---

## 6. Worked example (real deployed output)

Input: composite 0.8592, tier High, band RED, from C1 (physiological,
harmonised 1.0, weight 0.3649) and C3 (clinical notes, harmonised 0.7783,
weight 0.6351). C2 excluded by policy; C4 absent.

**Headline (inline):**
> High (0.86) from 2 of 4 modalities; clinical notes drove 57.5%.
> Caution: clinical notes is scarcity-inflated.

**Tier-flip points (exact):**

| Modality | Target | Required `h` | Current | Reachable |
|---|---|---|---|---|
| physiological | High | 0.4815 | 1.0000 | yes |
| physiological | Medium | −0.4229 | 1.0000 | **no** |
| clinical notes | High | 0.4804 | 0.7783 | yes |
| clinical notes | Medium | −0.0392 | 0.7783 | **no** |

Reading: neither modality can pull this patient below the High band on its own.
Both would have to fall together. That is a stronger and more actionable
statement than any weight vector.

**Honesty ledger:** 3 items — placeholder reference distributions (high),
conformal not calibrated at n=0 verdicts (high), C1 model version unrecorded
(low).

---

## 7. Limitations

- **L3 and L4 inherit the reference distributions.** While those remain
  synthetic placeholders, harmonised values — and therefore every counterfactual
  computed from them — are internally consistent but not calibrated to a real
  cohort. The ledger states this on every response rather than leaving it to a
  footnote. Replacing the three reference sets with real held-out scores is the
  single highest-value outstanding task.

- **Inflation requires base weights for the full eligible panel.** These are not
  recoverable from a stored fusion result, because the scarcity term depends on
  the base weight of *absent* modalities. Without them the layer reports
  `quantified: false` and every verdict falls back to `unquantified` rather than
  guessing.

- **The inflation bands (≥1.5 inflated, ≥1.15 moderate, <0.85 suppressed) are
  reasoned, not empirically tuned.** The ratio itself is principled; where to
  cut it for a verdict label is not, and should be validated against clinician
  judgement once verdict data exists.

- **Sufficiency holds the usable set fixed.** Sound for score counterfactuals,
  unsound across gate boundaries. Explaining across the gate discontinuity —
  "this assessment expires in four minutes and the tier will drop to GREY" —
  is the clearest remaining extension.

- **Tier cut-points are read from the fusion service where importable.** When
  they are not, the module falls back to defaults *and records that it did so*,
  so a mismatch surfaces rather than silently mislabelling counterfactual tiers.

- **No clinician evaluation yet.** The explanation's usefulness is argued from
  design principles, not measured. A structured review with Dr. Suraweera would
  be the natural next step.

---

## 8. References

- ACE — *Adaptive Confidence-weighted Expansion for Trustworthy Multi-Omics
  Multimodal Fusion*, arXiv 2607.20742 (2026)
- TIER-MoE — *Trust-Informed Expert Routing via Conditional Modality Risk*,
  arXiv 2607.27289 (2026)
- Lundberg & Lee — *A Unified Approach to Interpreting Model Predictions*
  (SHAP), NeurIPS 2017
- Mothilal et al. — *Explaining ML Classifiers through Diverse Counterfactual
  Explanations* (DiCE), FAT* 2020
- Ignatiev, Narodytska & Marques-Silva — *Abduction-Based Explanations for
  Machine Learning Models*, AAAI 2019
- Darwiche & Hirth — *On the Reasons Behind Decisions*, ECAI 2020
- Mitchell et al. — *Model Cards for Model Reporting*, FAT* 2019
