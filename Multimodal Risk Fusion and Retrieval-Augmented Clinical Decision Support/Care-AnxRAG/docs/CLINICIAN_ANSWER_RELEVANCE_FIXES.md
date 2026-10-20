# Psychiatrist-facing answer relevance: viva corrections

These deterministic restrictions address the seven-case engineering audit after the
2026-10-16 machine-screened PubMed expansion. They do NOT establish clinical
validity, answer usefulness, or benchmark accuracy.

## Observed failure modes

- P01: a GAD treatment question returned definitions/symptoms instead of treatment results.
- P02: social-anxiety CBT question abstained. Do not relax the requirement for an
  explicit, matching treatment/subtype sentence merely to force an answer.
- P03: exposure-versus-CBT comparison abstained. Require a genuine direct
  comparative finding; do not synthesize a head-to-head result from separate studies.
- P04: recent treatment review question returned an old 2021 meta-analysis and
  unrelated sleep/microbiome topics. Restrict by requested review evidence type,
  *publication date*, and an explicit treatment finding.
- P05: WHO medication statement was accompanied by COPD-specific and methods-only
  literature. Restrict requests for authoritative medication guidance to guideline
  or official health information with actual medication findings.
- P06: 'contradicts' question returned nonconflicting objectives and limitations.
  Require explicit high-confidence contradiction relation connecting two distinct
  retained studies, plus eligible findings; otherwise abstain.
- P07: panic diagnostic question returned treatment/prevalence facts instead of
  diagnostic assessment or criteria. Require explicit diagnostic statements.

## Scope and limitations

This is extractive, sentence-level answer gating layered **after** CARE hybrid
retrieval/re-ranking. Citation verification still requires source-substring support.
There is no additional free-text medical generation, invented guidelines,
unsupported summaries, or weakening of crisis and personal-diagnosis routing.

The recent-review window is currently a transparent heuristic of the previous
three years (based on *published_at*, not *updated_at*). This is not a clinical
definition of a current guideline, and must not be used to decide clinical
standards of care.

The selected sentences and related sources still need psychiatrist review.
A valid citation is not proof that a clinical question has been answered.
Ingestion remains machine-screened and non-adjudicated. The 47 unresolved staged
versions still block a formal corpus freeze.

## Crucial local-code verification

The research workstation had uncommitted modifications to query.py, rag.py,
retrieval.py and safety.py. A successful GitHub CI or a merged PR does not prove
that the Mac's `care-anxrag` command uses the new code. Before repeating P01-P07,
inspect `git status --short` and `git rev-parse HEAD` and use the original
workspace only after backed-up reconciliation.

Example diagnosis:

    python - <<'PY'
    import inspect
    from care_anxrag.query import QueryAnalyzer
    import care_anxrag.answer_selection as selection
    from care_anxrag.question_requirements import is_review_request
    q = "What treatments are supported for generalized anxiety disorder?"
    print("Query intent:", QueryAnalyzer().analyze(q).intent)
    print("Selector:", inspect.getfile(selection))
    print("Review classifier:", is_review_request("What recent systematic reviews address anxiety-disorder treatment?"))
    PY

## Test progression

1. Run `python -m pytest -q` and confirm no regressions.
2. On the existing indexed corpus, run `python scripts/viva_clinician_smoke.py --project-root .`.
3. Inspect `responses.json`, not just each `requires_manual_review` status.
4. Record failures and answer limitations. Do not claim actual clinical accuracy
   without independent human assessment.
