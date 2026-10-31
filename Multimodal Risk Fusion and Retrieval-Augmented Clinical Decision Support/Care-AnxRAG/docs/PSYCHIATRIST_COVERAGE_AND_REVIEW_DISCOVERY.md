# Psychiatrist evidence update: bibliography discovery and coverage triage

## Motivation from the 157-active-document workstation
The 2026-10-16 PR #39 v2 smoke test answered:
- P01: cited long-term pharmacological treatment review for GAD (older evidence).
- P05: WHO statement about antidepressants/SSRIs.
- P02, P03, P04, P06, P07: abstained rather than cite unrelated text.

The immediate defect was treating "Which recent reviews address anxiety treatment?"
as a treatment-OUTCOME question. A bibliography request should instead list
exact publication titles (source-level facts), not extract a statement of
efficacy. This is distinct from claiming that the treatments work.

## Implementation
Review-discovery intent is deliberately narrow (what/which/list/identify/find
recent systematic reviews or meta-analyses). It excludes efficacy and
conclusion questions.

Returned items are exact source titles with source IDs. Eligibility requires:
- An active, retrieval-selected, non-conflicted document.
- Evidence level systematic_review or meta_analysis.
- Title explicitly about anxiety and treatment/interventions; not solely
  epidemiology, biomarkers, risk, or prevention.
- For "recent", publication date within the last 1,095 days, not merely
  indexed/modified within that period.
- A source URL.
- Distinct document identities.

The exact-match source-title citation gate is enabled ONLY for such narrow
bibliographic questions. Clinical effect claims remain subject to original
exact-source-sentence verification.

Important: titles themselves do not establish efficacy, validity of study
methods, applicability or a professional clinical recommendation.

## Read-only coverage audit

Run in your Mac's existing virtualenv against the existing SQLite corpus:

    python scripts/clinician_corpus_coverage.py --project-root .

This generates a time-stamped audit at
artifacts/final-viva/coverage-audit/<run>/report.json and counts heuristic
keyword-matching active documents, recent treatment-review titles and source
breakdown. The coverage counts are NOT clinician-adjudicated gold labels, cannot
prove absence of evidence, and may include abstracts without useful findings.

In particular:
- P02: social anxiety/social phobia with CBT
- P03: panic + CBT + exposure
- P06: GAD with treatment and negative or uncertain results
- P07: panic diagnosis or diagnostic assessment

No ingestion, activation, schema migration or vector modification occurs.

## Next research step (do not skip)
After confirming PR CI, run the full seven-question smoke script and inspect
the actual answers and citations. If P02/P03/P07 still lack evidence, curate
and review relevant authoritative disorder-specific guidelines and papers
into Clinical Core or an appropriate research stratum using their original
licenses and provenance. Do not activate rejected abstracts merely to make a
question return an answer.

Existing 47 staged versions block a formal frozen corpus; 157 active
machine-screened documents are not clinically adjudicated. No B0-B5 or
CARE_full result is asserted by these engineering tests.
