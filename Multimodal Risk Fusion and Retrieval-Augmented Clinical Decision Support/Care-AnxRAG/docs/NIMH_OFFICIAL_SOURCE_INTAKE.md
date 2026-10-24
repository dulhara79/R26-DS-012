# NIMH official-source intake — psychiatrist-facing viva

## Grounded motivation (2026-10-16, user workstation)

- Corpus: **157 active documents** (156 PubMed research + 1 WHO source).
- Earlier P01, P04, P05 give real source-cited outputs; P02, P03, P06, P07 abstain.
- P02 coverage query found 15 keyword-matching documents, including RCTs,
  but no acceptable sentence was selected; keyword coverage is not adjudication.
- P07 diagnostic keyword count (6) consisted mainly of trial cohorts or
  epidemiology, not diagnostic guidance.
- P03 comparison count (1) was WHO general information, not a direct head-to-
  head comparison; do NOT force a comparison from that.
- P06 asks for contradictory findings without specifying a reference
  intervention or recommendation; do NOT interpret any negative result as
  automatically contradicting a guideline.

## Targeted independent official sources

Added official NIMH educational pages:
- **nimh_social_anxiety**:
  https://www.nimh.nih.gov/health/publications/social-anxiety-disorder-more-than-just-shyness
  — specifically discusses CBT for social anxiety.
- **nimh_panic_disorder**:
  https://www.nimh.nih.gov/health/publications/panic-disorder-when-fear-overwhelms
  — specifically discusses panic disorder diagnosis and assessment.

NIMH's published reuse policy permits attribution-preserving copying of the
webpage text; do not copy photographs or claim NIMH endorsement.
https://www.nimh.nih.gov/site-info/policies

These are **government health-information**, not clinical practice guidelines
or formal clinician-reviewed gold labels. Both use Clinical Core, have normal
source provenance, and explicitly set auto_promote: false. Neither changes
the existing active count until a reviewer consciously approves the version.

### 1. Read-only trial with existing local corpus

From the Mac's existing CARE-AnxRAG checkout, after fetching the PR into an
isolated worktree, point source registry explicitly at the PR's YAML:

    export CARE_SOURCE_REGISTRY=/tmp/care-viva-nimh-pr41/config/sources.yaml
    PYTHONPATH=/tmp/care-viva-nimh-pr41/src care-anxrag sync \
      --source nimh_social_anxiety --source nimh_panic_disorder \
      --dry-run --force --project-root .

This fetches and validates but does not add versions or advance cursors. Check
the output for source failures and validator rejection, and stop if any appear.

### 2. Stage (only after successful dry-run)

    PYTHONPATH=/tmp/care-viva-nimh-pr41/src care-anxrag sync \
      --source nimh_social_anxiety --source nimh_panic_disorder \
      --force --project-root .

The expected outcome is staging, NOT active source promotion. If output is
rejected or failed, do not claim ingestion success.

### 3. Review the original sources and stage records

    PYTHONPATH=/tmp/care-viva-nimh-pr41/src care-anxrag staging --project-root .

Check page identity, actual content, source freshness, segment context, scope
and all original URLs. A reviewer must specifically decide whether to
approve or reject each new version. Preserve recorded reviewer identity,
decision reason and evidence provenance. This repository's
docs/STAGING_REVIEW_WORKFLOW.md describes the auditable bulk CSV flow; it
may include all historical staged versions, not just these two.

For a single reviewed version, the CLI supports care-anxrag approve VERSION_ID.
**Never approve solely because this PR or CI is green**.

### 4. Re-evaluate the unchanged question set

Only after reviewed activation:
- Run all 10 psychiatrist smoke-test cases again, including N01/N02/S01.
- Check P02 and P07 with exact citations. Two or more diagnostic criteria
  should not be silently collapsed into one incomplete sentence.
- Retain P03/P06 abstention unless directly supported evidence is present.
- Record the new version counts and compare prior vs new results as an
  *engineering development iteration*, not a locked research benchmark.

**Corpus-freeze caveat:** pre-existing 47 unresolved staged versions remain a
blocker even if both new official pages are approved. No benchmark labels or
clinical adjudication are manufactured by this source intake.
