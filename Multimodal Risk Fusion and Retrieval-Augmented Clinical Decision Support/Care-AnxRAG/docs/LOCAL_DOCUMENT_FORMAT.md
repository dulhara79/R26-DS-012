# Local document format

Only add material you are legally and ethically authorized to process and redistribute as required by your deployment.

## Markdown with YAML front matter

```markdown
---
external_id: organization-guideline-001
title: Anxiety guideline title
url: https://example.org/source
published_at: 2025-01-10
updated_at: 2026-04-01
language: en
authors:
  - Example Organization
publication_types:
  - Clinical Guideline
topics:
  - anxiety
  - panic_disorder
metadata:
  licence: CC-BY-4.0
  pico:
    population:
      - Adults with generalized anxiety disorder
    intervention:
      - Cognitive behavioural therapy
    comparator:
      - Treatment as usual
    outcome:
      - Anxiety symptom severity at 12 weeks
---

# Overview

Authorized source text.

# Recommendations

Authorized source text.
```

Required after parsing:

- non-empty `external_id` or a path-derived ID;
- non-empty title;
- at least 300 characters by default;
- English (`en`, `eng`, or `english`) in the current implementation.

## Plain text

The filename stem becomes the title and relative path becomes the external ID. Plain text cannot express rich provenance, so Markdown front matter is preferred.

## HTML

The parser removes scripts/styles/navigation/footer elements and extracts headings, paragraphs, and list items. Put source metadata in a sidecar workflow or convert to Markdown when provenance matters.

## JSON

Supported keys include:

```json
{
  "external_id": "source-001",
  "title": "Document title",
  "text": "Document text",
  "url": "https://example.org/source",
  "published_at": "2025-01-10",
  "updated_at": "2026-04-01",
  "language": "en",
  "authors": ["Example Organization"],
  "publication_types": ["Systematic Review"],
  "topics": ["anxiety", "social_anxiety_disorder"],
  "metadata": {"licence": "CC-BY-4.0"}
}
```

## Promotion policy

The default local source is manual-review only. After synchronization:

```bash
care-anxrag staging --project-root .
care-anxrag approve VERSION_ID --project-root .
```

Do not label a document as a clinical guideline, systematic review, or other evidence type unless the source itself supports that classification.


## Clinical evidence facets and PICO annotations

CARE-AnxRAG stores clinical facets with explicit provenance.

Automatically normalized facets are limited to deterministic phrase matches for known anxiety subtypes, treatments, and broad populations. This is normalization of text that is actually present in the source; it is not model-generated clinical interpretation.

PICO fields are **annotation-only**. Put them under `metadata.pico`:

```yaml
metadata:
  pico:
    population:
      - Adults with generalized anxiety disorder
    intervention:
      - Cognitive behavioural therapy
    comparator:
      - Treatment as usual
    outcome:
      - GAD-7 symptom severity at 12 weeks
```

Each field may be a list of strings. Leave a field empty or omit it when the source/reviewer has not established that information.

Do not infer a comparator, outcome, treatment effect, population, or follow-up period merely because it would be plausible. Unknown stays unknown.

The stored chunk metadata contains `clinical_facets` with:

- normalized anxiety subtype concepts from source text/topics;
- normalized treatment concepts from source text;
- normalized population concepts from source text;
- the exact source/reviewer-supplied PICO annotation;
- provenance labels identifying deterministic phrase matching versus explicit metadata annotation.

Changing the PICO annotation changes the document version fingerprint so controlled annotation revisions remain auditable.
