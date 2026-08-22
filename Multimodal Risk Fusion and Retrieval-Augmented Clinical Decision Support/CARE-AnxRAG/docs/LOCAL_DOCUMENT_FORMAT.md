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
  population: adults
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
