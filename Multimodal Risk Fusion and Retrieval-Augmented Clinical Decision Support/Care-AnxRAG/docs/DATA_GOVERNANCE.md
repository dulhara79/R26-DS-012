# Data governance

## Source classes

### Clinical Core

High-authority, slower-changing sources such as official health information and licensed clinical guidelines. These sources may be eligible for automatic promotion only after parser, licence, and regression tests are approved.

### Research Frontier

New studies, reviews, and research updates. They enter staging by default. A trained reviewer should verify relevance, publication type, population, limitations, retraction/correction status, and applicability before promotion.

## Required provenance

Store and expose where permitted:

- source organization and stable external ID;
- canonical URL;
- publication and update dates;
- retrieval timestamp;
- authors/journal/DOI/PMID/PMCID where available;
- evidence type and authority policy;
- licence/reuse metadata;
- content fingerprint and version relationship;
- review decision and reason.

## Update policy

- Poll PubMed/NICE monitors daily and stable core pages weekly by default.
- Use overlap windows for modification-date APIs.
- Never advance a successful watermark after partial ingestion/index failure.
- Preserve superseded versions for reproducibility.
- Mark withdrawn/retracted evidence and remove it from active retrieval.
- Run periodic vector reconciliation and database integrity checks.

## Copyright and API terms

- PubMed abstracts/metadata and PMC full text have different reuse conditions.
- PMC ingestion is blocked unless licence metadata matches the configured allowlist.
- NICE syndication remains disabled until the project obtains the appropriate licence/API access and confirms AI use conditions.
- Public web availability is not automatically permission to reproduce, index, or redistribute content.

## Sensitive data

The default code does not persist user questions. Infrastructure logs, reverse proxies, tracing systems, and analytics may still capture them. Before collecting any user text:

- define purpose and lawful basis;
- minimize fields;
- prohibit unnecessary identifiers;
- encrypt in transit and at rest;
- limit access;
- define retention/deletion;
- document incident response;
- obtain ethics/privacy review where applicable.

## Review roles

Recommended separation of duties:

- source/licence owner;
- data engineer;
- mental-health domain reviewer;
- retrieval/evaluation owner;
- security/privacy owner;
- release approver.

## Reproducibility snapshot

For each experiment, archive:

- source registry and source-state export;
- active version IDs/content hashes;
- database schema version;
- model names and immutable revisions;
- prompt version;
- chunking/retrieval thresholds and weights;
- benchmark version and annotation protocol;
- code commit and dependency lock;
- evaluation report.


## PubMed linked corrections and integrity events

PubMed's `CommentsCorrectionsList` relationships are retained as provenance metadata.

CARE-AnxRAG applies a conservative policy:

- `RetractionOf`: if the referenced PMID is currently active in the same configured PubMed source, its active version is marked withdrawn and its vectors are removed from active retrieval. The historical version remains in the ledger for reproducibility.
- `ExpressionOfConcernFor`: do not automatically withdraw the referenced article. Create an evidence-review alert so a reviewer can inspect the concern and subsequent outcome.
- `ErratumFor`, `UpdateOf`, `CorrectedandRepublishedFrom`, and `RetractedandRepublishedFrom`: retain the relationship and create a review alert when the referenced article exists in the corpus. These relationships are not treated as equivalent to a retraction.
- A relation pointing to a PMID that is not present in the controlled corpus does not create or infer evidence for that PMID.
- Dry-run synchronization reports discoverable relationships but does not withdraw evidence or write review alerts.

Retraction notices that contain a valid linked PMID are retained for relationship processing even when the notice has no abstract. They are not promoted merely because they exist; normal validation and review rules still apply.

The relationship metadata is included in the version fingerprint so a newly added or changed PubMed integrity relationship can create a new auditable version even if the article abstract itself is unchanged.
