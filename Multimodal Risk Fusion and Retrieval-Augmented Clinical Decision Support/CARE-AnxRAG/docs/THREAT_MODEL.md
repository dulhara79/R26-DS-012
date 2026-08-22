# Threat model

## Assets

- authoritative evidence corpus and provenance;
- licensed source content;
- user questions;
- model/index configuration;
- admin credentials;
- review decisions and research results.

## Primary threats and controls

### Corpus poisoning

**Threat:** malicious or low-quality evidence enters retrieval.

**Controls:** source allowlist, authority/evidence metadata, staging, manual research review, licence/retraction checks, duplicate detection, active-state filtering, benchmark regressions.

### Prompt injection in retrieved text

**Threat:** a document instructs the model to ignore system rules or fabricate citations.

**Controls:** explicit untrusted-evidence delimiters, system instruction to ignore excerpt commands, structured JSON output, strict citation allowlist, answer abstention after failed repair.

### Stale/superseded evidence

**Threat:** old vectors remain physically present.

**Controls:** SQLite active-state join on every dense/lexical candidate, durable delete outbox, reconciliation, version history.

### Citation laundering

**Threat:** answer cites a real source ID for a claim unsupported by that source.

**Controls implemented:** source-ID allowlist and exact inline/structured parity.

**Additional required evaluation:** claim-level entailment/citation correctness with expert review. ID validation alone does not prove claim support.

### Administrative endpoint exposure

**Threat:** unauthorized sync, raw retrieval, review, or reconciliation.

**Controls:** administrative/debug HTTP routes fail closed until a non-empty `CARE_ADMIN_KEY`
is configured; constant-time key comparison; secret filtering; safe browser DOM rendering.

**Deployment requirement:** TLS, identity-aware proxy/OAuth, rate limiting, key rotation, audit logs, and non-empty admin key.

### XSS and malicious URLs

**Threat:** hostile source title/URL executes in browser UI.

**Controls:** DOM `textContent`, no evidence `innerHTML`, and HTTP/HTTPS URL allowlist.

### Denial of service

**Threat:** oversized queries, large update windows, model exhaustion, repeated synchronization.

**Controls:** Pydantic question length limit, source caps, NCBI truncation refusal, batching, timeouts, retry limits.

**Deployment requirement:** reverse-proxy body limits, concurrency limits, quotas, and job isolation.

### Safety misclassification

**Threat:** urgent risk is treated as ordinary RAG, or academic language is falsely escalated.

**Controls:** pre-retrieval first-person/imminence rules and negation handling.

**Residual risk:** this router is not clinically validated. Add a validated safety classifier, human escalation policy, localized resources, and red-team testing before user-facing deployment.
