# Security policy

CARE-AnxRAG is a research reference implementation, not a clinical service.

## Reporting a vulnerability

Do not open a public issue containing credentials, licensed evidence, user data, or an
exploitable security detail. Use the private reporting channel defined by the deploying
organization. This repository does not define a public response SLA.

## Supported release

Only the latest tagged release is intended to receive fixes.

## Minimum deployment controls

- Configure a non-empty `CARE_ADMIN_KEY`; administrative HTTP routes fail closed otherwise.
- Place the service behind TLS and an identity-aware reverse proxy.
- Apply request-size, concurrency, and rate limits.
- Keep SQLite, Chroma, `.env`, model caches, and licensed content access-restricted.
- Do not log sensitive user text by default.
- Pin and scan dependencies, images, and model revisions.
- Review every source licence and all data-retention obligations.
- Localize crisis resources and complete independent clinical/safety review.

See `docs/THREAT_MODEL.md` and `docs/DATA_GOVERNANCE.md` for the full control set.
