# Build from the repository ROOT, not from central_backend/:
# docker build -f central_backend/Dockerfile -t r26ds012-central .
FROM python:3.11-slim

RUN useradd -m -u 1000 user
WORKDIR /app

# In-process fusion needs both modules, numpy and frozen reference distributions.
COPY central_backend/requirements.txt /app/backend-requirements.txt
COPY fusion_service/requirements.txt /app/fusion-requirements.txt
RUN pip install --no-cache-dir -r /app/backend-requirements.txt \
    -r /app/fusion-requirements.txt

COPY --chown=user central_backend/*.py /app/central_backend/
COPY --chown=user fusion_service/fusion.py fusion_service/harmonise.py /app/fusion_service/
COPY --chown=user fusion_service/reference/*.json /app/fusion_service/reference/

RUN mkdir -p /data && chown user:user /data
ENV FUSION_SERVICE_DIR=/app/fusion_service
WORKDIR /app/central_backend
USER user
EXPOSE 8000

# One-off migrations must be executed separately: migrate_p0 has no distributed
# DDL lock and multiple replicas must not race against one another.
# Fail closed instead of defaulting to an ephemeral image-local SQLite DB.
CMD ["sh", "-c", "if [ -z \"$DATABASE_URL\" ]; then echo 'DATABASE_URL is required' >&2; exit 1; fi; exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
