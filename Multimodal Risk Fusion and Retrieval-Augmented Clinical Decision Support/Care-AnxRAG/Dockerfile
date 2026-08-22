FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CARE_PROJECT_ROOT=/app \
    CARE_HOME=/app/var

RUN groupadd --system care && useradd --system --gid care --create-home care

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install ".[production]"

COPY config ./config
COPY data ./data
COPY docs ./docs
COPY examples ./examples
COPY scripts ./scripts
COPY .env.example ./

RUN mkdir -p /app/var /home/care/.cache \
    && chown -R care:care /app /home/care

USER care

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
  CMD python -c "import json,urllib.request; data=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)); raise SystemExit(0 if data.get('status') == 'ok' else 1)"

CMD ["care-anxrag", "serve", "--host", "0.0.0.0", "--port", "8000"]
