FROM python:3.11-slim

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser . .
USER appuser

# transformers/huggingface_hub need a writable cache under a path we own
ENV HF_HOME=/app/.cache \
    TRANSFORMERS_CACHE=/app/.cache \
    PYTHONUNBUFFERED=1

EXPOSE 7860
# --workers 1 is deliberate: 2 CPU cores, and a second worker would load a
# second copy of Bio_ClinicalBERT into a 16 GB budget.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
