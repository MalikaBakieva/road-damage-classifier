# ---------------------------------------------------------------------------
# GOV-01 road damage classifier — inference image
# Multi-stage build: wheels are compiled in `builder`, the runtime image stays
# slim and runs as a non-root user.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps the image ~1.5 GB smaller than the CUDA default.
RUN pip install --prefix=/install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    RDC_MODEL_PATH=/app/models/binary/model.pt \
    RDC_CONFIDENCE_THRESHOLD=0.6 \
    RDC_MAX_UPLOAD_MB=10

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 rdc

COPY --from=builder /install /usr/local

WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY models/ ./models/

RUN chown -R rdc:rdc /app
USER rdc

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "rdc.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
