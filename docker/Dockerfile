# syntax=docker/dockerfile:1.7
# Multi-target image: `api` (FastAPI) dan `ui` (Streamlit). Build:
#   docker build -f docker/Dockerfile --target api -t minesense-api .
#   docker build -f docker/Dockerfile --target ui  -t minesense-ui  .

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends curl tini && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 10001 app
WORKDIR /app

# ---------- dependencies (layer cache) ----------
FROM base AS deps
COPY pyproject.toml README.md ./
RUN mkdir -p src/minesense && touch src/minesense/__init__.py \
 && pip install ".[pdf,qdrant]" && pip uninstall -y minesense

# ---------- API ----------
FROM deps AS api
COPY src ./src
COPY data ./data
ENV MINESENSE_ARTIFACTS_DIR=/app/artifacts MINESENSE_DATA_DIR=/app/data
# bangun index RAG + model classifier saat build → container siap tanpa langkah manual
RUN pip install --no-deps . \
 && minesense ingest \
 && minesense train \
 && mkdir -p /app/data/uploads /app/artifacts/feedback /tmp/prometheus \
 && chown -R app:app /app /tmp/prometheus
ENV PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 CMD curl -fs http://localhost:8000/health || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "minesense.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers", "--forwarded-allow-ips", "*"]

# ---------- UI ----------
FROM base AS ui
RUN pip install "streamlit>=1.33" "pandas>=2.0" "httpx>=0.27"
COPY ui ./ui
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD curl -fs http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false", "--client.toolbarMode=minimal"]
