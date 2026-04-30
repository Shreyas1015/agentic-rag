# One image, two roles — used by both the `api` and `worker` services in
# docker-compose.yml. The compose file overrides `command` per role.
#
# Build:        docker compose build api
# Run (api):    uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
# Run (worker): uv run celery -A app.core.celery_app worker --loglevel=info

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System libs needed by Docling (layout / table / OCR pipeline) and friends.
#   poppler-utils   : pdftoppm / pdftotext used by Docling for PDF rendering
#   tesseract-ocr   : OCR fallback for scanned PDFs (DOCLING_OCR_ENABLED=true)
#   libgl1, libglib : runtime libs for opencv (used by layout model)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv from the official image.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (better layer caching). uv reads .python-version
# but we have UV_PYTHON_DOWNLOADS=never so it must use the system Python 3.12.
# BuildKit cache mount keeps uv's download cache between builds — first build
# pulls from PyPI, subsequent builds reuse the local cache.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Copy app source.
COPY app ./app
COPY scripts ./scripts
COPY migrations ./migrations
COPY alembic.ini ./

# NOTE: We deliberately do NOT pre-download the BGE reranker at build time.
# /root/.cache is mounted as a named volume (`model_cache`) in compose, and the
# volume's content masks the image's content on container start — meaning any
# build-time download here gets shadowed at runtime. The model downloads on
# first request and persists in the volume across rebuilds.

# Persist HF / Docling model caches across container restarts via a named
# volume mounted at /root/.cache (declared in docker-compose.yml).
VOLUME ["/root/.cache"]

EXPOSE 8000

# Default command runs the api; the worker service in docker-compose.yml
# overrides this with the celery invocation.
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
