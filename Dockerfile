# ─────────────────────────────────────────────────────────────────
# DIPEX API — Dockerfile (Issue 06: exact Python version)
# ─────────────────────────────────────────────────────────────────
FROM python:3.12.7-slim-bookworm

# Set timezone to UTC — avoids datetime.utcnow() ambiguity
ENV TZ=UTC

# System deps needed for psycopg2, pymongo, confluent-kafka
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
# Install Python dependencies — torch excluded (not needed at runtime, saves ~900MB)
COPY requirements.docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.docker.txt


# Copy project
COPY . .

# Create runtime directories needed for pipeline execution.
# data/snapshots/ — Parquet snapshots from pipeline runs (written at runtime)
# audit/ and reports/ are deployed from the repo, but mkdir -p is safe (no-op if exists)
RUN mkdir -p data/snapshots models logs audit reports

# Expose API port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI — single worker: audit/reports use shared file paths, multiple
# workers cause race conditions. Scale via HF Spaces hardware upgrades if needed.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
