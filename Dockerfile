# ─────────────────────────────────────────────────────────────────
# DIPEX API — Dockerfile
# ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

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

# Create runtime directories
RUN mkdir -p data models audit logs reports

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
