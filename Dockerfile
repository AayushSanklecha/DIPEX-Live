# DIPEX Enterprise Analytics Platform
# ─────────────────────────────────────
# Multi-stage build: builder → slim runtime

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install "pip<24.1" && \
    pip install --default-timeout=1000 --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN addgroup --system dipex && adduser --system --group dipex && \
    mkdir -p /app/data/uploads /app/data/approved_outputs /app/data/experience \
    /app/data/model_registry /app/data/state /app/audit /app/reports \
    /app/governance && \
    chown -R dipex:dipex /app

# Copy source code
COPY --chown=dipex:dipex . .

USER dipex

# Expose API port
EXPOSE 8000

# Health check
# Health check — use Python's stdlib urllib so we need no extra packages in the runtime image
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request, sys; urllib.request.urlopen('http://localhost:8000/health'); sys.exit(0)" || exit 1

# Run uvicorn production server
CMD ["uvicorn", "api.app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--log-level", "info", \
    "--access-log"]
