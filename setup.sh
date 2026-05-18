#!/usr/bin/env bash
# =============================================================================
#  ADAP Analytics Platform — One-Shot Environment Setup
#  Run this ONCE after cloning the repo before starting the backend.
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
# =============================================================================

set -e  # Exit immediately on any error

echo ""
echo "========================================================="
echo "  ADAP Analytics Platform — Environment Setup"
echo "========================================================="
echo ""

# ── 1. Python dependencies ────────────────────────────────────
echo "[1/4] Installing Python dependencies..."
pip install -r requirements.txt
echo "      ✓ Python packages installed."
echo ""

# ── 2. spaCy language model (REQUIRED for NLP Column Analyzer) ─
echo "[2/4] Downloading spaCy English language model..."
python -m spacy download en_core_web_sm
echo "      ✓ spaCy en_core_web_sm downloaded."
echo ""

# ── 3. Create required directories ───────────────────────────
echo "[3/4] Creating required runtime directories..."
mkdir -p reports
mkdir -p data/snapshots
mkdir -p data/uploads
mkdir -p data/tmp
mkdir -p data/bronze
mkdir -p data/silver
mkdir -p data/gold
mkdir -p data/model_registry
mkdir -p audit
mkdir -p models
echo "      ✓ Runtime directories created."
echo ""

# ── 4. .env file ─────────────────────────────────────────────
echo "[4/4] Checking .env file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "      ✓ .env created from .env.example — please fill in your API keys."
else
    echo "      ✓ .env already exists — skipping."
fi
echo ""

echo "========================================================="
echo "  Setup complete! You can now start the platform:"
echo ""
echo "  Backend:   uvicorn api.app:app --reload --port 8000"
echo "  Frontend:  cd frontend && npm install && npm run dev"
echo "========================================================="
echo ""
