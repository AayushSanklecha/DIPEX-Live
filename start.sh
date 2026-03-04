#!/bin/bash
# Quick start script for Analytics Platform

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     Analytics Platform - Quick Start                     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Start server
echo ""
echo "🚀 Starting server on http://localhost:8000"
echo "📊 Dashboard will be available at: http://localhost:8000/dashboard"
echo "📚 API docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python main.py serve --port 8000
