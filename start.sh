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
echo "🚀 Starting backend server on http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo "✨ Starting Vite React frontend on http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Install frontend dependencies if needed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

# Run both servers concurrently
# We run the frontend in the background and backend in the foreground
(cd frontend && npm run dev) &
FRONTEND_PID=$!

python main.py serve --port 8000

# Cleanup when backend stops
kill $FRONTEND_PID
