#!/bin/bash

# ════════════════════════════════════════════
# Marrow — One-command setup & run
# ════════════════════════════════════════════

set -e

echo ""
echo "╔══════════════════════════════════╗"
echo "║   Marrow — RAG Voice Agent       ║"
echo "╚══════════════════════════════════╝"
echo ""

# ── Check Python ────────────────────────────
if ! command -v python3 &> /dev/null; then
  echo "❌ Python 3 not found. Please install Python 3.10+"
  exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✓ Python $PYTHON_VERSION detected"

# ── Check API key ────────────────────────────
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "⚠️  ANTHROPIC_API_KEY not set."
  echo ""
  read -p "   Paste your Anthropic API key: " API_KEY
  export ANTHROPIC_API_KEY="$API_KEY"
  echo ""
  echo "   To avoid entering it next time, run:"
  echo "   export ANTHROPIC_API_KEY='$API_KEY'"
  echo ""
fi

# ── Virtual environment ──────────────────────
if [ ! -d "venv" ]; then
  echo "→ Creating virtual environment..."
  python3 -m venv venv
fi

echo "→ Activating virtual environment..."
source venv/bin/activate

# ── Install dependencies ─────────────────────
echo "→ Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✓ All dependencies installed"
echo ""

# ── Create static dir if needed ─────────────
mkdir -p static data/chroma_db

# ── Launch ───────────────────────────────────
echo "════════════════════════════════════"
echo "  Starting Marrow on port 8000..."
echo "  Open: http://localhost:8000"
echo "════════════════════════════════════"
echo ""

python3 main.py
