#!/bin/bash
# 🔐 Privacy-First Federated Learning Platform - Quick Start

echo "🚀 Privacy-First Federated Learning Platform"
echo "=============================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🎯 Starting system..."
echo "   - Backend: Federated Learning Server"
echo "   - API: http://localhost:8000"
echo "   - Dashboard: http://localhost:8501"
echo ""

python3 main.py "$@"
