#!/bin/bash
# Quick setup script for openmythos-swarm

set -e

echo "================================"
echo "OpenMythos Swarm — Quick Setup"
echo "================================"
echo

# Check Python
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version detected"

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Run tests
echo
echo "Running integration test..."
python tests/test_basic.py

echo
echo "================================"
echo "✓ Setup complete!"
echo "================================"
echo
echo "Next steps:"
echo "  1. Start master: python master/server.py"
echo "  2. Start worker: python worker/client.py"
echo "  3. See README.md for deployment options"
echo
