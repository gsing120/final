#!/bin/bash
# Installation script for Gemma Advanced Trading System

echo "Installing Gemma Advanced Trading System..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p data
mkdir -p prompts

echo "Installation complete!"
echo "To activate the environment, run: source venv/bin/activate"
echo "To run the QBTS swing trading demo, run: python demos/qbts_swing_trading_demo.py"
