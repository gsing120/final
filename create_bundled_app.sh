#!/bin/bash

# Create a virtual environment for the bundled application
echo "Creating virtual environment..."
python3 -m venv bundled_app/gemma_venv

# Activate the virtual environment
echo "Activating virtual environment..."
source bundled_app/gemma_venv/bin/activate

# Install required packages
echo "Installing required packages..."
pip install pyinstaller
pip install yfinance pandas numpy matplotlib flask flask-cors scikit-learn

# Copy necessary files to bundled_app directory
echo "Copying application files..."
cp fixed_frontend_app.py bundled_app/
cp -r backend bundled_app/
cp -r static bundled_app/
cp -r templates bundled_app/
cp -r charts bundled_app/
cp -r docs bundled_app/
cp simplified_strategy_optimization.py bundled_app/
cp strategy_validation.py bundled_app/

# Create PyInstaller spec file
echo "Creating PyInstaller spec file..."
cd bundled_app
pyinstaller --name=GemmaAdvancedTrading \
  --onefile \
  --windowed \
  --add-data="static:static" \
  --add-data="templates:templates" \
  --add-data="charts:charts" \
  --add-data="docs:docs" \
  --hidden-import=pandas \
  --hidden-import=numpy \
  --hidden-import=matplotlib \
  --hidden-import=yfinance \
  --hidden-import=flask \
  --hidden-import=flask_cors \
  --hidden-import=scikit-learn \
  fixed_frontend_app.py

echo "Build process completed!"
echo "Executable will be available in bundled_app/dist/GemmaAdvancedTrading"

# Deactivate virtual environment
deactivate
