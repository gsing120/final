import sys
import os
import shutil

# Create a directory structure for the Windows application
windows_app_dir = '/home/ubuntu/gemma_advanced/windows_installer/GemmaAdvancedTrading'
os.makedirs(windows_app_dir, exist_ok=True)

# Create subdirectories
for subdir in ['backend', 'frontend', 'static', 'templates', 'charts', 'docs']:
    os.makedirs(os.path.join(windows_app_dir, subdir), exist_ok=True)

# Copy necessary files
# Backend files
backend_src = '/home/ubuntu/gemma_advanced/backend'
backend_dst = os.path.join(windows_app_dir, 'backend')
for item in os.listdir(backend_src):
    s = os.path.join(backend_src, item)
    d = os.path.join(backend_dst, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

# Copy optimization and strategy files
for file in ['simplified_strategy_optimization.py', 'strategy_validation.py', 'fixed_frontend_app.py']:
    src_file = os.path.join('/home/ubuntu/gemma_advanced', file)
    if os.path.exists(src_file):
        shutil.copy2(src_file, windows_app_dir)

# Copy static files
static_src = '/home/ubuntu/gemma_advanced/static'
static_dst = os.path.join(windows_app_dir, 'static')
if os.path.exists(static_src):
    for item in os.listdir(static_src):
        s = os.path.join(static_src, item)
        d = os.path.join(static_dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

# Copy templates
templates_src = '/home/ubuntu/gemma_advanced/templates'
templates_dst = os.path.join(windows_app_dir, 'templates')
if os.path.exists(templates_src):
    for item in os.listdir(templates_src):
        s = os.path.join(templates_src, item)
        d = os.path.join(templates_dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

# Copy charts
charts_src = '/home/ubuntu/gemma_advanced/charts'
charts_dst = os.path.join(windows_app_dir, 'charts')
if os.path.exists(charts_src):
    for item in os.listdir(charts_src):
        s = os.path.join(charts_src, item)
        d = os.path.join(charts_dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

# Copy documentation
docs_src = '/home/ubuntu/gemma_advanced/docs'
docs_dst = os.path.join(windows_app_dir, 'docs')
if os.path.exists(docs_src):
    for item in os.listdir(docs_src):
        s = os.path.join(docs_src, item)
        d = os.path.join(docs_dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

# Create Windows launcher script
launcher_content = '''@echo off
echo Starting Gemma Advanced Trading System...
start /B pythonw fixed_frontend_app.py
'''

with open(os.path.join(windows_app_dir, 'launch_gemma.bat'), 'w') as f:
    f.write(launcher_content)

# Create requirements.txt for Windows
requirements_content = '''
yfinance
pandas
numpy
matplotlib
flask
flask-cors
requests
scikit-learn
'''

with open(os.path.join(windows_app_dir, 'requirements.txt'), 'w') as f:
    f.write(requirements_content)

# Create Windows installation instructions
install_instructions = '''
# Gemma Advanced Trading System - Windows Installation

## Prerequisites
- Python 3.8 or higher
- pip package manager

## Installation Steps

1. **Install Python Dependencies**
   Open Command Prompt as Administrator and run:
   ```
   pip install -r requirements.txt
   ```

2. **Launch the Application**
   Double-click on `launch_gemma.bat` to start the application.
   
   The web interface will automatically open in your default browser.

## Troubleshooting

If you encounter any issues:
1. Ensure all dependencies are installed correctly
2. Check that no other application is using port 5000
3. Verify that you have internet access for market data retrieval

## System Requirements
- Windows 10 or higher
- 4GB RAM minimum (8GB recommended)
- 500MB free disk space
- Internet connection for market data
'''

with open(os.path.join(windows_app_dir, 'INSTALL.md'), 'w') as f:
    f.write(install_instructions)

# Create a README file
readme_content = '''
# Gemma Advanced Trading System

A sophisticated trading system with Gemma 3 integration for advanced market analysis, strategy generation, and optimization.

## Features

- Natural Language Market Analysis
- Advanced Mathematical Modeling
- Strategy Reasoning and Explanation
- Adaptive Learning
- Strategy Generation & Refinement
- Real-Time Signal Analysis
- Performance Validation

## Getting Started

See the `INSTALL.md` file for installation instructions.

## Documentation

Detailed documentation is available in the `docs` folder.
'''

with open(os.path.join(windows_app_dir, 'README.md'), 'w') as f:
    f.write(readme_content)

# Create a zip file of the Windows application
shutil.make_archive(
    '/home/ubuntu/gemma_advanced/windows_installer/GemmaAdvancedTrading_Windows',
    'zip',
    '/home/ubuntu/gemma_advanced/windows_installer',
    'GemmaAdvancedTrading'
)

print("Windows application package created successfully!")
print(f"Zip file location: /home/ubuntu/gemma_advanced/windows_installer/GemmaAdvancedTrading_Windows.zip")
