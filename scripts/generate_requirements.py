#!/usr/bin/env python3
"""
Generate requirements.txt file for Gemma Advanced Trading System
"""

import os
import sys

# Define required packages with versions
REQUIREMENTS = {
    # Core data processing
    "numpy": ">=1.20.0",
    "pandas": ">=1.3.0",
    "scipy": ">=1.7.0",
    
    # Machine learning
    "scikit-learn": ">=1.0.0",
    "tensorflow": ">=2.6.0",
    "torch": ">=1.9.0",
    
    # Data visualization
    "matplotlib": ">=3.4.0",
    "seaborn": ">=0.11.0",
    "plotly": ">=5.0.0",
    
    # Web framework
    "fastapi": ">=0.68.0",
    "uvicorn": ">=0.15.0",
    
    # Database
    "sqlalchemy": ">=1.4.0",
    "redis": ">=4.0.0",
    
    # Testing
    "pytest": ">=6.2.0",
    "pytest-cov": ">=2.12.0",
    
    # API clients
    "requests": ">=2.26.0",
    "websocket-client": ">=1.2.0",
    
    # Utilities
    "tqdm": ">=4.62.0",
    "joblib": ">=1.0.0",
    "pyyaml": ">=6.0",
    
    # Financial libraries
    "ta-lib": ">=0.4.0",
    "yfinance": ">=0.1.70",
    "alpaca-trade-api": ">=2.0.0",
}

def generate_requirements_file(output_path="requirements.txt"):
    """Generate requirements.txt file with all dependencies"""
    with open(output_path, "w") as f:
        for package, version in REQUIREMENTS.items():
            f.write(f"{package}{version}\n")
    
    print(f"Generated requirements file at: {output_path}")
    print(f"Total packages: {len(REQUIREMENTS)}")

if __name__ == "__main__":
    # Get output path from command line args or use default
    output_path = sys.argv[1] if len(sys.argv) > 1 else "requirements.txt"
    generate_requirements_file(output_path)
