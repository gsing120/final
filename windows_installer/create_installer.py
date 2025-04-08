import sys
import os
import PyInstaller.__main__

# Add the project root to the Python path
sys.path.append('/home/ubuntu/gemma_advanced')

# Define the entry point script
entry_point = '/home/ubuntu/gemma_advanced/fixed_frontend_app.py'

# Define the output directory
output_dir = '/home/ubuntu/gemma_advanced/windows_installer'

# Define PyInstaller options
options = [
    entry_point,
    '--name=GemmaAdvancedTrading',
    '--onefile',
    '--windowed',
    f'--distpath={output_dir}',
    f'--workpath={output_dir}/build',
    f'--specpath={output_dir}',
    '--add-data=/home/ubuntu/gemma_advanced/static:static',
    '--add-data=/home/ubuntu/gemma_advanced/templates:templates',
    '--add-data=/home/ubuntu/gemma_advanced/charts:charts',
    '--add-data=/home/ubuntu/gemma_advanced/demo_results:demo_results',
    '--icon=/home/ubuntu/gemma_advanced/static/favicon.ico',
]

# Run PyInstaller
PyInstaller.__main__.run(options)

print("Windows installer created successfully!")
