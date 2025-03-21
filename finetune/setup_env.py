import os
import subprocess
import sys

# Define paths
output_dir = "out"
venv_dir = "../.venv"
requirements_file = "requirements.txt"

# Create the directory
os.makedirs(output_dir, exist_ok=True)

# Create a virtual environment
subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)

# Install dependencies
pip_executable = os.path.join(venv_dir, "bin", "pip") if os.name != "nt" else os.path.join(venv_dir, "Scripts", "pip")
subprocess.run([pip_executable, "install", "-r", requirements_file], check=True)