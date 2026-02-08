"""Development environment setup script."""

import subprocess
import sys


def install_dependencies():
    """Install project dependencies."""
    print("Installing project dependencies...")
    
    # Install main dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    
    # Install development dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", ".[dev]"])
    
    print("\nSetup complete!")
    print("\nTo verify installation, run:")
    print("  pytest tests/")


if __name__ == "__main__":
    install_dependencies()
