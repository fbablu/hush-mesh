#!/usr/bin/env python3
"""
Setup script for ship classification project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_kaggle_config():
    """Setup Kaggle configuration"""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Note: In production, use environment variables or secure credential storage
    print("Kaggle API will be configured via environment variables")
    print("Make sure to set KAGGLE_USERNAME and KAGGLE_KEY")

def create_directories():
    """Create necessary directories"""
    dirs = ["data", "models", "results"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")

def main():
    print("Setting up Ship Classification Project...")
    
    install_requirements()
    setup_kaggle_config()
    create_directories()
    
    print("\nSetup complete!")
    print("Run: python ship_classifier.py")

if __name__ == "__main__":
    main()