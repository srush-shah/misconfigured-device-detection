#!/usr/bin/env python3
"""
Diagnostic script to check if models can be loaded correctly.
Run this to see why the web app might be showing "demo mode" message.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_app.model_loader import load_trained_models

def main():
    print("=" * 60)
    print("MODEL LOADING DIAGNOSTIC")
    print("=" * 60)
    print()
    
    # Check models directory
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / 'models' / 'saved'
    
    print(f"Models directory: {models_dir}")
    print(f"Directory exists: {models_dir.exists()}")
    print()
    
    if not models_dir.exists():
        print("❌ Models directory does not exist!")
        print("   Please train models using the Jupyter notebook first.")
        return 1
    
    # List files in directory
    print("Files in models directory:")
    for file in sorted(models_dir.iterdir()):
        size = file.stat().st_size
        print(f"  - {file.name} ({size:,} bytes)")
    print()
    
    # Try to load models
    print("Attempting to load models...")
    print("-" * 60)
    models = load_trained_models()
    print("-" * 60)
    print()
    
    if models is None:
        print("❌ No models loaded!")
        print("   This means the web app will run in demo mode.")
        return 1
    
    # Check which models were loaded
    print("Loaded models:")
    model_keys = [k for k in models.keys() if not k.startswith('_')]
    if not model_keys:
        print("  ❌ No models successfully loaded!")
    else:
        for key in model_keys:
            print(f"  ✓ {key}")
    
    # Check for errors
    if '_load_errors' in models:
        print()
        print("⚠️  Errors encountered while loading models:")
        for error in models['_load_errors']:
            print(f"  - {error}")
        print()
        print("Common causes:")
        print("  1. Missing dependencies (e.g., scikit-learn, xgboost)")
        print("  2. Version mismatches between training and inference")
        print("  3. Corrupted pickle files")
        print("  4. Missing custom classes/modules")
        return 1
    
    # Check if required models exist
    required_models = ['improved_rf', 'ensemble', 'xgb']
    missing = [m for m in required_models if m not in models]
    
    if missing:
        print()
        print(f"⚠️  Missing models: {', '.join(missing)}")
        print("   The web app can still work with available models.")
    
    if 'improved_rf' in models or 'ensemble' in models:
        print()
        print("✅ Models loaded successfully!")
        print("   The web app should work in full mode (not demo mode).")
        return 0
    else:
        print()
        print("⚠️  Best models (improved_rf, ensemble) not found.")
        print("   The web app will run in demo mode.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

