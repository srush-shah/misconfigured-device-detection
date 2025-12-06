"""
Model Loading Utilities
Handles loading trained models for the web application.
"""

import os
import pickle
import joblib
import torch
import numpy as np
import pandas as pd
from pathlib import Path

def load_trained_models(models_dir=None):
    """
    Load trained models from disk.
    
    Args:
        models_dir: Directory containing saved models
        
    Returns:
        Dictionary with loaded models
    """
    if models_dir is None:
        # Default models directory
        base_dir = Path(__file__).parent.parent
        models_dir = base_dir / 'models' / 'saved'
    
    models_dir = Path(models_dir)
    models = {}
    errors = []
    
    if not models_dir.exists():
        print(f"Models directory not found: {models_dir}")
        print("Please train models first using the Jupyter notebook.")
        return None
    
    # Load ensemble model (best model)
    ensemble_path = models_dir / 'ensemble_model.pkl'
    if ensemble_path.exists():
        try:
            with open(ensemble_path, 'rb') as f:
                models['ensemble'] = pickle.load(f)
            print("✓ Loaded ensemble model")
        except Exception as e:
            error_msg = f"Error loading ensemble model: {e}"
            print(error_msg)
            errors.append(error_msg)
            import traceback
            print(traceback.format_exc())
    
    # Load improved RandomForest
    rf_path = models_dir / 'improved_rf_model.pkl'
    if rf_path.exists():
        try:
            with open(rf_path, 'rb') as f:
                models['improved_rf'] = pickle.load(f)
            print("✓ Loaded improved RandomForest model")
        except Exception as e:
            error_msg = f"Error loading RandomForest model: {e}"
            print(error_msg)
            errors.append(error_msg)
            import traceback
            print(traceback.format_exc())
    
    # Load XGBoost
    xgb_path = models_dir / 'xgb_model.pkl'
    if xgb_path.exists():
        try:
            with open(xgb_path, 'rb') as f:
                models['xgb'] = pickle.load(f)
            print("✓ Loaded XGBoost model")
        except Exception as e:
            error_msg = f"Error loading XGBoost model: {e}"
            print(error_msg)
            errors.append(error_msg)
            import traceback
            print(traceback.format_exc())
    
    # Load pipeline scaler and feature columns
    scaler_path = models_dir / 'scaler.pkl'
    if scaler_path.exists():
        try:
            with open(scaler_path, 'rb') as f:
                models['scaler'] = pickle.load(f)
            print("✓ Loaded feature scaler")
        except Exception as e:
            error_msg = f"Error loading scaler: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    feature_cols_path = models_dir / 'feature_columns.pkl'
    if feature_cols_path.exists():
        try:
            with open(feature_cols_path, 'rb') as f:
                models['feature_columns'] = pickle.load(f)
            print("✓ Loaded feature columns")
        except Exception as e:
            error_msg = f"Error loading feature columns: {e}"
            print(error_msg)
            errors.append(error_msg)
    
    # Store errors in models dict for debugging
    if errors:
        models['_load_errors'] = errors
    
    return models if models else None

def save_trained_models(pipeline, models_dict, models_dir=None):
    """
    Save trained models to disk.
    
    Args:
        pipeline: MainPipeline instance
        models_dict: Dictionary with trained models
        models_dir: Directory to save models
    """
    if models_dir is None:
        base_dir = Path(__file__).parent.parent
        models_dir = base_dir / 'models' / 'saved'
    
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ensemble model
    if 'ensemble' in models_dict:
        with open(models_dir / 'ensemble_model.pkl', 'wb') as f:
            pickle.dump(models_dict['ensemble'], f)
        print(f"✓ Saved ensemble model to {models_dir / 'ensemble_model.pkl'}")
    
    # Save improved RandomForest
    if 'improved_rf' in models_dict:
        with open(models_dir / 'improved_rf_model.pkl', 'wb') as f:
            pickle.dump(models_dict['improved_rf'], f)
        print(f"✓ Saved improved RandomForest to {models_dir / 'improved_rf_model.pkl'}")
    
    # Save XGBoost
    if 'xgb' in models_dict:
        with open(models_dir / 'xgb_model.pkl', 'wb') as f:
            pickle.dump(models_dict['xgb'], f)
        print(f"✓ Saved XGBoost model to {models_dir / 'xgb_model.pkl'}")
    
    # Save feature columns
    if hasattr(pipeline, 'feature_columns') and pipeline.feature_columns:
        with open(models_dir / 'feature_columns.pkl', 'wb') as f:
            pickle.dump(pipeline.feature_columns, f)
        print(f"✓ Saved feature columns to {models_dir / 'feature_columns.pkl'}")

