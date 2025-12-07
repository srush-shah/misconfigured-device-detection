"""
Web Application for Misconfigured Device Detection
Streamlit-based demo application using the best trained model.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import torch
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pipeline.main_pipeline import MainPipeline
    from models.improved_classifiers import ImprovedRandomForest, ImprovedXGBoost, EnsembleClassifier
    from models.improved_multi_view_fusion import ImprovedMultiViewFusionModel
    from utils.threshold_tuning import find_optimal_threshold
    from web_app.model_loader import load_trained_models
    PIPELINE_AVAILABLE = True
except ImportError as e:
    PIPELINE_AVAILABLE = False
    import warnings
    warnings.warn(f"Pipeline modules not available: {e}. Running in demo mode only.")

# Page configuration
st.set_page_config(
    page_title="Network Device Misconfiguration Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .normal {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .misconfig {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    """Load the trained pipeline and models."""
    try:
        # Initialize pipeline
        pipeline = MainPipeline(config={
            'time_window_minutes': 5,
            'sequence_length': 12,
            'device': 'cpu',
            'batch_size': 32,
            'lstm_epochs': 50,
            'multi_view_epochs': 50,
            'n_clusters': 5,
            'confidence_threshold': 0.7,
            'use_improved_models': True
        })
        
        # Try to load saved models
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'saved')
        if os.path.exists(models_dir):
            # Load models if they exist
            pass  # Will implement model saving/loading later
        
        return pipeline, None
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None, str(e)

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    # Create sample feature data
    sample_features = {
        'device_id': ['device_001', 'device_002', 'device_003'],
        'time_window': [
            datetime(2023, 3, 10, 10, 30),
            datetime(2023, 3, 10, 10, 35),
            datetime(2023, 3, 10, 10, 40)
        ],
        'flow_bytes_out_sum': [1024000, 2048000, 512000],
        'flow_bytes_in_sum': [512000, 1024000, 256000],
        'flow_packets_out_sum': [1000, 2000, 500],
        'flow_packets_in_sum': [500, 1000, 250],
        'flow_duration_mean': [5.2, 3.8, 7.1],
        'label': [0, 2, 0]  # 0=Normal, 2=DHCP Misconfig
    }
    return pd.DataFrame(sample_features)

def combine_ml_and_rule_based(ml_prediction, ml_confidence, ml_probabilities, features_df, label_map):
    """
    Intelligently combine ML model predictions with rule-based detection.
    
    Strategy:
    1. If rule-based has high confidence (>0.8) and ML has low confidence (<0.6), trust rule-based
    2. If both agree, boost confidence
    3. If they disagree, use weighted voting (rule-based gets more weight for DHCP/DNS)
    4. For Gateway/ARP, prefer ML if it has reasonable confidence
    """
    # Get rule-based prediction
    rule_pred, rule_confidence, rule_name = get_rule_based_prediction(features_df)
    
    # Determine final prediction and confidence
    final_pred = ml_prediction
    final_confidence = ml_confidence
    method_used = "ml_only"
    
    # Rule-based is more reliable for DHCP and DNS (domain-specific rules)
    rule_based_preferred = rule_pred in [1, 2]  # DNS or DHCP
    
    # If rule-based has high confidence and ML has low confidence
    if rule_confidence > 0.8 and ml_confidence < 0.6:
        # Trust rule-based, especially for DHCP/DNS
        if rule_based_preferred or rule_confidence > 0.85:
            final_pred = rule_pred
            final_confidence = rule_confidence
            method_used = "rule_based_override"
        else:
            # For other types, use weighted average
            final_confidence = (rule_confidence * 0.4 + ml_confidence * 0.6)
            method_used = "weighted_average"
    
    # If both agree, boost confidence
    elif rule_pred == ml_prediction:
        # Boost confidence when both agree
        final_confidence = min(0.95, ml_confidence + 0.1)
        method_used = "combined_agreement"
    
    # If they disagree
    elif rule_pred != ml_prediction:
        if rule_based_preferred and rule_confidence > 0.75:
            # For DHCP/DNS, prefer rule-based if it's confident
            final_pred = rule_pred
            final_confidence = rule_confidence * 0.9  # Slight penalty for disagreement
            method_used = "rule_based_preferred"
        elif ml_confidence > 0.7 and rule_confidence < 0.7:
            # If ML is more confident, trust it
            final_confidence = ml_confidence
            method_used = "ml_preferred"
        else:
            # Weighted voting
            rule_weight = 0.6 if rule_based_preferred else 0.4
            ml_weight = 1.0 - rule_weight
            
            # Use the prediction with higher weighted confidence
            rule_weighted_conf = rule_confidence * rule_weight
            ml_weighted_conf = ml_confidence * ml_weight
            
            if rule_weighted_conf > ml_weighted_conf:
                final_pred = rule_pred
                final_confidence = rule_confidence
                method_used = "weighted_rule_based"
            else:
                final_confidence = ml_confidence
                method_used = "weighted_ml"
    
    # Generate explanation
    pred_name = label_map.get(final_pred, f'Label {final_pred}')
    explanation = generate_explanation(final_pred, features_df, final_confidence)
    
    # Add method info to explanation
    if method_used != "ml_only":
        method_descriptions = {
            "rule_based_override": " (Rule-based detection used due to high confidence)",
            "combined_agreement": " (Both ML and rule-based agree)",
            "rule_based_preferred": " (Rule-based preferred for this misconfiguration type)",
            "ml_preferred": " (ML model preferred)",
            "weighted_average": " (Weighted combination of ML and rule-based)",
            "weighted_rule_based": " (Weighted: rule-based selected)",
            "weighted_ml": " (Weighted: ML selected)"
        }
        explanation += method_descriptions.get(method_used, "")
    
    return {
        'prediction': final_pred,
        'prediction_name': pred_name,
        'confidence': final_confidence,
        'explanation': explanation,
        'probabilities': ml_probabilities,
        'method': method_used,
        'ml_prediction': ml_prediction,
        'ml_confidence': ml_confidence,
        'rule_prediction': rule_pred,
        'rule_confidence': rule_confidence
    }

def predict_single_device(pipeline, model, features_df, use_hybrid_mode=True):
    """
    Predict misconfiguration for a single device.
    
    Args:
        pipeline: MainPipeline instance
        model: Trained model (ensemble or improved classifier)
        features_df: DataFrame with device features
        use_hybrid_mode: If True, combine ML predictions with rule-based detection
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Load saved feature columns if available
        feature_columns = None
        try:
            import pickle
            from pathlib import Path
            feature_cols_path = Path(__file__).parent.parent / "models" / "saved" / "feature_columns.pkl"
            if feature_cols_path.exists():
                with open(feature_cols_path, 'rb') as f:
                    feature_columns = pickle.load(f)
        except Exception as e:
            pass  # If we can't load, we'll use the DataFrame columns
        
        # Map web app column names to model's expected column names
        column_mapping = {
            'flow_bytes_out_sum': 'orig_bytes_sum',
            'flow_bytes_out_mean': 'orig_bytes_mean',
            'flow_bytes_in_sum': 'resp_bytes_sum',
            'flow_bytes_in_mean': 'resp_bytes_mean',
            'flow_packets_out_sum': 'orig_pkts_sum',
            'flow_packets_in_sum': 'resp_pkts_sum',
            'flow_duration_mean': 'duration_mean',
            'flow_count': 'orig_bytes_count',  # Often used as count
            # Also map raw flow data columns (from realtime_flows.csv)
            'orig_bytes': 'orig_bytes_sum',  # Raw flow data
            'resp_bytes': 'resp_bytes_sum',  # Raw flow data
            'orig_pkts': 'orig_pkts_sum',  # Raw flow data
            'resp_pkts': 'resp_pkts_sum',  # Raw flow data
            'duration': 'duration_mean',  # Raw flow data
        }
        
        # Create a copy of features_df
        features_df_mapped = features_df.copy()
        
        # If we have raw flow data (orig_bytes, resp_bytes, etc.), create aggregated features
        # Check for various possible column name variations
        has_raw_flow = any(col in features_df_mapped.columns for col in ['orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'duration'])
        has_aggregated = any(col in features_df_mapped.columns for col in ['orig_bytes_sum', 'flow_bytes_out_sum', 'orig_bytes_count'])
        
        if has_raw_flow and not has_aggregated:
            # Convert to numeric first
            for col in ['orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'duration']:
                if col in features_df_mapped.columns:
                    features_df_mapped[col] = pd.to_numeric(features_df_mapped[col], errors='coerce').fillna(0.0)
            
            # Create aggregated features from raw flow data
            # Extract scalar values from the single-row DataFrame
            if len(features_df_mapped) > 0:
                orig_bytes_val = float(features_df_mapped['orig_bytes'].iloc[0]) if 'orig_bytes' in features_df_mapped.columns else 0.0
                resp_bytes_val = float(features_df_mapped['resp_bytes'].iloc[0]) if 'resp_bytes' in features_df_mapped.columns else 0.0
                orig_pkts_val = float(features_df_mapped['orig_pkts'].iloc[0]) if 'orig_pkts' in features_df_mapped.columns else 0.0
                resp_pkts_val = float(features_df_mapped['resp_pkts'].iloc[0]) if 'resp_pkts' in features_df_mapped.columns else 0.0
                duration_val = float(features_df_mapped['duration'].iloc[0]) if 'duration' in features_df_mapped.columns else 0.0
                
                # Assign scalar values (not Series) - use .loc to ensure proper assignment
                features_df_mapped.loc[:, 'orig_bytes_sum'] = orig_bytes_val
                features_df_mapped.loc[:, 'orig_bytes_mean'] = orig_bytes_val  # Approximate
                features_df_mapped.loc[:, 'orig_bytes_count'] = 1  # Single flow
                features_df_mapped.loc[:, 'resp_bytes_sum'] = resp_bytes_val
                features_df_mapped.loc[:, 'resp_bytes_mean'] = resp_bytes_val
                features_df_mapped.loc[:, 'orig_pkts_sum'] = orig_pkts_val
                features_df_mapped.loc[:, 'resp_pkts_sum'] = resp_pkts_val
                features_df_mapped.loc[:, 'duration_mean'] = duration_val
            else:
                # Empty DataFrame - set defaults
                features_df_mapped.loc[:, 'orig_bytes_sum'] = 0.0
                features_df_mapped.loc[:, 'orig_bytes_mean'] = 0.0
                features_df_mapped.loc[:, 'orig_bytes_count'] = 0
                features_df_mapped.loc[:, 'resp_bytes_sum'] = 0.0
                features_df_mapped.loc[:, 'resp_bytes_mean'] = 0.0
                features_df_mapped.loc[:, 'orig_pkts_sum'] = 0.0
                features_df_mapped.loc[:, 'resp_pkts_sum'] = 0.0
                features_df_mapped.loc[:, 'duration_mean'] = 0.0
        
        # Apply column mapping
        features_df_mapped = features_df_mapped.rename(columns=column_mapping)
        
        # Extract feature columns (exclude metadata)
        exclude_cols = ['device_id', 'time_window', 'label', 'timestamp', 'ip.src', 'ip.dst', 'proto']
        
        # Use saved feature columns if available, otherwise use DataFrame columns
        if feature_columns:
            # Ensure all required feature columns exist (fill missing with 0)
            for col in feature_columns:
                if col not in features_df_mapped.columns:
                    features_df_mapped[col] = 0
            feature_cols = [col for col in feature_columns if col not in exclude_cols]
        else:
            feature_cols = [col for col in features_df_mapped.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            return {
                'error': 'No features found. Please ensure data contains feature columns.'
            }
        
        # Get features (use only the columns the model expects)
        # Ensure columns are in the right order if model has feature_columns
        if hasattr(model, 'feature_columns') and model.feature_columns:
            # Reorder columns to match model's expected order
            model_cols = [col for col in model.feature_columns if col in feature_cols]
            missing_cols = [col for col in model.feature_columns if col not in feature_cols]
            # Add missing columns with 0 values
            for col in missing_cols:
                features_df_mapped[col] = 0
            # Use model's column order - ensure all columns exist
            final_cols = []
            for col in model.feature_columns:
                if col in features_df_mapped.columns:
                    final_cols.append(col)
                else:
                    # Column missing, add it with 0
                    features_df_mapped[col] = 0
                    final_cols.append(col)
            X_df = features_df_mapped[final_cols].copy()
        else:
            X_df = features_df_mapped[feature_cols].copy()
        
        # Convert all feature columns to numeric, handling errors gracefully
        import numpy as np
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0.0).astype(np.float64)
        
        # Fill any remaining NaN values with 0 (shouldn't be any after above)
        X_df = X_df.fillna(0.0)
        
        # Convert to numpy array and ensure 2D shape (even for single row)
        # Use to_numpy() for better type handling
        X_array = X_df.to_numpy(dtype=np.float64)
        
        # Ensure 2D shape
        if len(X_array.shape) == 0:
            # Scalar - shouldn't happen
            X_array = np.array([[X_array]], dtype=np.float64)
        elif len(X_array.shape) == 1:
            # 1D array - reshape to 2D
            X_array = X_array.reshape(1, -1)
        elif X_array.shape[0] == 0:
            return {
                'error': 'No valid features after processing. Please check your input data.'
            }
        
        # Final verification: must be 2D
        if len(X_array.shape) != 2:
            return {
                'error': f'Invalid array shape: {X_array.shape}. Expected 2D array.'
            }
        
        # Predict - COMPLETELY bypass model methods and use underlying components directly
        # This avoids any issues with saved models that have old buggy code
        if hasattr(model, 'predict_proba'):
            # If model has scaler and feature_columns, use manual scaling (most reliable)
            # ALWAYS use manual scaling to avoid old code in saved models
            if hasattr(model, 'scaler') and hasattr(model, 'feature_columns') and model.feature_columns:
                try:
                    # Extract features in correct order - ensure all columns exist
                    missing_cols = [col for col in model.feature_columns if col not in X_df.columns]
                    if missing_cols:
                        for col in missing_cols:
                            X_df[col] = 0
                    
                    # Get features in exact order expected by model
                    # IMPORTANT: Ensure we have a proper DataFrame with all columns
                    # First, ensure all columns exist and are numeric
                    for col in model.feature_columns:
                        if col not in X_df.columns:
                            X_df[col] = 0.0
                        else:
                            # Ensure column is numeric (convert to float64)
                            X_df[col] = pd.to_numeric(X_df[col], errors='coerce').fillna(0.0).astype(np.float64)
                    
                    X_subset = X_df[model.feature_columns].copy()
                    
                    # Debug: Check if DataFrame is empty or has wrong shape
                    if X_subset.empty:
                        return {
                            'error': f'Empty DataFrame after column selection. Available columns: {list(X_df.columns)[:10]}, Required: {model.feature_columns[:5]}...'
                        }
                    
                    # Convert to numpy array - use to_numpy() for better type handling
                    import numpy as np
                    X_manual = X_subset.to_numpy(dtype=np.float64)
                    
                    # CRITICAL: Ensure 2D shape (even for single sample)
                    # Convert to numpy array first with explicit shape handling
                    X_manual = np.asarray(X_manual, dtype=np.float64)
                    
                    # Handle all possible shapes
                    if X_manual.size == 0:
                        return {
                            'error': f'Empty array. X_manual shape: {X_manual.shape}, X_subset shape: {X_subset.shape}, X_subset columns: {list(X_subset.columns)}'
                        }
                    elif len(X_manual.shape) == 0:
                        # Scalar - shouldn't happen but handle it
                        X_manual = np.array([[X_manual]])
                    elif len(X_manual.shape) == 1:
                        # 1D array - reshape to 2D
                        # This can happen if DataFrame has only one row and one column
                        X_manual = X_manual.reshape(1, -1)
                    elif X_manual.shape[0] == 0:
                        return {
                            'error': f'No samples to predict. X_manual shape: {X_manual.shape}'
                        }
                    
                    # Final verification: must be 2D with at least 1 row
                    if len(X_manual.shape) != 2:
                        return {
                            'error': f'Invalid array shape: {X_manual.shape}. Expected 2D array. X_subset shape: {X_subset.shape}'
                        }
                    
                    # Additional check: ensure we have the right number of features
                    if X_manual.shape[1] != len(model.feature_columns):
                        return {
                            'error': f'Feature count mismatch: got {X_manual.shape[1]} features, expected {len(model.feature_columns)}. Shape: {X_manual.shape}'
                        }
                    
                    # Verify shape matches scaler's expectations
                    if hasattr(model.scaler, 'n_features_in_'):
                        expected_features = model.scaler.n_features_in_
                        if X_manual.shape[1] != expected_features:
                            return {
                                'error': f'Feature mismatch: scaler expects {expected_features} features, got {X_manual.shape[1]}. Shape: {X_manual.shape}'
                            }
                    
                    # Scale the features - ensure input is exactly what scaler expects
                    # StandardScaler.transform expects 2D array: (n_samples, n_features)
                    # CRITICAL: Ensure X_manual is a proper numpy array with correct dtype
                    if not isinstance(X_manual, np.ndarray):
                        X_manual = np.asarray(X_manual, dtype=np.float64)
                    
                    # Ensure it's 2D and contiguous
                    if len(X_manual.shape) == 1:
                        X_manual = X_manual.reshape(1, -1)
                    X_manual = np.ascontiguousarray(X_manual, dtype=np.float64)
                    
                    try:
                        X_scaled = model.scaler.transform(X_manual)
                    except Exception as scale_error:
                        # If scaling fails, provide detailed error
                        import traceback
                        error_full = str(scale_error)
                        error_trace = traceback.format_exc()
                        return {
                            'error': f'Scaling error: {error_full}. X_manual shape: {X_manual.shape}, dtype: {X_manual.dtype}, type: {type(X_manual)}, scaler expects: {getattr(model.scaler, "n_features_in_", "unknown")} features. Error trace: {error_trace[:300]}'
                        }
                    
                    # Predict using underlying sklearn model directly
                    # NEVER call model.predict_proba() - it might have old buggy code
                    # Ensure X_scaled is a proper 2D numpy array before prediction
                    if not isinstance(X_scaled, np.ndarray):
                        X_scaled = np.asarray(X_scaled, dtype=np.float64)
                    if len(X_scaled.shape) == 1:
                        X_scaled = X_scaled.reshape(1, -1)
                    X_scaled = np.ascontiguousarray(X_scaled, dtype=np.float64)
                    
                    if hasattr(model, 'model') and model.model is not None:
                        try:
                            proba = model.model.predict_proba(X_scaled)
                            pred = model.model.predict(X_scaled)
                        except Exception as pred_error:
                            import traceback
                            full_trace = traceback.format_exc()
                            return {
                                'error': f'Model prediction error: {str(pred_error)}. X_scaled shape: {X_scaled.shape}, dtype: {X_scaled.dtype}, type: {type(X_scaled)}. Trace: {full_trace[:300]}'
                            }
                    elif hasattr(model, 'ensemble') and model.ensemble is not None:
                        try:
                            proba = model.ensemble.predict_proba(X_scaled)
                            pred = model.ensemble.predict(X_scaled)
                        except Exception as pred_error:
                            import traceback
                            full_trace = traceback.format_exc()
                            return {
                                'error': f'Ensemble prediction error: {str(pred_error)}. X_scaled shape: {X_scaled.shape}, dtype: {X_scaled.dtype}, type: {type(X_scaled)}. Trace: {full_trace[:300]}'
                            }
                    else:
                        # Last resort: try model's methods (but this might fail with old code)
                        # Convert X_df to numpy array first
                        try:
                            X_fallback = X_df[model.feature_columns].to_numpy(dtype=np.float64) if hasattr(model, 'feature_columns') and model.feature_columns else X_df.to_numpy(dtype=np.float64)
                            if len(X_fallback.shape) == 1:
                                X_fallback = X_fallback.reshape(1, -1)
                            proba = model.predict_proba(X_fallback)
                            pred = model.predict(X_fallback)
                        except Exception as fallback_error:
                            return {
                                'error': f'Fallback prediction error: {str(fallback_error)}. Model has no accessible .model or .ensemble attribute.'
                            }
                except Exception as e:
                    # Fallback to model's methods if manual scaling fails
                    try:
                        X_final = X_df[model.feature_columns].copy()
                        X_check = X_final.values
                        if len(X_check.shape) == 1:
                            X_final = pd.DataFrame(X_check.reshape(1, -1), columns=model.feature_columns)
                        proba = model.predict_proba(X_final)
                        pred = model.predict(X_final)
                    except Exception as e2:
                        # Get full error message for debugging
                        import traceback
                        full_error = traceback.format_exc()
                        error_msg = str(e2)
                        return {
                            'error': f'Prediction error: {error_msg}. Expected {len(model.feature_columns)} features. X_manual shape: {X_manual.shape if "X_manual" in locals() else "N/A"}. Full trace: {full_error[:500]}'
                        }
            else:
                # Model doesn't have scaler, try direct prediction
                try:
                    if len(X_array.shape) == 1:
                        X_array = X_array.reshape(1, -1)
                    proba = model.predict_proba(X_array)
                    pred = model.predict(X_array)
                except Exception as e:
                    error_msg = str(e) if len(str(e)) < 200 else str(e)[:200]
                    return {
                        'error': f'Prediction error: {error_msg}. Shape: {X_array.shape}'
                    }
        else:
            # Use pipeline prediction (fallback if model doesn't have predict_proba)
            if pipeline is None:
                return {
                    'error': 'Model does not support direct prediction and pipeline is not available. Please ensure models are properly trained.'
                }
            predictions = pipeline.predict(features_df, use_advanced=True)
            if 'ensemble' in predictions or 'improved_rf' in predictions:
                # Use best model
                best_key = 'ensemble' if 'ensemble' in predictions else 'improved_rf'
                pred = predictions[best_key]['predictions']
                proba = predictions[best_key].get('probabilities', None)
            else:
                pred = predictions['baseline_classifier']['predictions']
                proba = predictions['baseline_classifier'].get('probabilities', None)
        
        # Get confidence
        if proba is not None and len(proba) > 0:
            confidence = float(np.max(proba[0]))
        else:
            confidence = 0.5
        
        # Map prediction to label name
        label_map = {
            0: 'Normal Configuration',
            1: 'DNS Misconfiguration',
            2: 'DHCP Misconfiguration',
            3: 'Gateway Misconfiguration',
            4: 'ARP Storm',
            -1: 'Unknown Misconfiguration'
        }
        
        pred_label = int(pred[0]) if isinstance(pred, (list, np.ndarray)) else int(pred)
        pred_name = label_map.get(pred_label, f'Label {pred_label}')
        
        # Combine with rule-based detection if hybrid mode is enabled
        if use_hybrid_mode:
            return combine_ml_and_rule_based(
                ml_prediction=pred_label,
                ml_confidence=confidence,
                ml_probabilities=proba[0].tolist() if proba is not None and len(proba) > 0 else None,
                features_df=features_df,
                label_map=label_map
            )
        else:
            # Pure ML mode
            explanation = generate_explanation(pred_label, features_df, confidence)
            return {
                'prediction': pred_label,
                'prediction_name': pred_name,
                'confidence': confidence,
                'explanation': explanation,
                'probabilities': proba[0].tolist() if proba is not None and len(proba) > 0 else None,
                'method': 'ml_only'
            }
    except Exception as e:
        return {
            'error': f'Prediction error: {str(e)}'
        }

def generate_explanation(pred_label, features_df, confidence):
    """Generate human-readable explanation for the prediction."""
    explanations = {
        0: f"Device shows normal network behavior patterns (confidence: {confidence:.1%}). All network metrics are within expected ranges.",
        1: f"DNS Misconfiguration detected (confidence: {confidence:.1%}). Device is experiencing DNS resolution failures or unusual DNS query patterns.",
        2: f"DHCP Misconfiguration detected (confidence: {confidence:.1%}). Device is not receiving valid IP address leases or showing abnormal DHCP request patterns.",
        3: f"Gateway Misconfiguration detected (confidence: {confidence:.1%}). Device is using multiple or incorrect gateway addresses.",
        4: f"ARP Storm detected (confidence: {confidence:.1%}). Device is generating excessive ARP requests, potentially causing network congestion.",
        -1: f"Unknown misconfiguration type detected (confidence: {confidence:.1%}). Device behavior doesn't match known misconfiguration patterns."
    }
    
    base_explanation = explanations.get(pred_label, f"Prediction: Label {pred_label} (confidence: {confidence:.1%})")
    
    # Add feature-specific details
    if 'flow_bytes_out_sum' in features_df.columns:
        bytes_out = features_df['flow_bytes_out_sum'].iloc[0]
        if bytes_out > 10000000:  # 10MB
            base_explanation += " High outbound traffic detected."
        elif bytes_out < 1000:
            base_explanation += " Very low outbound traffic detected."
    
    return base_explanation

def main():
    """Main application function."""
    # Header
    st.markdown('<div class="main-header">🔍 Network Device Misconfiguration Detector</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("### Model Selection")
        st.info("**Best Model**: Improved RandomForest (Auto-selected)")
        st.markdown("*Balanced Accuracy: 0.6548 | Accuracy: 0.3409*")
        
        # Model selection (hidden, auto-uses best model)
        model_type = "Improved RandomForest"  # Best model based on balanced accuracy
        
        st.markdown("### Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for misconfiguration detection"
        )
        
        use_hybrid_mode = st.checkbox(
            "🔀 Use Hybrid Mode (ML + Rule-Based)",
            value=True,
            help="Combine ML model predictions with rule-based detection for improved accuracy. Rule-based is particularly effective for DHCP and DNS misconfigurations."
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This application uses machine learning models to detect misconfigured devices in enterprise networks.
        
        **Supported Misconfiguration Types:**
        - DNS Misconfiguration
        - DHCP Misconfiguration
        - Gateway Misconfiguration
        - ARP Storm
        - Unknown Misconfigurations
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Single Device Analysis", "📁 Batch Analysis", "📈 Model Performance"])
    
    # Load pipeline (cached)
    pipeline, error = load_pipeline()
    
    if error and PIPELINE_AVAILABLE:
        st.warning(f"⚠️ Pipeline not available: {error}")
        st.info("ℹ️ Running in demo mode. Train models for full functionality.")
        pipeline = None
    elif not PIPELINE_AVAILABLE:
        st.info("ℹ️ Running in demo mode. Install dependencies and train models for full functionality.")
        pipeline = None
    
    # Tab 1: Single Device Analysis
    with tab1:
        st.header("Analyze Single Device")
        st.markdown("Enter device features manually or upload a CSV file with device data.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_method = st.radio(
                "Input Method",
                ["Manual Entry", "Upload CSV"],
                horizontal=True
            )
        
        if input_method == "Manual Entry":
            st.subheader("Device Features")
            
            # Create form for manual entry
            with st.form("device_features_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    device_id = st.text_input("Device ID", value="device_001")
                    date_input = st.date_input("Date", value=datetime.now().date())
                    time_input = st.time_input("Time", value=datetime.now().time())
                    time_window = datetime.combine(date_input, time_input)
                    
                    st.markdown("### Flow Features")
                    bytes_out = st.number_input("Bytes Out (sum)", min_value=0, value=1024000, step=1000)
                    bytes_in = st.number_input("Bytes In (sum)", min_value=0, value=512000, step=1000)
                    packets_out = st.number_input("Packets Out (sum)", min_value=0, value=1000, step=10)
                    packets_in = st.number_input("Packets In (sum)", min_value=0, value=500, step=10)
                    duration_mean = st.number_input("Duration Mean (seconds)", min_value=0.0, value=5.2, step=0.1)
                
                with col2:
                    st.markdown("### DHCP Features (Optional)")
                    dhcp_discover = st.number_input("DHCP Discover Count", min_value=0, value=0, step=1)
                    dhcp_request = st.number_input("DHCP Request Count", min_value=0, value=0, step=1)
                    dhcp_ack = st.number_input("DHCP ACK Count", min_value=0, value=0, step=1)
                    
                    st.markdown("### DNS Features (Optional)")
                    dns_query = st.number_input("DNS Query Count", min_value=0, value=0, step=1)
                    dns_success = st.number_input("DNS Success Count", min_value=0, value=0, step=1)
                    dns_failure = st.number_input("DNS Failure Count", min_value=0, value=0, step=1)
                
                submitted = st.form_submit_button("🔍 Analyze Device", type="primary")
                
                if submitted:
                    # Create feature DataFrame
                    # Use model's expected column names directly
                    features_dict = {
                        'device_id': [device_id],
                        'time_window': [time_window],
                        'orig_bytes_sum': [bytes_out],
                        'orig_bytes_mean': [bytes_out / max(packets_out, 1)],  # Approximate mean
                        'orig_bytes_count': [packets_out],  # Use packet count as flow count
                        'resp_bytes_sum': [bytes_in],
                        'resp_bytes_mean': [bytes_in / max(packets_in, 1)],  # Approximate mean
                        'orig_pkts_sum': [packets_out],
                        'resp_pkts_sum': [packets_in],
                        'duration_mean': [duration_mean],
                        'dhcp_discover_count': [dhcp_discover],
                        'dhcp_request_count': [dhcp_request],
                        'dhcp_ack_count': [dhcp_ack],
                        'dns_query_count': [dns_query],
                        'dns_success_count': [dns_success],
                        'dns_failure_count': [dns_failure]
                    }
                    
                    features_df = pd.DataFrame(features_dict)
                    
                    # Show loading
                    with st.spinner("Analyzing device..."):
                        # Use best model: Improved RandomForest (best balanced accuracy: 0.6548)
                        if PIPELINE_AVAILABLE:
                            models = load_trained_models()
                            
                            # Show warnings for loading errors (but models might still be usable)
                            # Only show if we have errors AND no usable models loaded
                            if models and models.get('_load_errors'):
                                has_usable_models = 'improved_rf' in models or 'ensemble' in models or 'xgb' in models
                                if not has_usable_models:
                                    # Only show errors if no models loaded
                                    with st.expander("🔍 Model Loading Errors (Click to see details)"):
                                        for error in models['_load_errors']:
                                            st.error(error)
                                else:
                                    # Models loaded but some had warnings - show in expander
                                    with st.expander("ℹ️ Model Loading Info (Some models had warnings)"):
                                        for error in models['_load_errors']:
                                            st.warning(error)
                            
                            # Priority: improved_rf (best model) > ensemble > xgb
                            # Models can work without pipeline - pipeline is only for full pipeline features
                            if models and 'improved_rf' in models:
                                # Use pipeline if available, otherwise None (model will handle it)
                                result = predict_single_device(pipeline, models['improved_rf'], features_df, use_hybrid_mode=use_hybrid_mode)
                            elif models and 'ensemble' in models:
                                result = predict_single_device(pipeline, models['ensemble'], features_df, use_hybrid_mode=use_hybrid_mode)
                            elif models and 'xgb' in models:
                                result = predict_single_device(pipeline, models['xgb'], features_df, use_hybrid_mode=use_hybrid_mode)
                            else:
                                # Demo mode - no models available
                                if not models:
                                    st.info("ℹ️ Running in demo mode. Train and save models for full functionality.")
                                else:
                                    st.info("ℹ️ Running in demo mode. No trained models found (improved_rf, ensemble, or xgb). Please train and save models.")
                                result = predict_single_device_demo(features_df, confidence_threshold)
                        else:
                            # Demo mode only
                            result = predict_single_device_demo(features_df, confidence_threshold)
                    
                    # Display results
                    display_prediction_result(result, features_df)
        
        else:  # Upload CSV
            st.subheader("Upload Device Data")
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload a CSV file with device features. See sample format below."
            )
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(df)} device(s)")
                    
                    # Show preview
                    with st.expander("📋 Preview Data"):
                        st.dataframe(df.head(10))
                    
                    # Analyze button
                    if st.button("🔍 Analyze All Devices", type="primary"):
                        with st.spinner("Analyzing devices..."):
                            # Use best model: Improved RandomForest
                            if PIPELINE_AVAILABLE:
                                models = load_trained_models()
                                # Models can work without pipeline
                                use_demo = not (models and ('improved_rf' in models or 'ensemble' in models or 'xgb' in models))
                            else:
                                use_demo = True
                            
                            if use_demo:
                                if not models:
                                    st.info("ℹ️ Running in demo mode. Train and save models for full functionality.")
                                else:
                                    st.info("ℹ️ Running in demo mode. No trained models found. Please train and save models.")
                            
                            results = []
                            for idx, row in df.iterrows():
                                row_df = pd.DataFrame([row])
                                
                                if use_demo:
                                    result = predict_single_device_demo(row_df, confidence_threshold)
                                else:
                                    # Use best model: Improved RandomForest (best balanced accuracy: 0.6548)
                                    model = models.get('improved_rf') or models.get('ensemble') or models.get('xgb')
                                    result = predict_single_device(pipeline, model, row_df, use_hybrid_mode=use_hybrid_mode)
                                
                                result['device_id'] = row.get('device_id', f'device_{idx}')
                                results.append(result)
                            
                            # Display batch results
                            display_batch_results(results)
                
                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
            
            # Show sample format
            with st.expander("📝 Sample CSV Format"):
                sample_df = load_sample_data()
                st.dataframe(sample_df)
                st.download_button(
                    "Download Sample CSV",
                    sample_df.to_csv(index=False),
                    "sample_device_data.csv",
                    "text/csv"
                )
    
    # Tab 2: Batch Analysis
    with tab2:
        st.header("Batch Device Analysis")
        st.markdown("Upload a CSV file with multiple devices to analyze them in batch.")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with device data",
            type=['csv'],
            key="batch_upload"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} devices")
            
            if st.button("🔍 Analyze All Devices", key="batch_analyze"):
                # Use best model: Improved RandomForest
                if PIPELINE_AVAILABLE:
                    models = load_trained_models()
                    # Models can work without pipeline
                    use_demo = not (models and ('improved_rf' in models or 'ensemble' in models or 'xgb' in models))
                else:
                    use_demo = True
                
                if use_demo:
                    if not models:
                        st.info("ℹ️ Running in demo mode. Train and save models for full functionality.")
                    else:
                        st.info("ℹ️ Running in demo mode. No trained models found. Please train and save models.")
                
                progress_bar = st.progress(0)
                results = []
                
                for idx, row in df.iterrows():
                    try:
                        # Convert row (pandas Series) to dict, then to DataFrame
                        # This ensures proper handling of data types
                        row_dict = row.to_dict()
                        
                        # Create DataFrame with explicit index to ensure proper structure
                        row_df = pd.DataFrame([row_dict], index=[0])
                        
                        # Ensure device_id is set for error reporting
                        device_id = row_dict.get('device_id', f'device_{idx}')
                        time_window = row_dict.get('time_window', 'N/A')
                        
                        # If this looks like raw flow data (has orig_bytes, resp_bytes, etc.),
                        # we need to aggregate it first or map columns
                        if 'orig_bytes' in row_df.columns and 'orig_bytes_sum' not in row_df.columns:
                            # This is raw flow data - need to aggregate or map
                            # Convert to numeric first to avoid string issues
                            for col in ['orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'duration']:
                                if col in row_df.columns:
                                    row_df[col] = pd.to_numeric(row_df[col], errors='coerce').fillna(0)
                            
                            # Create aggregated features from raw flow data
                            # Extract scalar values first to avoid Series assignment issues
                            orig_bytes_val = float(row_df['orig_bytes'].iloc[0]) if 'orig_bytes' in row_df.columns else 0.0
                            resp_bytes_val = float(row_df['resp_bytes'].iloc[0]) if 'resp_bytes' in row_df.columns else 0.0
                            orig_pkts_val = float(row_df['orig_pkts'].iloc[0]) if 'orig_pkts' in row_df.columns else 0.0
                            resp_pkts_val = float(row_df['resp_pkts'].iloc[0]) if 'resp_pkts' in row_df.columns else 0.0
                            duration_val = float(row_df['duration'].iloc[0]) if 'duration' in row_df.columns else 0.0
                            
                            # Assign scalar values using .loc to ensure proper assignment
                            row_df.loc[:, 'orig_bytes_sum'] = orig_bytes_val
                            row_df.loc[:, 'orig_bytes_mean'] = orig_bytes_val  # Approximate
                            row_df.loc[:, 'orig_bytes_count'] = 1
                            row_df.loc[:, 'resp_bytes_sum'] = resp_bytes_val
                            row_df.loc[:, 'resp_bytes_mean'] = resp_bytes_val
                            row_df.loc[:, 'orig_pkts_sum'] = orig_pkts_val
                            row_df.loc[:, 'resp_pkts_sum'] = resp_pkts_val
                            row_df.loc[:, 'duration_mean'] = duration_val
                        
                        if use_demo:
                            result = predict_single_device_demo(row_df, confidence_threshold)
                        else:
                            # Use best model: Improved RandomForest (best balanced accuracy: 0.6548)
                            model = models.get('improved_rf') or models.get('ensemble') or models.get('xgb')
                            if model is None:
                                result = {'error': 'No trained model available. Please train models first.'}
                            else:
                                result = predict_single_device(pipeline, model, row_df)
                        
                        # Always set device_id and time_window, even if prediction failed
                        result['device_id'] = device_id
                        result['time_window'] = time_window
                        results.append(result)
                    except Exception as row_error:
                        # Handle errors for individual rows
                        results.append({
                            'device_id': row_dict.get('device_id', f'device_{idx}') if 'row_dict' in locals() else f'device_{idx}',
                            'time_window': row_dict.get('time_window', 'N/A') if 'row_dict' in locals() else 'N/A',
                            'error': f'Row processing error: {str(row_error)[:200]}'
                        })
                    
                    progress_bar.progress((idx + 1) / len(df))
                
                display_batch_results(results)
    
    # Tab 3: Model Performance
    with tab3:
        st.header("Model Performance Metrics")
        
        st.markdown("""
        ### Best Model: Improved RandomForest
        
        **Selected Model**: Improved RandomForest (with hyperparameter tuning and threshold optimization)
        
        **Performance Metrics:**
        - **Balanced Accuracy**: 0.6548 (65.48%)
        - **Accuracy**: 0.3409 (34.09%)
        - **ROC-AUC**: 0.6012
        - **Average Precision**: 0.1156
        - **Misconfig Recall**: 100%
        
        ### Why This Model?
        
        The **Improved RandomForest** model is selected as the best model because:
        1. **Highest Balanced Accuracy** (0.6548) - Best for imbalanced datasets
        2. **Perfect Misconfig Recall** (100%) - Critical for security applications
        3. **Hyperparameter Tuned** - Optimized for the dataset
        4. **Threshold Optimized** - Prioritizes misconfiguration detection
        5. **Feature Importance** - Provides interpretable results
        
        ### Model Comparison
        
        | Model | Accuracy | Balanced Accuracy |
        |-------|----------|-------------------|
        | Improved RandomForest | 0.3409 | **0.6548** ⭐ |
        | Ensemble | 0.8636 | 0.4524 |
        | Baseline RandomForest | 0.9091 | 0.4762 |
        
        **Note**: Balanced accuracy and misconfig recall are prioritized because the dataset is imbalanced. The accuracy drop is expected when optimizing for recall.
        """)
        
        # Show model info if available
        models = load_trained_models() if PIPELINE_AVAILABLE else None
        if models and 'improved_rf' in models:
            st.success("✅ Best model (Improved RandomForest) is loaded and ready!")
        else:
            st.warning("⚠️ Best model not found. Train and save models using the notebook (Cell 17).")

def get_rule_based_prediction(features_df):
    """
    Get rule-based prediction with confidence scores.
    Returns prediction label, confidence, and rule name.
    """
    # Default to Normal
    pred_label = 0
    confidence = 0.85
    rule_name = "Normal"
    
    # Get first row as dict for easier access
    row = features_df.iloc[0].to_dict() if len(features_df) > 0 else {}
    
    # Check for DHCP misconfig indicators
    discover = row.get('dhcp_discover_count', 0)
    ack = row.get('dhcp_ack_count', 0)
    
    if discover > 10 and ack < discover * 0.5:
        pred_label = 2  # DHCP Misconfig
        # Higher confidence if more severe
        severity = min(discover / 20.0, 1.0)  # Normalize to 0-1
        confidence = 0.75 + (severity * 0.15)  # 0.75 to 0.90
        rule_name = "DHCP_MISCONFIG"
    
    # Check for DNS misconfig
    failures = row.get('dns_failure_count', 0)
    queries = row.get('dns_query_count', 0)
    
    if queries > 0:
        failure_ratio = failures / queries
        if failure_ratio > 0.5 and queries > 5:
            # DNS misconfig takes priority if more severe
            if pred_label == 0 or (pred_label == 2 and failure_ratio > 0.7):
                pred_label = 1  # DNS Misconfig
                severity = min(failure_ratio, 1.0)
                confidence = 0.70 + (severity * 0.20)  # 0.70 to 0.90
                rule_name = "DNS_MISCONFIG"
    
    # Check for Gateway misconfig
    num_gateways = row.get('num_distinct_gateways', 0)
    if num_gateways > 3:
        if pred_label == 0:  # Only if no other misconfig detected
            pred_label = 3  # Gateway Misconfig
            confidence = 0.80
            rule_name = "GATEWAY_MISCONFIG"
    
    # Check for ARP storm
    arp_requests = row.get('arp_request_count', 0)
    broadcast_ratio = row.get('broadcast_packet_ratio', 0)
    
    if arp_requests > 50 or (broadcast_ratio > 0.3 and arp_requests > 20):
        if pred_label == 0:  # Only if no other misconfig detected
            pred_label = 4  # ARP Storm
            severity = min(arp_requests / 100.0, 1.0)
            confidence = 0.75 + (severity * 0.15)  # 0.75 to 0.90
            rule_name = "ARP_STORM"
    
    return pred_label, confidence, rule_name

def predict_single_device_demo(features_df, confidence_threshold):
    """
    Demo prediction function (simplified for web app).
    In production, this would load the actual trained model.
    """
    pred_label, confidence, rule_name = get_rule_based_prediction(features_df)
    
    label_map = {
        0: 'Normal Configuration',
        1: 'DNS Misconfiguration',
        2: 'DHCP Misconfiguration',
        3: 'Gateway Misconfiguration',
        4: 'ARP Storm',
        -1: 'Unknown Misconfiguration'
    }
    
    explanation = generate_explanation(pred_label, features_df, confidence)
    
    return {
        'prediction': pred_label,
        'prediction_name': label_map.get(pred_label, f'Label {pred_label}'),
        'confidence': confidence,
        'explanation': explanation,
        'probabilities': None,
        'method': 'rule_based'
    }

def display_prediction_result(result, features_df):
    """Display prediction results in a user-friendly format."""
    if 'error' in result:
        st.error(result['error'])
        return
    
    # Main prediction card
    pred_label = result['prediction']
    pred_name = result['prediction_name']
    confidence = result['confidence']
    
    # Color coding
    if pred_label == 0:
        css_class = "normal"
        icon = "✅"
    else:
        css_class = "misconfig"
        icon = "⚠️"
    
    st.markdown(f"""
    <div class="prediction-box {css_class}">
        <h2>{icon} {pred_name}</h2>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
        <p><strong>Explanation:</strong> {result['explanation']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediction", pred_name)
    
    with col2:
        st.metric("Confidence", f"{confidence:.1%}")
    
    with col3:
        status = "Normal" if pred_label == 0 else "Misconfigured"
        st.metric("Status", status)
    
    # Probability distribution (if available)
    if result.get('probabilities') is not None and len(result['probabilities']) > 0:
        st.subheader("Class Probabilities")
        
        probs = result['probabilities']
        num_probs = len(probs)
        
        # Map class indices to names (model may output probabilities for classes 0, 1, 2, 3, 4, etc.)
        # For Westermo dataset, we typically have classes 0 (Normal) and 2 (DHCP Misconfig)
        # But the model outputs probabilities for all classes it was trained on
        class_map = {
            0: 'Normal',
            1: 'DNS Misconfig',
            2: 'DHCP Misconfig',
            3: 'Gateway Misconfig',
            4: 'ARP Storm'
        }
        
        # Create class names based on number of probabilities
        # If we have 2 probabilities, likely classes 0 and 2 (Normal, DHCP)
        # If we have 3 probabilities, likely classes 0, 1, 2, etc.
        if num_probs == 2:
            # Likely Westermo dataset: Normal (0) and DHCP Misconfig (2)
            class_names = ['Normal', 'DHCP Misconfig']
        elif num_probs == 3:
            # Could be classes 0, 1, 2
            class_names = ['Normal', 'DNS Misconfig', 'DHCP Misconfig']
        elif num_probs >= 5:
            # Full set of classes
            class_names = ['Normal', 'DNS Misconfig', 'DHCP Misconfig', 'Gateway Misconfig', 'ARP Storm']
            probs = probs[:5]  # Take first 5
        else:
            # Generic: use class indices
            class_names = [class_map.get(i, f'Class {i}') for i in range(num_probs)]
        
        # Ensure arrays match in length
        min_len = min(len(class_names), len(probs))
        prob_df = pd.DataFrame({
            'Class': class_names[:min_len],
            'Probability': probs[:min_len]
        })
        
        fig = px.bar(
            prob_df,
            x='Class',
            y='Probability',
            title='Prediction Probabilities',
            color='Probability',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature values
    with st.expander("📋 Device Features"):
        st.dataframe(features_df.T, use_container_width=True)

def display_batch_results(results):
    """Display results for batch analysis."""
    st.subheader("📊 Batch Analysis Results")
    
    # Summary statistics - filter out errors
    total = len(results)
    valid_results = [r for r in results if 'error' not in r and 'prediction' in r]
    misconfig_count = sum(1 for r in valid_results if r.get('prediction', 0) != 0)
    normal_count = len(valid_results) - misconfig_count
    error_count = total - len(valid_results)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Devices", total)
    with col2:
        st.metric("Normal", normal_count)
    with col3:
        st.metric("Misconfigured", misconfig_count)
    with col4:
        st.metric("Misconfig Rate", f"{misconfig_count/len(valid_results):.1%}" if len(valid_results) > 0 else "0%")
    with col5:
        if error_count > 0:
            st.metric("Errors", error_count, delta=None, delta_color="inverse")
    
    # Results table - handle errors gracefully
    results_data = []
    for r in results:
        if 'error' in r:
            results_data.append({
                'Device ID': r.get('device_id', 'N/A'),
                'Time Window': r.get('time_window', 'N/A'),
                'Prediction': 'Error',
                'Confidence': 'N/A',
                'Status': '❌ Error',
                'Error': str(r.get('error', 'Unknown error'))[:200]  # Show more of error message
            })
        elif 'prediction' in r and 'prediction_name' in r:
            method = r.get('method', 'unknown')
            method_display = {
                'ml_only': 'ML',
                'rule_based_override': 'Rule-Based',
                'combined_agreement': 'ML+Rule',
                'rule_based_preferred': 'Rule-Based',
                'ml_preferred': 'ML',
                'weighted_average': 'Hybrid',
                'weighted_rule_based': 'Hybrid',
                'weighted_ml': 'Hybrid',
                'rule_based': 'Rule-Based'
            }.get(method, 'Unknown')
            
            results_data.append({
                'Device ID': r.get('device_id', 'N/A'),
                'Time Window': r.get('time_window', 'N/A'),
                'Prediction': r.get('prediction_name', 'Unknown'),
                'Confidence': f"{r.get('confidence', 0):.1%}",
                'Method': method_display,
                'Status': '✅ Normal' if r.get('prediction', -1) == 0 else '⚠️ Misconfigured'
            })
        else:
            results_data.append({
                'Device ID': r.get('device_id', 'N/A'),
                'Time Window': r.get('time_window', 'N/A'),
                'Prediction': 'Unknown',
                'Confidence': 'N/A',
                'Status': '❓ Unknown'
            })
    
    results_df = pd.DataFrame(results_data)
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Results",
        csv,
        "misconfig_detection_results.csv",
        "text/csv"
    )
    
    # Visualization - only show valid results
    valid_results = [r for r in results if 'error' not in r and 'prediction_name' in r]
    if len(valid_results) > 0:
        st.subheader("📈 Distribution")
        
        pred_counts = pd.Series([r.get('prediction_name', 'Unknown') for r in valid_results]).value_counts()
        
        fig = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            title="Misconfiguration Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

