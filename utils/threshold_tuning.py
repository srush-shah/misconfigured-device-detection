"""
Threshold Tuning Utilities
Optimize prediction thresholds for better precision/recall balance.
"""

import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score


def find_optimal_threshold(y_true, y_proba, positive_class=1, metric='f1'):
    """
    Find optimal threshold for binary classification.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        positive_class: Label of positive class
        metric: Metric to optimize ('f1', 'precision', 'recall', 'balanced_f1')
    
    Returns:
        Optimal threshold value
    """
    # Convert to binary
    y_binary = (y_true == positive_class).astype(int)
    
    # Get precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_binary, y_proba)
    
    if metric == 'f1':
        # Find threshold that maximizes F1
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
    elif metric == 'precision':
        # Find threshold that maximizes precision (while recall > 0.1)
        valid_idx = recall > 0.1
        if np.any(valid_idx):
            optimal_idx = np.argmax(precision[valid_idx])
            optimal_idx = np.where(valid_idx)[0][optimal_idx]
        else:
            optimal_idx = 0
    elif metric == 'recall':
        # Find threshold that maximizes recall (while precision > 0.1)
        valid_idx = precision > 0.1
        if np.any(valid_idx):
            optimal_idx = np.argmax(recall[valid_idx])
            optimal_idx = np.where(valid_idx)[0][optimal_idx]
        else:
            optimal_idx = 0
    elif metric == 'balanced_f1':
        # Find threshold that balances precision and recall
        # Weighted F1: 0.5 * precision + 0.5 * recall
        balanced_scores = 0.5 * precision + 0.5 * recall
        optimal_idx = np.argmax(balanced_scores)
    else:
        optimal_idx = 0
    
    if optimal_idx >= len(thresholds):
        optimal_idx = len(thresholds) - 1
    
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return optimal_threshold


def apply_threshold(y_proba, threshold, positive_class=1, classes=None):
    """
    Apply threshold to probabilities to get predictions.
    
    Args:
        y_proba: Predicted probabilities
        threshold: Threshold value
        positive_class: Label of positive class
        classes: Class labels (if y_proba is multi-class)
    
    Returns:
        Binary predictions
    """
    if classes is None:
        # Assume binary classification
        return (y_proba >= threshold).astype(int) * positive_class
    
    # Multi-class: find index of positive class
    positive_idx = list(classes).index(positive_class) if positive_class in classes else 1
    if len(y_proba.shape) == 1:
        # Binary probabilities
        return (y_proba >= threshold).astype(int) * positive_class
    else:
        # Multi-class probabilities
        predictions = np.zeros(len(y_proba), dtype=int)
        predictions[y_proba[:, positive_idx] >= threshold] = positive_class
        return predictions


def optimize_threshold_for_class(y_true, y_proba, target_class, classes=None):
    """
    Optimize threshold for a specific class in multi-class setting.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        target_class: Class to optimize for
        classes: Class labels
    
    Returns:
        Optimal threshold and metrics
    """
    if classes is None:
        classes = np.unique(y_true)
    
    # Get probabilities for target class
    target_idx = list(classes).index(target_class) if target_class in classes else 0
    if len(y_proba.shape) == 1:
        target_proba = y_proba
    else:
        target_proba = y_proba[:, target_idx]
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_true, target_proba, target_class, metric='balanced_f1')
    
    # Apply threshold and calculate metrics
    y_pred = apply_threshold(target_proba, optimal_threshold, target_class, classes)
    
    precision = precision_score(y_true, y_pred, labels=[target_class], average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=[target_class], average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=[target_class], average='macro', zero_division=0)
    
    return {
        'threshold': optimal_threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

