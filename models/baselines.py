"""
Baseline Models
Rule-based detector, RandomForest classifier, and IsolationForest anomaly detector.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class RuleBasedDetector:
    """Rule-based misconfiguration detector."""
    
    def __init__(self):
        """Initialize rule-based detector."""
        self.rules = []
        self._setup_rules()
    
    def _setup_rules(self):
        """Setup detection rules."""
        # DHCP misconfig rules
        self.rules.append({
            'name': 'DHCP_MISCONFIG',
            'condition': lambda row: (
                row.get('dhcp_discover_count', 0) > 10 and
                row.get('dhcp_ack_count', 0) < row.get('dhcp_discover_count', 0) * 0.5
            ),
            'label': 2  # DHCP misconfig
        })
        
        # DNS misconfig rules
        self.rules.append({
            'name': 'DNS_MISCONFIG',
            'condition': lambda row: (
                row.get('dns_failure_ratio', 0) > 0.5 and
                row.get('dns_query_count', 0) > 5
            ),
            'label': 1  # DNS misconfig
        })
        
        # ARP storm rules
        self.rules.append({
            'name': 'ARP_STORM',
            'condition': lambda row: (
                row.get('arp_request_count', 0) > 50 or
                (row.get('broadcast_packet_ratio', 0) > 0.3 and row.get('arp_request_count', 0) > 20)
            ),
            'label': 4  # ARP storm
        })
        
        # Gateway misconfig rules
        self.rules.append({
            'name': 'GATEWAY_MISCONFIG',
            'condition': lambda row: (
                row.get('num_distinct_gateways', 0) > 3
            ),
            'label': 3  # Gateway misconfig
        })
    
    def predict(self, df):
        """
        Apply rules to detect misconfigurations.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with predictions: device_id, time_window, predicted_label, rule_name
        """
        results = []
        
        for idx, row in df.iterrows():
            predicted_label = 0  # Normal by default
            rule_name = 'NORMAL'
            
            for rule in self.rules:
                if rule['condition'](row):
                    predicted_label = rule['label']
                    rule_name = rule['name']
                    break
            
            results.append({
                'device_id': row.get('device_id', 'unknown'),
                'time_window': row.get('time_window', None),
                'predicted_label': predicted_label,
                'rule_name': rule_name,
                'confidence': 1.0 if predicted_label != 0 else 0.0
            })
        
        return pd.DataFrame(results)


class BaselineClassifier:
    """Baseline RandomForest classifier."""
    
    def __init__(self, n_estimators=300, max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42):
        """
        Initialize baseline classifier with improved defaults.
        
        Args:
            n_estimators: Number of trees (increased from 200 to 300)
            max_depth: Maximum tree depth (increased from 10 to 12)
            min_samples_split: Minimum samples to split (added)
            min_samples_leaf: Minimum samples in leaf (added)
            random_state: Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def fit(self, X, y, feature_columns=None):
        """
        Train the classifier.
        
        Args:
            X: Feature DataFrame or numpy array
            y: Labels
            feature_columns: List of feature column names (if X is DataFrame)
        """
        if isinstance(X, pd.DataFrame):
            if feature_columns is None:
                # Exclude non-feature columns
                exclude_cols = ['device_id', 'time_window', 'label']
                feature_columns = [col for col in X.columns if col not in exclude_cols]
            self.feature_columns = feature_columns
            X = X[feature_columns].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        """
        Predict labels.
        
        Args:
            X: Feature DataFrame or numpy array
            
        Returns:
            Predicted labels
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict label probabilities.
        
        Args:
            X: Feature DataFrame or numpy array
            
        Returns:
            Probability matrix
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class BaselineAnomalyDetector:
    """Baseline IsolationForest anomaly detector."""
    
    def __init__(self, contamination=0.05, random_state=42):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random seed
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def fit(self, X, feature_columns=None):
        """
        Train on normal data only.
        
        Args:
            X: Feature DataFrame or numpy array (normal data only)
            feature_columns: List of feature column names (if X is DataFrame)
        """
        if isinstance(X, pd.DataFrame):
            if feature_columns is None:
                exclude_cols = ['device_id', 'time_window', 'label']
                feature_columns = [col for col in X.columns if col not in exclude_cols]
            self.feature_columns = feature_columns
            X = X[feature_columns].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
    
    def predict(self, X):
        """
        Predict anomalies.
        
        Args:
            X: Feature DataFrame or numpy array
            
        Returns:
            -1 for anomalies, 1 for normal
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def decision_function(self, X):
        """
        Get anomaly scores (lower = more anomalous).
        
        Args:
            X: Feature DataFrame or numpy array
            
        Returns:
            Anomaly scores
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

