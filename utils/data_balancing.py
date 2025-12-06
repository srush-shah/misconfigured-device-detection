"""
Data Balancing Utilities
Handles imbalanced datasets using resampling techniques.
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
import warnings

def _check_imblearn():
    """Check if imbalanced-learn is available."""
    try:
        from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.combine import SMOTEENN, SMOTETomek
        return True, SMOTE, ADASYN, BorderlineSMOTE, RandomUnderSampler, SMOTEENN, SMOTETomek
    except ImportError:
        return False, None, None, None, None, None, None

# Check availability at module load
IMBLEARN_AVAILABLE, SMOTE, ADASYN, BorderlineSMOTE, RandomUnderSampler, SMOTEENN, SMOTETomek = _check_imblearn()

if not IMBLEARN_AVAILABLE:
    # Don't print warning here - let the user code handle it
    pass


class DataBalancer:
    """Handle imbalanced datasets using various resampling techniques."""
    
    def __init__(self, method='smote', random_state=42):
        """
        Initialize data balancer.
        
        Args:
            method: Resampling method ('smote', 'adasyn', 'borderline_smote', 
                    'smoteenn', 'smotetomek', 'undersample', 'none')
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.random_state = random_state
        self.resampler = None
        self._init_resampler()
    
    def _init_resampler(self):
        """Initialize the resampler based on method."""
        # Re-check availability in case package was installed after import
        global IMBLEARN_AVAILABLE, SMOTE, ADASYN, BorderlineSMOTE, RandomUnderSampler, SMOTEENN, SMOTETomek
        if not IMBLEARN_AVAILABLE:
            IMBLEARN_AVAILABLE, SMOTE, ADASYN, BorderlineSMOTE, RandomUnderSampler, SMOTEENN, SMOTETomek = _check_imblearn()
        
        if not IMBLEARN_AVAILABLE:
            print("Warning: imbalanced-learn not available. Using class weights only.")
            self.resampler = None
            return
        
        if self.method == 'smote':
            # Use smaller k_neighbors for very small minority classes
            # Default is 5, but reduce if minority class is very small
            self.resampler = SMOTE(random_state=self.random_state, k_neighbors=3)
        elif self.method == 'adasyn':
            self.resampler = ADASYN(random_state=self.random_state, n_neighbors=5)
        elif self.method == 'borderline_smote':
            self.resampler = BorderlineSMOTE(random_state=self.random_state, k_neighbors=5)
        elif self.method == 'smoteenn':
            self.resampler = SMOTEENN(random_state=self.random_state)
        elif self.method == 'smotetomek':
            self.resampler = SMOTETomek(random_state=self.random_state)
        elif self.method == 'undersample':
            self.resampler = RandomUnderSampler(random_state=self.random_state)
        elif self.method == 'none':
            self.resampler = None
        else:
            print(f"Warning: Unknown method '{self.method}'. Using SMOTE.")
            self.resampler = SMOTE(random_state=self.random_state)
    
    def analyze_imbalance(self, y):
        """
        Analyze class distribution and calculate imbalance ratio.
        
        Args:
            y: Target labels (array-like)
            
        Returns:
            Dictionary with imbalance statistics
        """
        counts = Counter(y)
        total = len(y)
        
        stats = {
            'class_counts': dict(counts),
            'class_proportions': {k: v/total for k, v in counts.items()},
            'total_samples': total,
            'n_classes': len(counts)
        }
        
        if len(counts) > 1:
            max_count = max(counts.values())
            min_count = min(counts.values())
            stats['imbalance_ratio'] = max_count / min_count if min_count > 0 else float('inf')
        else:
            stats['imbalance_ratio'] = 1.0
        
        return stats
    
    def balance_data(self, X, y, feature_columns=None, sampling_strategy='auto'):
        """
        Balance the dataset using the selected resampling method.
        
        Args:
            X: Feature DataFrame
            y: Target labels (array-like)
            feature_columns: List of feature column names (if X is DataFrame)
            sampling_strategy: Target ratio for resampling:
                - 'auto': Balance to 1:1 (default SMOTE behavior)
                - float: Target ratio (e.g., 0.3 = 30% minority, 70% majority)
                - dict: Target counts per class
                - 'minority': Only oversample minority class
            
        Returns:
            Tuple of (X_resampled, y_resampled) or (X, y) if no resampling
        """
        if self.resampler is None:
            print("No resampling applied (method='none' or imbalanced-learn not available)")
            return X, y
        
        # Get feature columns
        if feature_columns is None:
            if isinstance(X, pd.DataFrame):
                # Exclude metadata columns
                exclude_cols = ['device_id', 'time_window', 'label']
                feature_columns = [col for col in X.columns if col not in exclude_cols]
            else:
                feature_columns = list(range(X.shape[1]))
        
        # Extract features
        if isinstance(X, pd.DataFrame):
            X_features = X[feature_columns].values
            X_metadata = X[['device_id', 'time_window']].copy() if 'device_id' in X.columns else None
        else:
            X_features = X
            X_metadata = None
        
        # Analyze before resampling
        print("\n" + "="*60)
        print("DATA BALANCING")
        print("="*60)
        stats_before = self.analyze_imbalance(y)
        print(f"\nBefore resampling:")
        print(f"  Total samples: {stats_before['total_samples']:,}")
        for label, count in sorted(stats_before['class_counts'].items()):
            prop = stats_before['class_proportions'][label]
            label_name = {0: 'Normal', 1: 'DNS Misconfig', 2: 'DHCP Misconfig', 
                         3: 'Gateway Misconfig', 4: 'ARP Storm'}.get(label, f'Label {label}')
            print(f"  {label_name}: {count:,} ({prop:.1%})")
        print(f"  Imbalance ratio: {stats_before['imbalance_ratio']:.2f}:1")
        
        # Check if minority class is too small for reliable SMOTE
        minority_count = min(stats_before['class_counts'].values())
        majority_count = max(stats_before['class_counts'].values())
        minority_label = min(stats_before['class_counts'], key=stats_before['class_counts'].get)
        majority_label = max(stats_before['class_counts'], key=stats_before['class_counts'].get)
        
        # If minority class has < 20 samples, use repeated undersampling instead of oversampling
        # This avoids creating synthetic samples while using the full dataset
        if minority_count < 20 and self.method == 'smote' and IMBLEARN_AVAILABLE:
            print(f"\n⚠️  Minority class has only {minority_count} samples.")
            print(f"   Using REPEATED UNDERSAMPLING to use full dataset without synthetic data.")
            
            # Calculate how many rounds we can do
            num_rounds = majority_count // minority_count
            total_majority_samples = num_rounds * minority_count
            
            print(f"   Will create {num_rounds} balanced subsets:")
            print(f"   - Each subset: {minority_count} Normal + {minority_count} DHCP Misconfig")
            print(f"   - Total: {total_majority_samples} Normal (from {majority_count} available) + {minority_count * num_rounds} DHCP Misconfig")
            print(f"   - Final ratio: {total_majority_samples}:{minority_count * num_rounds} = 1:1")
            
            # Perform repeated undersampling
            # Get indices for each class (these are indices into the feature array)
            minority_indices = np.where(y == minority_label)[0]
            majority_indices = np.where(y == majority_label)[0]
            
            # Shuffle majority indices for random selection
            np.random.seed(self.random_state)
            shuffled_majority_indices = majority_indices.copy()
            np.random.shuffle(shuffled_majority_indices)
            
            # Create multiple balanced subsets
            selected_majority_indices = []
            selected_minority_indices = []
            
            for round_num in range(num_rounds):
                start_idx = round_num * minority_count
                end_idx = start_idx + minority_count
                
                # Select non-overlapping majority samples for this round
                round_majority_indices = shuffled_majority_indices[start_idx:end_idx]
                selected_majority_indices.extend(round_majority_indices.tolist())
                
                # Use all minority samples in each round
                selected_minority_indices.extend(minority_indices.tolist())
            
            # Combine all selected indices (as numpy array for indexing)
            all_selected_indices = np.array(selected_majority_indices + selected_minority_indices)
            
            # Extract resampled data
            X_resampled = X_features[all_selected_indices]
            y_resampled = y[all_selected_indices]
            
            print(f"\n✓ Applied REPEATED UNDERSAMPLING ({num_rounds} rounds)")
            print(f"   Using {len(selected_majority_indices)} majority samples + {len(selected_minority_indices)} minority samples")
            
            # Store indices for DataFrame reconstruction
            # These are indices into the original X DataFrame (since X_features was extracted from X)
            if isinstance(X, pd.DataFrame):
                # Map feature array indices back to DataFrame indices
                # X_features was created from X[feature_columns], so indices align
                all_selected_indices = all_selected_indices  # Keep as is for now
            
            # Skip the normal resampling step since we've already done it
            # We'll reconstruct the DataFrame below
        else:
            # Normal resampling (SMOTE, ADASYN, etc.)
            # Apply resampling
            try:
                X_resampled, y_resampled = self.resampler.fit_resample(X_features, y)
                # Check if we used undersampling
                if hasattr(self.resampler, '__class__') and 'UnderSampler' in self.resampler.__class__.__name__:
                    resampling_type = "UNDERSAMPLING"
                else:
                    resampling_type = self.method.upper()
                print(f"\n✓ Applied {resampling_type}")
            except Exception as e:
                print(f"⚠️  Error during resampling: {e}")
                print("  Returning original data")
                return X, y
        
        # Analyze after resampling
        stats_after = self.analyze_imbalance(y_resampled)
        print(f"\nAfter resampling:")
        print(f"  Total samples: {stats_after['total_samples']:,}")
        for label, count in sorted(stats_after['class_counts'].items()):
            prop = stats_after['class_proportions'][label]
            label_name = {0: 'Normal', 1: 'DNS Misconfig', 2: 'DHCP Misconfig',
                         3: 'Gateway Misconfig', 4: 'ARP Storm'}.get(label, f'Label {label}')
            print(f"  {label_name}: {count:,} ({prop:.1%})")
        print(f"  Imbalance ratio: {stats_after['imbalance_ratio']:.2f}:1")
        
        # Reconstruct DataFrame if original was DataFrame
        if isinstance(X, pd.DataFrame):
            # Create new DataFrame with resampled features
            X_resampled_df = pd.DataFrame(X_resampled, columns=feature_columns)
            
            # If we have metadata, try to preserve it
            if X_metadata is not None:
                # Check if we did repeated undersampling
                if minority_count < 20 and self.method == 'smote' and IMBLEARN_AVAILABLE and 'all_selected_indices' in locals():
                    # For repeated undersampling, map back to original DataFrame indices
                    # all_selected_indices are indices into X_features array, which corresponds to X DataFrame rows
                    # Since X_features was extracted from X[feature_columns], the indices align directly
                    original_df_indices = X.index[all_selected_indices]
                    X_metadata_resampled = X.loc[original_df_indices, ['device_id', 'time_window']].reset_index(drop=True)
                else:
                    # For normal resampling (SMOTE, etc.)
                    n_original = len(X)
                    n_resampled = len(X_resampled)
                    n_new = n_resampled - n_original
                    
                    if n_new > 0:
                        # Duplicate metadata for synthetic samples
                        # Sample from existing metadata to create synthetic device IDs
                        np.random.seed(self.random_state)
                        synthetic_indices = np.random.choice(len(X_metadata), n_new, replace=True)
                        synthetic_metadata = X_metadata.iloc[synthetic_indices].copy()
                        # Mark as synthetic
                        synthetic_metadata['device_id'] = synthetic_metadata['device_id'].astype(str) + '_synthetic'
                        X_metadata_resampled = pd.concat([X_metadata, synthetic_metadata], ignore_index=True)
                    else:
                        X_metadata_resampled = X_metadata.iloc[:n_resampled].copy()
                
                X_resampled_df = pd.concat([X_metadata_resampled.reset_index(drop=True), 
                                           X_resampled_df.reset_index(drop=True)], axis=1)
            
            # Add label column
            X_resampled_df['label'] = y_resampled
            
            return X_resampled_df, y_resampled
        else:
            return X_resampled, y_resampled
    
    def get_class_weights(self, y):
        """
        Calculate class weights for use with sklearn models.
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary of class weights or 'balanced' string
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))


def create_balanced_split(X, y, test_size=0.2, random_state=42, 
                         balance_method='smote', balance_train_only=True, sampling_strategy='auto'):
    """
    Create train/test split with optional data balancing.
    
    Args:
        X: Feature DataFrame or array
        y: Target labels
        test_size: Proportion of test set
        random_state: Random seed
        balance_method: Resampling method ('smote', 'adasyn', 'none', etc.)
        balance_train_only: If True, only balance training set (recommended)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # First, create stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nInitial split (stratified):")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    
    # Balance training data if requested
    if balance_method != 'none' and balance_train_only:
        balancer = DataBalancer(method=balance_method, random_state=random_state)
        
        # Get feature columns if X is DataFrame
        feature_columns = None
        if isinstance(X_train, pd.DataFrame):
            exclude_cols = ['device_id', 'time_window', 'label']
            feature_columns = [col for col in X_train.columns if col not in exclude_cols]
        
        X_train_balanced, y_train_balanced = balancer.balance_data(
            X_train, y_train, feature_columns=feature_columns, sampling_strategy=sampling_strategy
        )
        
        # Ensure label column is in X_train_balanced if it's a DataFrame
        if isinstance(X_train_balanced, pd.DataFrame) and 'label' not in X_train_balanced.columns:
            X_train_balanced['label'] = y_train_balanced
        
        return X_train_balanced, X_test, y_train_balanced, y_test
    
    # If no resampling, ensure label columns are preserved
    if isinstance(X_train, pd.DataFrame) and 'label' not in X_train.columns:
        X_train['label'] = y_train
    if isinstance(X_test, pd.DataFrame) and 'label' not in X_test.columns:
        X_test['label'] = y_test
    
    return X_train, X_test, y_train, y_test

