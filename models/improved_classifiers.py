"""
Improved Classifiers with Hyperparameter Tuning
Enhanced models for better performance on imbalanced data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class ImprovedRandomForest:
    """RandomForest with hyperparameter tuning for imbalanced data."""
    
    def __init__(self, random_state=42, cv_folds=5):
        """
        Initialize improved RandomForest.
        
        Args:
            random_state: Random seed
            cv_folds: Number of CV folds for hyperparameter tuning
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.best_params_ = None
    
    def fit(self, X, y, feature_columns=None, tune_hyperparameters=True):
        """
        Train with optional hyperparameter tuning.
        
        Args:
            X: Feature DataFrame or numpy array
            y: Labels
            feature_columns: List of feature column names
            tune_hyperparameters: If True, perform GridSearchCV
        """
        if isinstance(X, pd.DataFrame):
            if feature_columns is None:
                exclude_cols = ['device_id', 'time_window', 'label']
                feature_columns = [col for col in X.columns if col not in exclude_cols]
            self.feature_columns = feature_columns
            X = X[feature_columns].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        if tune_hyperparameters:
            print("  Tuning hyperparameters...")
            # Use F1-macro as scoring (better for imbalanced data)
            f1_macro_scorer = make_scorer(f1_score, average='macro')
            
            # Parameter grid optimized for imbalanced data
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            base_model = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1
            )
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=f1_macro_scorer,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            print(f"  Best params: {self.best_params_}")
        else:
            # Use default optimized parameters
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            self.model.fit(X_scaled, y)
    
    def predict(self, X):
        """Predict labels."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        # Ensure X is 2D (even for single sample)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        # Ensure X is 2D (even for single sample)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importances(self):
        """Get feature importances."""
        if self.model is None or self.feature_columns is None:
            return None
        return dict(zip(self.feature_columns, self.model.feature_importances_))
    
    def predict_with_threshold(self, X, threshold=0.5, positive_class=1):
        """
        Predict with custom threshold for better precision/recall balance.
        
        Args:
            X: Feature DataFrame or numpy array
            threshold: Probability threshold
            positive_class: Label of positive class
        
        Returns:
            Predictions with threshold applied
        """
        proba = self.predict_proba(X)
        classes = self.model.classes_
        
        # Find index of positive class
        positive_idx = list(classes).index(positive_class) if positive_class in classes else 1
        
        # Apply threshold
        predictions = np.zeros(len(proba), dtype=int)
        predictions[proba[:, positive_idx] >= threshold] = positive_class
        
        return predictions


class ImprovedXGBoost:
    """XGBoost classifier optimized for imbalanced data."""
    
    def __init__(self, random_state=42, cv_folds=5):
        """
        Initialize XGBoost classifier.
        
        Args:
            random_state: Random seed
            cv_folds: Number of CV folds
        """
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.best_params_ = None
    
    def fit(self, X, y, feature_columns=None, tune_hyperparameters=True):
        """
        Train XGBoost with hyperparameter tuning.
        
        Args:
            X: Feature DataFrame or numpy array
            y: Labels
            feature_columns: List of feature column names
            tune_hyperparameters: If True, perform GridSearchCV
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        if isinstance(X, pd.DataFrame):
            if feature_columns is None:
                exclude_cols = ['device_id', 'time_window', 'label']
                feature_columns = [col for col in X.columns if col not in exclude_cols]
            self.feature_columns = feature_columns
            X = X[feature_columns].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        if tune_hyperparameters:
            print("  Tuning XGBoost hyperparameters...")
            f1_macro_scorer = make_scorer(f1_score, average='macro')
            
            # Calculate scale_pos_weight for imbalanced data
            unique, counts = np.unique(y, return_counts=True)
            class_counts = dict(zip(unique, counts))
            if len(class_counts) == 2:
                majority_count = max(class_counts.values())
                minority_count = min(class_counts.values())
                scale_pos_weight = majority_count / minority_count
            else:
                scale_pos_weight = 1.0
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'scale_pos_weight': [scale_pos_weight, 1.0]
            }
            
            base_model = XGBClassifier(
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=f1_macro_scorer,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            self.best_params_ = grid_search.best_params_
            print(f"  Best params: {self.best_params_}")
        else:
            # Default optimized parameters
            unique, counts = np.unique(y, return_counts=True)
            class_counts = dict(zip(unique, counts))
            if len(class_counts) == 2:
                majority_count = max(class_counts.values())
                minority_count = min(class_counts.values())
                scale_pos_weight = majority_count / minority_count
            else:
                scale_pos_weight = 1.0
            
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            self.model.fit(X_scaled, y)
    
    def predict(self, X):
        """Predict labels."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        # Ensure X is 2D (even for single sample)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        # Ensure X is 2D (even for single sample)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class EnsembleClassifier:
    """Ensemble of multiple classifiers with voting."""
    
    def __init__(self, random_state=42):
        """
        Initialize ensemble classifier.
        
        Args:
            random_state: Random seed
        """
        self.random_state = random_state
        self.rf_model = ImprovedRandomForest(random_state=random_state, cv_folds=3)
        self.gb_model = None
        self.xgb_model = None
        self.ensemble = None
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def fit(self, X, y, feature_columns=None, use_xgboost=True):
        """
        Train ensemble of classifiers.
        
        Args:
            X: Feature DataFrame or numpy array
            y: Labels
            feature_columns: List of feature column names
            use_xgboost: If True, include XGBoost in ensemble
        """
        if isinstance(X, pd.DataFrame):
            if feature_columns is None:
                exclude_cols = ['device_id', 'time_window', 'label']
                feature_columns = [col for col in X.columns if col not in exclude_cols]
            self.feature_columns = feature_columns
            X = X[feature_columns].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Train RandomForest
        print("  Training RandomForest...")
        self.rf_model.fit(X, y, feature_columns=None, tune_hyperparameters=False)
        
        # Train GradientBoosting
        print("  Training GradientBoosting...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=self.random_state
        )
        self.gb_model.fit(X_scaled, y)
        
        # Train XGBoost if available
        if use_xgboost and XGBOOST_AVAILABLE:
            print("  Training XGBoost...")
            self.xgb_model = ImprovedXGBoost(random_state=self.random_state, cv_folds=3)
            self.xgb_model.fit(X, y, feature_columns=None, tune_hyperparameters=False)
        
        # Create voting ensemble
        estimators = [
            ('rf', self.rf_model.model),
            ('gb', self.gb_model)
        ]
        
        if self.xgb_model is not None:
            estimators.append(('xgb', self.xgb_model.model))
        
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # Use probability voting
            n_jobs=-1
        )
        self.ensemble.fit(X_scaled, y)
    
    def predict(self, X):
        """Predict using ensemble."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        # Ensure X is 2D (even for single sample)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.ensemble.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_columns].values
        # Ensure X is 2D (even for single sample)
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.ensemble.predict_proba(X_scaled)

