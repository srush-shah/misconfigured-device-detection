"""
ML Models module for Use Case 7.
Contains baseline and advanced models.
"""

from .baselines import RuleBasedDetector, BaselineClassifier, BaselineAnomalyDetector
from .lstm_autoencoder import LSTMAutoencoder, LSTMAutoencoderTrainer, SequenceDataset
from .multi_view_fusion import MultiViewFusionModel
from .open_set_detector import OpenSetDetector
from .improved_classifiers import ImprovedRandomForest, ImprovedXGBoost, EnsembleClassifier
from .improved_lstm_autoencoder import ImprovedLSTMAutoencoder, ImprovedLSTMAutoencoderTrainer
from .improved_multi_view_fusion import ImprovedMultiViewFusionModel
from .improved_open_set_detector import ImprovedOpenSetDetector

__all__ = [
    'RuleBasedDetector',
    'BaselineClassifier',
    'BaselineAnomalyDetector',
    'LSTMAutoencoder',
    'LSTMAutoencoderTrainer',
    'SequenceDataset',
    'MultiViewFusionModel',
    'OpenSetDetector',
    'ImprovedRandomForest',
    'ImprovedXGBoost',
    'EnsembleClassifier',
    'ImprovedLSTMAutoencoder',
    'ImprovedLSTMAutoencoderTrainer',
    'ImprovedMultiViewFusionModel',
    'ImprovedOpenSetDetector'
]

