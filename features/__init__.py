"""
Feature extraction and aggregation module.
Combines features from DHCP, DNS, and Flow parsers.
"""

from .feature_aggregator import FeatureAggregator
from .sequence_builder import SequenceBuilder

__all__ = ['FeatureAggregator', 'SequenceBuilder']

