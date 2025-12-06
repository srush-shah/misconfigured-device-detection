"""
Sequence Builder
Creates time-series sequences for each device for LSTM/GRU models.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


class SequenceBuilder:
    """Build time-series sequences from feature DataFrame."""
    
    def __init__(self, sequence_length=12):
        """
        Initialize sequence builder.
        
        Args:
            sequence_length: Number of time windows per sequence (default: 12)
        """
        self.sequence_length = sequence_length
    
    def build_sequences(self, df, feature_columns, device_id_col='device_id', time_col='time_window'):
        """
        Build sequences for each device.
        
        Args:
            df: Feature DataFrame
            feature_columns: List of feature column names to include
            device_id_col: Name of device ID column
            time_col: Name of time window column
            
        Returns:
            dict mapping device_id to numpy array of shape (num_sequences, sequence_length, num_features)
        """
        sequences = defaultdict(list)
        
        # Group by device
        for device_id, device_df in df.groupby(device_id_col):
            # Sort by time
            device_df = device_df.sort_values(time_col)
            
            # Extract feature values
            feature_values = device_df[feature_columns].values
            
            # Create sliding windows
            for i in range(len(feature_values) - self.sequence_length + 1):
                sequence = feature_values[i:i + self.sequence_length]
                sequences[device_id].append(sequence)
        
        # Convert to numpy arrays
        result = {}
        for device_id, seq_list in sequences.items():
            if seq_list:
                result[device_id] = np.array(seq_list)
        
        return result
    
    def build_sequences_with_labels(self, df, feature_columns, label_col, 
                                    device_id_col='device_id', time_col='time_window'):
        """
        Build sequences with corresponding labels.
        
        Args:
            df: Feature DataFrame with labels
            feature_columns: List of feature column names
            label_col: Name of label column
            device_id_col: Name of device ID column
            time_col: Name of time window column
            
        Returns:
            tuple: (sequences_dict, labels_dict) where each maps device_id to arrays
        """
        sequences = defaultdict(list)
        labels = defaultdict(list)
        
        for device_id, device_df in df.groupby(device_id_col):
            device_df = device_df.sort_values(time_col)
            
            feature_values = device_df[feature_columns].values
            label_values = device_df[label_col].values
            
            for i in range(len(feature_values) - self.sequence_length + 1):
                sequence = feature_values[i:i + self.sequence_length]
                # Use label from the last window in the sequence
                label = label_values[i + self.sequence_length - 1]
                
                sequences[device_id].append(sequence)
                labels[device_id].append(label)
        
        # Convert to numpy arrays
        seq_result = {}
        label_result = {}
        for device_id in sequences:
            if sequences[device_id]:
                seq_result[device_id] = np.array(sequences[device_id])
                label_result[device_id] = np.array(labels[device_id])
        
        return seq_result, label_result

