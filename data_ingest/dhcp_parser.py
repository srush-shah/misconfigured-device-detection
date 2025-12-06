"""
DHCP Log Parser
Extracts DHCP-related features from log files.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import re


class DHCPParser:
    """Parse DHCP logs and extract per-device, per-window statistics."""
    
    def __init__(self, time_window_minutes=5):
        """
        Initialize DHCP parser.
        
        Args:
            time_window_minutes: Size of time window in minutes (default: 5)
        """
        self.time_window_minutes = time_window_minutes
    
    def parse_log_file(self, log_file_path):
        """
        Parse DHCP log file.
        
        Expected format: timestamp, device_id, message_type, result
        Example: 2024-01-01 10:00:00, 192.168.1.100, DISCOVER, success
        
        Args:
            log_file_path: Path to DHCP log file
            
        Returns:
            DataFrame with columns: timestamp, device_id, message_type, result
        """
        try:
            df = pd.read_csv(log_file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error parsing DHCP log: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['timestamp', 'device_id', 'message_type', 'result'])
    
    def extract_features(self, df):
        """
        Extract per-device, per-window DHCP features.
        
        Args:
            df: DataFrame with DHCP log entries
            
        Returns:
            DataFrame with features per (device_id, time_window)
        """
        if df.empty:
            return pd.DataFrame()
        
        # Create time windows
        df['time_window'] = df['timestamp'].dt.floor(f'{self.time_window_minutes}min')
        
        # Group by device and time window
        features = []
        
        for (device_id, window), group in df.groupby(['device_id', 'time_window']):
            feature_dict = {
                'device_id': device_id,
                'time_window': window,
                'dhcp_discover_count': len(group[group['message_type'] == 'DISCOVER']),
                'dhcp_request_count': len(group[group['message_type'] == 'REQUEST']),
                'dhcp_ack_count': len(group[group['message_type'] == 'ACK']),
                'dhcp_nak_count': len(group[group['message_type'] == 'NAK']),
                'dhcp_release_count': len(group[group['message_type'] == 'RELEASE']),
                'dhcp_inform_count': len(group[group['message_type'] == 'INFORM']),
                'lease_renew_count': len(group[group['message_type'] == 'REQUEST'])  # Simplified
            }
            
            # Calculate ratios
            request_count = feature_dict['dhcp_request_count']
            if request_count > 0:
                feature_dict['failed_lease_ratio'] = (
                    feature_dict['dhcp_discover_count'] + request_count - feature_dict['dhcp_ack_count']
                ) / request_count
            else:
                feature_dict['failed_lease_ratio'] = 0.0
            
            # Total DHCP messages
            feature_dict['dhcp_total_count'] = len(group)
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def process(self, log_file_path):
        """
        Complete processing pipeline.
        
        Args:
            log_file_path: Path to DHCP log file
            
        Returns:
            DataFrame with extracted features
        """
        df = self.parse_log_file(log_file_path)
        features = self.extract_features(df)
        return features

