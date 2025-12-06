"""
DNS Log Parser
Extracts DNS-related features from log files.
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import datetime


class DNSParser:
    """Parse DNS logs and extract per-device, per-window statistics."""
    
    def __init__(self, time_window_minutes=5):
        """
        Initialize DNS parser.
        
        Args:
            time_window_minutes: Size of time window in minutes (default: 5)
        """
        self.time_window_minutes = time_window_minutes
    
    def calculate_entropy(self, domains):
        """Calculate entropy of domain names."""
        if len(domains) == 0:
            return 0.0
        
        counter = Counter(domains)
        total = len(domains)
        entropy = 0.0
        
        for count in counter.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def parse_log_file(self, log_file_path):
        """
        Parse DNS log file.
        
        Expected format: timestamp, device_id, query_domain, response_code, response_time
        Example: 2024-01-01 10:00:00, 192.168.1.100, example.com, NOERROR, 0.05
        
        Args:
            log_file_path: Path to DNS log file
            
        Returns:
            DataFrame with columns: timestamp, device_id, query_domain, response_code, response_time
        """
        try:
            df = pd.read_csv(log_file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error parsing DNS log: {e}")
            return pd.DataFrame(columns=['timestamp', 'device_id', 'query_domain', 'response_code', 'response_time'])
    
    def extract_features(self, df):
        """
        Extract per-device, per-window DNS features.
        
        Args:
            df: DataFrame with DNS log entries
            
        Returns:
            DataFrame with features per (device_id, time_window)
        """
        if df.empty:
            return pd.DataFrame()
        
        # Create time windows
        df['time_window'] = df['timestamp'].dt.floor(f'{self.time_window_minutes}min')
        
        features = []
        
        for (device_id, window), group in df.groupby(['device_id', 'time_window']):
            # Success/failure counts
            success_codes = ['NOERROR', 'NXDOMAIN']  # NXDOMAIN is technically a valid response
            failures = group[~group['response_code'].isin(success_codes)]
            nxdomain = group[group['response_code'] == 'NXDOMAIN']
            
            domains = group['query_domain'].tolist()
            
            feature_dict = {
                'device_id': device_id,
                'time_window': window,
                'dns_query_count': len(group),
                'dns_success_count': len(group[group['response_code'] == 'NOERROR']),
                'dns_failure_count': len(failures),
                'dns_nxdomain_count': len(nxdomain),
                'num_unique_domains': len(set(domains)),
                'entropy_of_domains': self.calculate_entropy(domains),
                'mean_response_time': group['response_time'].mean() if 'response_time' in group.columns else 0.0
            }
            
            # Calculate failure ratio
            if feature_dict['dns_query_count'] > 0:
                feature_dict['dns_failure_ratio'] = feature_dict['dns_failure_count'] / feature_dict['dns_query_count']
            else:
                feature_dict['dns_failure_ratio'] = 0.0
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def process(self, log_file_path):
        """
        Complete processing pipeline.
        
        Args:
            log_file_path: Path to DNS log file
            
        Returns:
            DataFrame with extracted features
        """
        df = self.parse_log_file(log_file_path)
        features = self.extract_features(df)
        return features

