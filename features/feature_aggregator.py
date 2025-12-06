"""
Feature Aggregator
Combines features from multiple sources (DHCP, DNS, Flow) into unified feature vectors.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class FeatureAggregator:
    """Aggregate features from DHCP, DNS, and Flow sources."""
    
    def __init__(self):
        """Initialize feature aggregator."""
        pass
    
    def aggregate(self, dhcp_features, dns_features, flow_features):
        """
        Aggregate features from all sources.
        
        Args:
            dhcp_features: DataFrame with DHCP features
            dns_features: DataFrame with DNS features
            flow_features: DataFrame with Flow features
            
        Returns:
            Combined DataFrame with all features per (device_id, time_window)
        """
        # Start with device_id and time_window
        all_devices = set()
        all_windows = set()
        
        for df in [dhcp_features, dns_features, flow_features]:
            if not df.empty:
                all_devices.update(df['device_id'].unique())
                all_windows.update(df['time_window'].unique())
        
        # Create base DataFrame with all combinations
        if not all_devices or not all_windows:
            return pd.DataFrame()
        
        # Merge all features
        result = None
        
        for df in [dhcp_features, dns_features, flow_features]:
            if not df.empty:
                if result is None:
                    result = df.copy()
                else:
                    # Preserve label column if it exists
                    label_col = None
                    if 'label' in result.columns and 'label' in df.columns:
                        # Use label from flow_features (most reliable) or first non-null
                        label_col = result['label'].copy()
                        df_label = df['label'].copy()
                        # Merge labels: prefer non-zero labels (misconfigurations)
                        merged_label = label_col.copy()
                        mask = (merged_label == 0) & (df_label != 0)
                        merged_label[mask] = df_label[mask]
                        label_col = merged_label
                    elif 'label' in df.columns:
                        label_col = df['label'].copy()
                    elif 'label' in result.columns:
                        label_col = result['label'].copy()
                    
                    result = result.merge(
                        df,
                        on=['device_id', 'time_window'],
                        how='outer',
                        suffixes=('', '_dup')
                    )
                    # Remove duplicate columns
                    result = result.loc[:, ~result.columns.str.endswith('_dup')]
                    
                    # Restore label column if it was preserved
                    if label_col is not None:
                        # Align label with result index
                        if len(label_col) == len(result):
                            result['label'] = label_col.values
                        else:
                            # Re-merge label based on device_id and time_window
                            label_df = pd.DataFrame({
                                'device_id': result['device_id'].values,
                                'time_window': result['time_window'].values,
                                'label': label_col.values if len(label_col) == len(result) else 0
                            })
                            result = result.drop(columns=['label'], errors='ignore')
                            result = result.merge(label_df[['device_id', 'time_window', 'label']], 
                                                on=['device_id', 'time_window'], how='left')
                            result['label'] = result['label'].fillna(0).astype(int)
        
        # Fill NaN values with 0 (but preserve label column)
        if result is not None and not result.empty:
            label_col = result['label'].copy() if 'label' in result.columns else None
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            if 'label' in numeric_cols:
                numeric_cols = numeric_cols.drop('label')
            result[numeric_cols] = result[numeric_cols].fillna(0)
            
            # Restore label column
            if label_col is not None:
                result['label'] = label_col
        
        # Sort by device_id and time_window
        result = result.sort_values(['device_id', 'time_window']).reset_index(drop=True)
        
        return result
    
    def get_feature_groups(self, df):
        """
        Split features into groups for multi-view fusion.
        
        Args:
            df: Combined feature DataFrame
            
        Returns:
            dict with keys: 'dhcp', 'dns', 'flow', each containing feature names
        """
        # Exclude metadata columns
        exclude_cols = ['device_id', 'time_window', 'label']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # DHCP features (from DHCP logs)
        dhcp_features = [col for col in feature_cols if col.startswith('dhcp_') or col.startswith('lease_') or col.startswith('failed_')]
        
        # DNS features (from DNS logs)
        dns_features = [col for col in feature_cols if col.startswith('dns_') or col.startswith('num_unique_') or col.startswith('entropy_')]
        
        # Flow features (from flow/conn logs or Westermo flows)
        # Include Westermo-style column names (orig_bytes_sum, resp_bytes_sum, etc.)
        flow_features = [col for col in feature_cols if (
            col.startswith('bytes_') or col.startswith('flows_') or 
            col.startswith('arp_') or col.startswith('icmp_') or col.startswith('broadcast_') or
            col.startswith('mean_') or col.startswith('var_') or col.startswith('num_distinct_') or
            col.startswith('orig_') or col.startswith('resp_') or col.startswith('duration_') or
            col.startswith('sBytes') or col.startswith('rBytes') or col.startswith('sPackets') or
            col.startswith('rPackets') or col.startswith('sAddress') or col.startswith('rAddress')
        )]
        
        # If no flow features found with prefixes, use all remaining features as flow features
        if not flow_features and not dhcp_features and not dns_features:
            flow_features = feature_cols
        
        return {
            'dhcp': dhcp_features,
            'dns': dns_features,
            'flow': flow_features
        }

