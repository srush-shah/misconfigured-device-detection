"""
Flow/PCAP Parser
Extracts network flow features from CSV or PCAP files.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class FlowParser:
    """Parse flow/PCAP data and extract per-device, per-window statistics."""
    
    def __init__(self, time_window_minutes=5):
        """
        Initialize Flow parser.
        
        Args:
            time_window_minutes: Size of time window in minutes (default: 5)
        """
        self.time_window_minutes = time_window_minutes
    
    def parse_csv_file(self, csv_file_path):
        """
        Parse flow CSV file.
        
        Expected columns: timestamp, src_ip, dst_ip, bytes_sent, bytes_received, 
                          protocol, packet_count, duration
        
        Args:
            csv_file_path: Path to flow CSV file
            
        Returns:
            DataFrame with flow records
        """
        try:
            df = pd.read_csv(csv_file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error parsing flow CSV: {e}")
            return pd.DataFrame()
    
    def extract_features(self, df):
        """
        Extract per-device, per-window flow features.
        
        Args:
            df: DataFrame with flow records
            
        Returns:
            DataFrame with features per (device_id, time_window)
        """
        if df.empty:
            return pd.DataFrame()
        
        # Create time windows
        df['time_window'] = df['timestamp'].dt.floor(f'{self.time_window_minutes}min')
        
        features = []
        
        # Process both as source and destination
        for device_col in ['src_ip', 'dst_ip']:
            device_field = device_col.replace('_ip', '')
            
            for (device_id, window), group in df.groupby([device_col, 'time_window']):
                # Calculate inter-arrival times
                group_sorted = group.sort_values('timestamp')
                if len(group_sorted) > 1:
                    inter_arrivals = group_sorted['timestamp'].diff().dt.total_seconds()
                    inter_arrivals = inter_arrivals[inter_arrivals > 0]
                    mean_iat = inter_arrivals.mean() if len(inter_arrivals) > 0 else 0.0
                    var_iat = inter_arrivals.var() if len(inter_arrivals) > 0 else 0.0
                else:
                    mean_iat = 0.0
                    var_iat = 0.0
                
                # ARP features (if protocol column exists)
                if 'protocol' in group.columns:
                    arp_requests = len(group[group['protocol'] == 'ARP'])
                    # Simplified: assume ARP if small packets to broadcast
                    arp_requests += len(group[(group['bytes_sent'] < 100) & 
                                             (group['dst_ip'].str.endswith('.255', na=False))])
                else:
                    arp_requests = 0
                
                # ICMP features
                if 'protocol' in group.columns:
                    icmp_unreachable = len(group[group['protocol'] == 'ICMP'])
                else:
                    icmp_unreachable = 0
                
                # Broadcast ratio
                total_packets = len(group)
                broadcast_packets = len(group[group['dst_ip'].str.endswith('.255', na=False)]) if 'dst_ip' in group.columns else 0
                broadcast_ratio = broadcast_packets / total_packets if total_packets > 0 else 0.0
                
                # Gateway features
                unique_gateways = group['dst_ip'].nunique() if 'dst_ip' in group.columns else 0
                
                feature_dict = {
                    'device_id': device_id,
                    'time_window': window,
                    f'bytes_{device_field}': group['bytes_sent'].sum() if 'bytes_sent' in group.columns else 0,
                    f'bytes_received_{device_field}': group['bytes_received'].sum() if 'bytes_received' in group.columns else 0,
                    f'flows_{device_field}': len(group),
                    'mean_inter_arrival_time': mean_iat,
                    'var_inter_arrival_time': var_iat,
                    'arp_request_count': arp_requests,
                    'icmp_unreachable_count': icmp_unreachable,
                    'broadcast_packet_ratio': broadcast_ratio,
                    'num_distinct_gateways': unique_gateways
                }
                
                features.append(feature_dict)
        
        # Aggregate by device_id and time_window
        if features:
            features_df = pd.DataFrame(features)
            # Sum features for same device/window (from src and dst perspectives)
            agg_dict = {
                'bytes_src': 'sum',
                'bytes_received_src': 'sum',
                'flows_src': 'sum',
                'bytes_dst': 'sum',
                'bytes_received_dst': 'sum',
                'flows_dst': 'sum',
                'mean_inter_arrival_time': 'mean',
                'var_inter_arrival_time': 'mean',
                'arp_request_count': 'sum',
                'icmp_unreachable_count': 'sum',
                'broadcast_packet_ratio': 'mean',
                'num_distinct_gateways': 'max'
            }
            
            # Rename columns for aggregation
            features_df = features_df.groupby(['device_id', 'time_window']).agg({
                col: 'sum' if 'bytes' in col or 'flows' in col or 'count' in col else 'mean'
                for col in features_df.columns if col not in ['device_id', 'time_window']
            }).reset_index()
            
            # Rename to final feature names
            features_df = features_df.rename(columns={
                'bytes_src': 'bytes_out',
                'bytes_received_src': 'bytes_in',
                'flows_src': 'flows_out',
                'flows_dst': 'flows_in'
            })
            
            return features_df
        
        return pd.DataFrame()
    
    def process(self, csv_file_path):
        """
        Complete processing pipeline.
        
        Args:
            csv_file_path: Path to flow CSV file
            
        Returns:
            DataFrame with extracted features
        """
        df = self.parse_csv_file(csv_file_path)
        features = self.extract_features(df)
        return features

