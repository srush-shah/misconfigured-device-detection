"""
Westermo Network Traffic Dataset Loader
Handles loading and labeling of Westermo dataset for device misconfiguration detection.

The Westermo dataset includes:
- PCAP files (raw network traffic)
- CSV flow files (pre-processed)
- events.txt (timestamps and labels for network events)

Misconfiguration types in Westermo:
- BAD-MISCONF: Invalid IP address (e.g., 198.134.18.37 instead of 198.18.134.37)
- BAD-MISCONF-DUPLICATION: Duplicate IP address on multiple devices
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import re
from pathlib import Path
import glob
import zipfile


class WestermoLoader:
    """Load and process Westermo network traffic dataset."""
    
    def __init__(self, base_path, dataset_type='reduced', time_window_minutes=5):
        """
        Initialize Westermo loader.
        
        Args:
            base_path: Path to Westermo dataset (e.g., 'data/raw/westermo')
            dataset_type: 'reduced' or 'extended' (default: 'reduced')
            time_window_minutes: Size of time window in minutes (default: 5)
        """
        self.base_path = Path(base_path)
        self.dataset_type = dataset_type
        self.time_window_minutes = time_window_minutes
        
        # Paths
        self.data_path = self.base_path / 'data' / dataset_type
        self.events_file = self.base_path / 'data' / 'events.txt'
        self.pcap_path = self.data_path / 'pcaps'
        self.flows_path = self.data_path / 'flows'
        
        # Event timestamps (loaded from events.txt)
        self.misconfig_events = []
        self.duplication_events = []
        
    def parse_events_file(self):
        """
        Parse events.txt to extract misconfiguration event timestamps.
        
        Returns:
            Dictionary with event types and their time ranges
        """
        if not self.events_file.exists():
            print(f"Warning: Events file not found: {self.events_file}")
            return {}
        
        events = {
            'misconfig': [],  # Invalid IP misconfigurations
            'duplication': [],  # Duplicate IP misconfigurations
            'normal': []  # Normal periods (everything else)
        }
        
        with open(self.events_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('[') and 'INFO' in line:
                    continue
                
                # Parse timestamp and event type
                # Format: [timestamp -- date -- elapsed] [EVENT-TYPE] description
                match = re.match(r'\[([\d.]+)', line)
                if not match:
                    continue
                
                timestamp = float(match.group(1))
                
                # Misconfiguration events
                if 'BAD-MISCONF-START' in line:
                    # Extract device and IP info
                    device_match = re.search(r'(\w+):\s*([\d.]+)\s*-->\s*([\d.]+)', line)
                    if device_match:
                        device = device_match.group(1)
                        old_ip = device_match.group(2)
                        new_ip = device_match.group(3)
                        events['misconfig'].append({
                            'start': timestamp,
                            'device': device,
                            'old_ip': old_ip,
                            'new_ip': new_ip,
                            'type': 'invalid_ip'
                        })
                
                elif 'BAD-MISCONF-END' in line:
                    # Find the most recent misconfig start
                    if events['misconfig']:
                        events['misconfig'][-1]['end'] = timestamp
                
                # Duplication events
                elif 'BAD-MISCONF-DUPLICATION-START' in line:
                    # Extract device and IP info
                    # Format: hub|centre(phy) will be given ip 198.18.134.6 (same as bottom)
                    ip_match = re.search(r'ip\s+([\d.]+)', line)
                    device_match = re.search(r'(\w+)\s+will be given', line)
                    if ip_match:
                        ip = ip_match.group(1)
                        device = device_match.group(1) if device_match else 'unknown'
                        events['duplication'].append({
                            'start': timestamp,
                            'device': device,
                            'ip': ip,
                            'type': 'duplicate_ip'
                        })
                
                elif 'BAD-MISCONF-DUPLICATION-END' in line:
                    # Find the most recent duplication start
                    if events['duplication']:
                        events['duplication'][-1]['end'] = timestamp
        
        print(f"Parsed {len(events['misconfig'])} misconfiguration events")
        print(f"Parsed {len(events['duplication'])} duplication events")
        
        return events
    
    def load_flow_files(self, max_files=None):
        """
        Load CSV flow files from Westermo dataset.
        Handles both direct CSV files and ZIP archives containing CSV files.
        
        Args:
            max_files: Maximum number of files to load (None for all)
            
        Returns:
            DataFrame with flow data
        """
        if not self.flows_path.exists():
            print(f"Warning: Flows directory not found: {self.flows_path}")
            return pd.DataFrame()
        
        import zipfile
        
        # Find CSV files and ZIP files
        csv_files = list(self.flows_path.glob('*.csv'))
        zip_files = list(self.flows_path.glob('*.zip'))
        
        all_flows = []
        
        # Load direct CSV files
        if csv_files:
            if max_files:
                csv_files = csv_files[:max_files]
            
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if not df.empty:
                        all_flows.append(df)
                        print(f"  Loaded {len(df):,} flows from {csv_file.name}")
                except Exception as e:
                    print(f"  Error loading {csv_file.name}: {e}")
        
        # Load CSV files from ZIP archives
        if zip_files:
            if max_files and csv_files:
                # Adjust max_files if we already loaded some
                remaining = max_files - len(csv_files) if max_files else None
            else:
                remaining = max_files
            
            if remaining is None or remaining > 0:
                for zip_file in zip_files[:remaining] if remaining else zip_files:
                    try:
                        with zipfile.ZipFile(zip_file, 'r') as z:
                            # List CSV files in ZIP
                            csv_in_zip = [f for f in z.namelist() if f.endswith('.csv')]
                            
                            for csv_name in csv_in_zip:
                                # Read CSV from ZIP
                                with z.open(csv_name) as f:
                                    df = pd.read_csv(f)
                                    if not df.empty:
                                        all_flows.append(df)
                                        print(f"  Loaded {len(df):,} flows from {csv_name} in {zip_file.name}")
                    except Exception as e:
                        print(f"  Error loading {zip_file.name}: {e}")
        
        if not all_flows:
            print(f"Warning: No flow data found in {self.flows_path}")
            return pd.DataFrame()
        
        # Combine all flows
        combined = pd.concat(all_flows, ignore_index=True)
        print(f"Total flows loaded: {len(combined):,}")
        
        return combined
    
    def label_flows_with_events(self, flows_df, events):
        """
        Label flows based on misconfiguration events.
        
        Args:
            flows_df: DataFrame with flow data
            events: Dictionary with event timestamps from parse_events_file()
            
        Returns:
            DataFrame with 'label' column added
        """
        if flows_df.empty:
            return flows_df
        
        # Initialize labels (0 = Normal)
        flows_df['label'] = 0
        
        # Check if we have timestamp column (Westermo uses 'start')
        timestamp_col = None
        for col in ['start', 'timestamp', 'ts', 'start_time', 'time']:
            if col in flows_df.columns:
                timestamp_col = col
                break
        
        if not timestamp_col:
            print("Warning: No timestamp column found in flows. Cannot label events.")
            return flows_df
        
        # Convert timestamp to Unix timestamp if needed
        # Westermo 'start' column is already Unix timestamp (seconds since epoch)
        if timestamp_col == 'start':
            flows_df['unix_timestamp'] = flows_df[timestamp_col]
        elif flows_df[timestamp_col].dtype == 'object':
            flows_df[timestamp_col] = pd.to_datetime(flows_df[timestamp_col])
            if flows_df[timestamp_col].dtype.name.startswith('datetime'):
                flows_df['unix_timestamp'] = flows_df[timestamp_col].astype('int64') // 10**9
            else:
                flows_df['unix_timestamp'] = flows_df[timestamp_col]
        else:
            flows_df['unix_timestamp'] = flows_df[timestamp_col]
        
        # Label misconfiguration events (invalid IP) -> Label 2 (DHCP misconfig)
        for event in events['misconfig']:
            if 'end' in event:
                mask = (flows_df['unix_timestamp'] >= event['start']) & \
                       (flows_df['unix_timestamp'] <= event['end'])
                flows_df.loc[mask, 'label'] = 2  # DHCP misconfig
        
        # Label duplication events -> Label 2 (DHCP misconfig)
        for event in events['duplication']:
            if 'end' in event:
                mask = (flows_df['unix_timestamp'] >= event['start']) & \
                       (flows_df['unix_timestamp'] <= event['end'])
                flows_df.loc[mask, 'label'] = 2  # DHCP misconfig
        
        # Count labels
        label_counts = flows_df['label'].value_counts().sort_index()
        print(f"\nLabel distribution:")
        for label, count in label_counts.items():
            label_name = {0: 'Normal', 2: 'DHCP Misconfig'}.get(label, f'Label {label}')
            print(f"  {label_name}: {count:,} ({count/len(flows_df)*100:.1f}%)")
        
        return flows_df
    
    def convert_pcap_to_zeek(self, pcap_file, output_dir):
        """
        Convert PCAP file to Zeek logs.
        
        Args:
            pcap_file: Path to PCAP file
            output_dir: Directory to save Zeek logs
            
        Returns:
            Path to generated conn.log file
        """
        import subprocess
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run Zeek to convert PCAP
        cmd = ['zeek', '-r', str(pcap_file), '-C', '-w', str(output_dir / 'conn.log')]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"Converted {pcap_file.name} to Zeek logs")
            return output_dir / 'conn.log'
        except subprocess.CalledProcessError as e:
            print(f"Error converting PCAP: {e}")
            return None
        except FileNotFoundError:
            print("Warning: Zeek not found. Install Zeek to convert PCAP files.")
            return None
    
    def load_all(self, use_flows=True, convert_pcaps=False, max_files=None):
        """
        Load all data from Westermo dataset.
        
        Args:
            use_flows: Use pre-processed CSV flow files (True) or convert PCAPs (False)
            convert_pcaps: Convert PCAP files to Zeek logs (requires Zeek installed)
            max_files: Maximum number of files to load
            
        Returns:
            Dictionary with 'conn', 'dns', 'dhcp' DataFrames and 'events'
        """
        print("="*60)
        print(f"Loading Westermo Dataset ({self.dataset_type})")
        print("="*60)
        
        # Parse events file
        events = self.parse_events_file()
        
        results = {
            'conn': pd.DataFrame(),
            'dns': pd.DataFrame(),
            'dhcp': pd.DataFrame(),
            'events': events
        }
        
        if use_flows:
            # Load pre-processed flow files
            flows_df = self.load_flow_files(max_files=max_files)
            
            if not flows_df.empty:
                # Map Westermo column names to standard format for compatibility
                column_mapping = {
                    'sIPs': 'orig_ip',
                    'rIPs': 'resp_ip',
                    'sAddress': 'orig_mac',
                    'rAddress': 'resp_mac',
                    'sBytesSum': 'orig_bytes',
                    'rBytesSum': 'resp_bytes',
                    'sPackets': 'orig_pkts',
                    'rPackets': 'resp_pkts',
                    'start': 'timestamp',
                    'duration': 'duration',
                    'protocol': 'proto'
                }
                
                # Rename columns that exist
                for old_col, new_col in column_mapping.items():
                    if old_col in flows_df.columns and new_col not in flows_df.columns:
                        flows_df[new_col] = flows_df[old_col]
                
                # Label flows based on events
                flows_df = self.label_flows_with_events(flows_df, events)
                
                # Ensure timestamp is datetime for compatibility
                if 'timestamp' in flows_df.columns:
                    if flows_df['timestamp'].dtype != 'datetime64[ns]':
                        # Convert from Unix timestamp if needed
                        if flows_df['timestamp'].dtype in ['int64', 'float64']:
                            flows_df['timestamp'] = pd.to_datetime(flows_df['timestamp'], unit='s')
                
                results['conn'] = flows_df
        
        elif convert_pcaps:
            # Convert PCAP files to Zeek logs
            if not self.pcap_path.exists():
                print(f"Warning: PCAP directory not found: {self.pcap_path}")
                return results
            
            pcap_files = list(self.pcap_path.glob('*.pcap'))
            if not pcap_files:
                print(f"Warning: No PCAP files found in {self.pcap_path}")
                return results
            
            zeek_output = self.base_path / 'zeek_logs' / self.dataset_type
            zeek_output.mkdir(parents=True, exist_ok=True)
            
            for pcap_file in pcap_files[:max_files] if max_files else pcap_files:
                self.convert_pcap_to_zeek(pcap_file, zeek_output)
            
            # Then load Zeek logs using existing Zeek parser
            from .zeek_parser import ZeekParser
            parser = ZeekParser()
            conn_log = zeek_output / 'conn.log'
            if conn_log.exists():
                results['conn'] = parser.parse_zeek_log(str(conn_log), log_type='conn')
        
        return results

