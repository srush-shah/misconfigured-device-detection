#!/usr/bin/env python
"""
Process Real-Time Network Data
Extracts features from network logs/CSV files for misconfiguration detection.

Usage:
    python process_realtime_data.py --flow-csv flows.csv --output features.csv
    python process_realtime_data.py --flow-csv flows.csv --dhcp-log dhcp.log --dns-log dns.log --output features.csv
    python process_realtime_data.py --westermo-path /path/to/westermo/data --output features.csv
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.main_pipeline import MainPipeline


def main():
    parser = argparse.ArgumentParser(
        description='Process real-time network data and extract features for misconfiguration detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process flow CSV only
  python process_realtime_data.py --flow-csv flows.csv --output features.csv
  
  # Process with all log files
  python process_realtime_data.py --flow-csv flows.csv --dhcp-log dhcp.log --dns-log dns.log --output features.csv
  
  # Process Westermo dataset
  python process_realtime_data.py --westermo-path data/raw/westermo --output features.csv
  
  # Process and save to default location
  python process_realtime_data.py --flow-csv flows.csv
        """
    )
    
    # Input options (mutually exclusive groups)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--flow-csv',
        type=str,
        help='Path to flow CSV file'
    )
    input_group.add_argument(
        '--westermo-path',
        type=str,
        help='Path to Westermo dataset directory'
    )
    
    # Optional log files
    parser.add_argument(
        '--dhcp-log',
        type=str,
        help='Path to DHCP log file (optional)'
    )
    parser.add_argument(
        '--dns-log',
        type=str,
        help='Path to DNS log file (optional)'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        default='realtime_features.csv',
        help='Output CSV file path (default: realtime_features.csv)'
    )
    
    # Pipeline options
    parser.add_argument(
        '--time-window',
        type=int,
        default=5,
        help='Time window in minutes (default: 5)'
    )
    
    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of files to process (default: all)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress information'
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if args.flow_csv and not os.path.exists(args.flow_csv):
        print(f"❌ Error: Flow CSV file not found: {args.flow_csv}")
        sys.exit(1)
    
    if args.dhcp_log and not os.path.exists(args.dhcp_log):
        print(f"❌ Error: DHCP log file not found: {args.dhcp_log}")
        sys.exit(1)
    
    if args.dns_log and not os.path.exists(args.dns_log):
        print(f"❌ Error: DNS log file not found: {args.dns_log}")
        sys.exit(1)
    
    if args.westermo_path and not os.path.exists(args.westermo_path):
        print(f"❌ Error: Westermo path not found: {args.westermo_path}")
        sys.exit(1)
    
    # Initialize pipeline
    if args.verbose:
        print("🔧 Initializing pipeline...")
    
    pipeline = MainPipeline(config={
        'time_window_minutes': args.time_window,
        'sequence_length': 12,  # Default sequence length for time-series models
        'device': 'cpu',
        'use_improved_models': True
    })
    
    # Process data
    try:
        if args.verbose:
            print("📊 Processing network data...")
        
        if args.westermo_path:
            # Process Westermo dataset
            if args.verbose:
                print(f"  Loading Westermo dataset from: {args.westermo_path}")
            
            features = pipeline.ingest_data(
                westermo_base_path=args.westermo_path,
                dataset_type='westermo',
                max_files=args.max_files
            )
        else:
            # Process individual files
            if args.verbose:
                print(f"  Loading flow data from: {args.flow_csv}")
                if args.dhcp_log:
                    print(f"  Loading DHCP log from: {args.dhcp_log}")
                if args.dns_log:
                    print(f"  Loading DNS log from: {args.dns_log}")
            
            # Process individual CSV file(s)
            # The pipeline's ingest_data expects Westermo format, so we'll
            # use a simpler approach for individual CSV files
            
            if args.flow_csv:
                # Load flow CSV
                df = pd.read_csv(args.flow_csv)
                
                if args.verbose:
                    print(f"  Loaded {len(df)} rows from CSV")
                    print(f"  Columns: {list(df.columns)[:10]}...")
                
                # Ensure required columns exist
                # The CSV from pcap_to_csv.py has: device_id, timestamp, orig_bytes, etc.
                # We need to convert 'timestamp' to 'time_window' for aggregation
                
                if 'timestamp' in df.columns and 'time_window' not in df.columns:
                    # Convert timestamp to datetime if needed
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    # Create time_window by flooring to time window intervals
                    df['time_window'] = df['timestamp'].dt.floor(f"{args.time_window}min")
                    if args.verbose:
                        print(f"  Created time_window from timestamp (window: {args.time_window} min)")
                
                if 'device_id' not in df.columns:
                    print(f"⚠️  Warning: Missing 'device_id' column. Adding defaults...")
                    df['device_id'] = df.index.map(lambda x: f'device_{x}')
                
                if 'time_window' not in df.columns:
                    print(f"⚠️  Warning: Missing 'time_window' column. Using single window...")
                    from datetime import datetime
                    df['time_window'] = pd.Timestamp.now().floor(f"{args.time_window}min")
                
                # Aggregate flow data by device_id and time_window
                # The CSV from pcap_to_csv.py has: device_id, timestamp, orig_bytes, resp_bytes, orig_pkts, resp_pkts, duration, proto
                
                if args.verbose:
                    print(f"  Aggregating flows by device and time window...")
                
                # Group by device_id and time_window, then aggregate
                agg_dict = {}
                
                # Bytes
                if 'orig_bytes' in df.columns:
                    agg_dict['orig_bytes'] = ['sum', 'mean', 'count']
                if 'resp_bytes' in df.columns:
                    agg_dict['resp_bytes'] = ['sum', 'mean']
                
                # Packets
                if 'orig_pkts' in df.columns:
                    agg_dict['orig_pkts'] = 'sum'
                if 'resp_pkts' in df.columns:
                    agg_dict['resp_pkts'] = 'sum'
                
                # Duration
                if 'duration' in df.columns:
                    agg_dict['duration'] = 'mean'
                
                # Protocol (most common)
                if 'proto' in df.columns:
                    agg_dict['proto'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else 0
                
                # Count unique devices before aggregation
                unique_devices_before = df['device_id'].nunique()
                total_rows_before = len(df)
                
                if args.verbose:
                    print(f"   Before aggregation: {total_rows_before} flows from {unique_devices_before} unique devices")
                
                if not agg_dict:
                    print("⚠️  Warning: No aggregatable columns found. Using basic aggregation...")
                    # Fallback: just count flows per device/time_window
                    features = df.groupby(['device_id', 'time_window']).size().reset_index(name='flow_count')
                else:
                    # Aggregate - this combines multiple flows from same device in same time window
                    # IMPORTANT: Use dropna=False to preserve all device/time_window combinations
                    features = df.groupby(['device_id', 'time_window'], dropna=False).agg(agg_dict).reset_index()
                    
                    # Flatten column names (handle MultiIndex from aggregation)
                    new_columns = ['device_id', 'time_window']
                    for col in features.columns[2:]:
                        if isinstance(col, tuple):
                            if col[1]:
                                new_columns.append(f"{col[0]}_{col[1]}")
                            else:
                                new_columns.append(col[0])
                        else:
                            new_columns.append(col)
                    features.columns = new_columns
                    
                    # Fill NaN values with 0 (devices with no flows in a window will have NaN)
                    # This ensures all devices are preserved
                    numeric_cols = features.select_dtypes(include=[np.number]).columns
                    features[numeric_cols] = features[numeric_cols].fillna(0)
                    
                    # Rename to match expected feature names
                    # Map orig_bytes_sum -> flow_bytes_out_sum, etc.
                    rename_map = {
                        'orig_bytes_sum': 'flow_bytes_out_sum',
                        'orig_bytes_mean': 'flow_bytes_out_mean',
                        'orig_bytes_count': 'flow_count',
                        'resp_bytes_sum': 'flow_bytes_in_sum',
                        'resp_bytes_mean': 'flow_bytes_in_mean',
                        'orig_pkts_sum': 'flow_packets_out_sum',
                        'resp_pkts_sum': 'flow_packets_in_sum',
                        'duration_mean': 'flow_duration_mean'
                    }
                    features = features.rename(columns=rename_map)
                
                # Count unique devices after aggregation
                unique_devices_after = features['device_id'].nunique()
                total_rows_after = len(features)
                
                if args.verbose:
                    print(f"   After aggregation: {total_rows_after} device/time_window combinations from {unique_devices_after} unique devices")
                    if unique_devices_before != unique_devices_after:
                        print(f"   ⚠️  Warning: Device count changed from {unique_devices_before} to {unique_devices_after}")
                        # Find missing devices
                        devices_before = set(df['device_id'].unique())
                        devices_after = set(features['device_id'].unique())
                        missing_devices = devices_before - devices_after
                        if missing_devices:
                            print(f"   Missing devices: {list(missing_devices)[:10]}...")
                    else:
                        print(f"   ✓ All {unique_devices_after} devices preserved")
                
                if args.verbose:
                    print(f"   ✓ Flow data processed: {len(features)} aggregated features")
                    print(f"   Features: {list(features.columns)[:10]}...")
            
            # Note: DHCP and DNS log processing would require proper parsers
            # For now, they should be included in the flow CSV if needed
            if args.dhcp_log or args.dns_log:
                print("⚠️  Note: Individual DHCP/DNS log parsing not yet implemented.")
                print("   Include these features in your flow CSV file instead.")
        
        if features is None or len(features) == 0:
            print("❌ Error: No features extracted. Check your input files.")
            sys.exit(1)
        
        if args.verbose:
            print(f"✅ Extracted features for {len(features)} device(s)/time window(s)")
            print(f"   Features: {list(features.columns)[:10]}...")  # Show first 10 columns
        
        # Save to CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(output_path, index=False)
        
        print(f"✅ Features saved to: {output_path}")
        print(f"   Rows: {len(features)}, Columns: {len(features.columns)}")
        
        # Show summary
        if args.verbose:
            print("\n📋 Feature Summary:")
            print(f"   Device IDs: {features['device_id'].nunique() if 'device_id' in features.columns else 'N/A'}")
            if 'label' in features.columns:
                print(f"   Labels: {features['label'].value_counts().to_dict()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    sys.exit(main())

