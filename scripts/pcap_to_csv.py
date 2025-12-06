#!/usr/bin/env python
"""
Convert PCAP file to Flow CSV
Converts network capture (PCAP) to CSV format for misconfiguration detection.

Usage:
    python pcap_to_csv.py realtime_capture.pcap realtime_flows.csv
    python pcap_to_csv.py realtime_capture.pcap realtime_flows.csv --method zeek
    python pcap_to_csv.py realtime_capture.pcap realtime_flows.csv --method tshark
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
import pandas as pd
from datetime import datetime

def convert_with_tshark(pcap_file, output_csv, debug=False):
    """Convert PCAP to CSV using tshark."""
    print(f"📊 Converting {pcap_file} to CSV using tshark...")
    
    # Check if tshark is available
    try:
        result = subprocess.run(['tshark', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("❌ Error: tshark not found. Install with: brew install wireshark (macOS) or apt-get install tshark (Linux)")
        return False
    
    # First, check if PCAP has any IP traffic
    print("  Checking PCAP file for IP traffic...")
    check_cmd = ['tshark', '-r', str(pcap_file), '-Y', 'ip', '-c', '1']
    check_result = subprocess.run(check_cmd, capture_output=True, text=True)
    
    if check_result.returncode != 0 or not check_result.stdout.strip():
        print("⚠️  Warning: No IP traffic found in PCAP file")
        print("   The PCAP might be empty or contain only non-IP traffic (e.g., ARP only)")
        print("   Try capturing with: tshark -i en0 -f 'ip' -a duration:300 -w capture.pcap")
        return False
    
    # Tshark fields that match our expected format
    fields = [
        'frame.number',
        'frame.time',
        'ip.src',
        'ip.dst',
        'ip.proto',
        'tcp.srcport',
        'tcp.dstport',
        'udp.srcport',
        'udp.dstport',
        'tcp.len',
        'udp.length',
        'frame.len',
        'tcp.flags',
        'tcp.analysis.flags'
    ]
    
    # Run tshark
    cmd = [
        'tshark',
        '-r', str(pcap_file),
        '-Y', 'ip',  # Only IP packets
        '-T', 'fields',
        '-E', 'header=y',
        '-E', 'separator=,',
        '-E', 'quote=d',
        '-E', 'occurrence=f'
    ] + [f'-e {field}' for field in fields]
    
    try:
        print("  Running tshark...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if not result.stdout.strip():
            print("⚠️  Warning: tshark produced no output")
            print(f"   stderr: {result.stderr[:200]}")
            return False
        
        # Parse the output
        from io import StringIO
        df = pd.read_csv(StringIO(result.stdout))
        
        # Strip whitespace from column names (tshark sometimes adds spaces)
        df.columns = df.columns.str.strip()
        
        if df.empty:
            print("⚠️  Warning: No data extracted from PCAP file")
            print(f"   tshark output: {result.stdout[:200]}")
            return False
        
        print(f"  Extracted {len(df)} packets from PCAP")
        
        if debug:
            print(f"  DEBUG - All columns: {list(df.columns)}")
            print(f"  DEBUG - First few rows:\n{df.head()}")
            print(f"  DEBUG - Data types:\n{df.dtypes}")
        
        # Convert to flow format
        print("  Processing flows...")
        
        # Map tshark columns to our format
        # Use source IP as device_id - try multiple variations
        device_id_col = None
        for col in ['ip.src', 'Source', 'src_ip', 'source_ip', 'ip_src']:
            if col in df.columns:
                device_id_col = col
                break
        
        if device_id_col:
            # Filter out empty/NaN values
            df = df[df[device_id_col].notna() & (df[device_id_col].astype(str).str.strip() != '')]
            if len(df) == 0:
                print(f"❌ Error: All rows have empty source IP in column '{device_id_col}'")
                return False
            df['device_id'] = df[device_id_col].astype(str)
            print(f"  Using '{device_id_col}' as device_id ({len(df)} valid rows)")
        else:
            print(f"❌ Error: No source IP column found in PCAP")
            print(f"   Available columns: {list(df.columns)}")
            if debug:
                print(f"   First few rows:\n{df.head()}")
            return False
        
        # Convert frame.time to timestamp
        if 'frame.time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['frame.time'], errors='coerce')
            df['timestamp'] = df['timestamp'].fillna(pd.Timestamp.now())
        elif 'Time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['Time'], errors='coerce')
            df['timestamp'] = df['timestamp'].fillna(pd.Timestamp.now())
        else:
            df['timestamp'] = datetime.now()
        
        # Calculate bytes (use frame.len as total bytes)
        if 'frame.len' in df.columns:
            df['orig_bytes'] = pd.to_numeric(df['frame.len'], errors='coerce').fillna(0).astype(int)
        elif 'Length' in df.columns:
            df['orig_bytes'] = pd.to_numeric(df['Length'], errors='coerce').fillna(0).astype(int)
        else:
            df['orig_bytes'] = 0
        
        # For tshark, we only have one direction per packet
        df['resp_bytes'] = 0
        
        # Packets (each frame is a packet)
        df['orig_pkts'] = 1
        df['resp_pkts'] = 0
        
        # Duration (we'll set to 0 for individual packets, aggregation will handle it)
        df['duration'] = 0.0
        
        # Protocol
        if 'ip.proto' in df.columns:
            df['proto'] = pd.to_numeric(df['ip.proto'], errors='coerce').fillna(0).astype(int)
        elif 'Protocol' in df.columns:
            # Protocol might be text (TCP, UDP, etc.)
            proto_map = {'TCP': 6, 'UDP': 17, 'ICMP': 1}
            df['proto'] = df['Protocol'].map(proto_map).fillna(0).astype(int)
        else:
            df['proto'] = 0
        
        # Select and rename columns for output
        output_cols = ['device_id', 'timestamp', 'orig_bytes', 'resp_bytes', 
                      'orig_pkts', 'resp_pkts', 'duration', 'proto']
        
        # Add source/dest IPs for reference
        if 'ip.src' in df.columns:
            output_cols.append('ip.src')
        if 'ip.dst' in df.columns:
            output_cols.append('ip.dst')
        
        output_df = df[output_cols].copy()
        
        # Save to CSV
        output_df.to_csv(output_csv, index=False)
        print(f"✅ Converted {len(output_df)} flows to {output_csv}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running tshark: {e}")
        print(f"   stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return False


def convert_with_zeek(pcap_file, output_csv, debug=False):
    """Convert PCAP to CSV using Zeek (recommended method)."""
    print(f"📊 Converting {pcap_file} to CSV using Zeek...")
    
    # Check if zeek is available
    try:
        result = subprocess.run(['zeek', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("❌ Error: Zeek not found. Install with: brew install zeek (macOS) or apt-get install zeek (Linux)")
        return False
    
    pcap_path = Path(pcap_file).absolute()
    output_dir = Path(output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to output directory for Zeek
    original_dir = os.getcwd()
    os.chdir(output_dir)
    
    try:
        # Run Zeek on PCAP
        print("  Running Zeek...")
        cmd = ['zeek', '-r', str(pcap_path), '-C']  # -C for no checksums
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check for conn.log
        conn_log = Path('conn.log')
        if not conn_log.exists():
            print("⚠️  Warning: conn.log not generated. Checking for other logs...")
            logs = list(Path('.').glob('*.log'))
            if logs:
                print(f"   Found logs: {[str(l) for l in logs]}")
            return False
        
        # Read conn.log (Zeek format is tab-separated)
        print("  Reading conn.log...")
        
        # Try reading with different approaches
        try:
            # First, read header to see what columns exist
            with open(conn_log, 'r') as f:
                header_line = None
                for line in f:
                    if line.startswith('#fields'):
                        header_line = line
                        break
                if header_line:
                    # Parse header: #fields	ts	uid	id.orig_h	...
                    headers = header_line.strip().split('\t')[1:]  # Skip '#fields'
                    print(f"  Found Zeek columns: {headers[:10]}...")  # Show first 10
        except Exception as e:
            print(f"  Warning: Could not read header: {e}")
        
        # Read the actual data (skip comment lines)
        df = pd.read_csv(conn_log, sep='\t', comment='#', low_memory=False)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        if df.empty:
            print("⚠️  Warning: No data in conn.log")
            print(f"   File size: {conn_log.stat().st_size} bytes")
            return False
        
        print(f"  Loaded {len(df)} rows from conn.log")
        print(f"  Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        
        if debug:
            print(f"  DEBUG - All columns: {list(df.columns)}")
            print(f"  DEBUG - First few rows:\n{df.head()}")
            print(f"  DEBUG - Data types:\n{df.dtypes}")
        
        # Map Zeek columns to our format
        # Zeek conn.log columns: ts, uid, id.orig_h, id.orig_p, id.resp_h, id.resp_p, proto, service, duration, orig_bytes, resp_bytes, ...
        
        output_df = pd.DataFrame()
        
        # Device ID (use originator IP) - try multiple possible column names
        device_id_col = None
        for col in ['id.orig_h', 'orig_ip', 'src_ip', 'source', 'id_orig_h']:
            if col in df.columns:
                device_id_col = col
                break
        
        if device_id_col:
            output_df['device_id'] = df[device_id_col].astype(str)
            print(f"  Using '{device_id_col}' as device_id")
        else:
            print(f"❌ Error: No source IP column found in Zeek output")
            print(f"   Available columns: {list(df.columns)}")
            print(f"   First few rows:\n{df.head()}")
            return False
        
        # Timestamp - try multiple column names
        ts_col = None
        for col in ['ts', 'timestamp', 'time', 'start']:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col:
            if ts_col == 'ts':
                # Zeek uses Unix timestamp
                output_df['timestamp'] = pd.to_datetime(df[ts_col], unit='s', errors='coerce')
            else:
                output_df['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
            # Fill any NaT with current time
            output_df['timestamp'] = output_df['timestamp'].fillna(pd.Timestamp.now())
            print(f"  Using '{ts_col}' as timestamp")
        else:
            output_df['timestamp'] = datetime.now()
            print("  Warning: No timestamp column found, using current time")
        
        # Bytes - try multiple column names
        orig_bytes_col = None
        for col in ['orig_bytes', 'orig_bytes_sum', 'bytes_orig', 'sBytesSum']:
            if col in df.columns:
                orig_bytes_col = col
                break
        
        if orig_bytes_col:
            output_df['orig_bytes'] = pd.to_numeric(df[orig_bytes_col], errors='coerce').fillna(0).astype(int)
        else:
            output_df['orig_bytes'] = 0
        
        resp_bytes_col = None
        for col in ['resp_bytes', 'resp_bytes_sum', 'bytes_resp', 'rBytesSum']:
            if col in df.columns:
                resp_bytes_col = col
                break
        
        if resp_bytes_col:
            output_df['resp_bytes'] = pd.to_numeric(df[resp_bytes_col], errors='coerce').fillna(0).astype(int)
        else:
            output_df['resp_bytes'] = 0
        
        # Packets - try multiple column names
        orig_pkts_col = None
        for col in ['orig_pkts', 'orig_packets', 'pkts_orig', 'sPackets']:
            if col in df.columns:
                orig_pkts_col = col
                break
        
        if orig_pkts_col:
            output_df['orig_pkts'] = pd.to_numeric(df[orig_pkts_col], errors='coerce').fillna(0).astype(int)
        else:
            # If no packet count, assume 1 packet per flow
            output_df['orig_pkts'] = 1
        
        resp_pkts_col = None
        for col in ['resp_pkts', 'resp_packets', 'pkts_resp', 'rPackets']:
            if col in df.columns:
                resp_pkts_col = col
                break
        
        if resp_pkts_col:
            output_df['resp_pkts'] = pd.to_numeric(df[resp_pkts_col], errors='coerce').fillna(0).astype(int)
        else:
            output_df['resp_pkts'] = 0
        
        # Duration - try multiple column names
        duration_col = None
        for col in ['duration', 'dur', 'flow_duration']:
            if col in df.columns:
                duration_col = col
                break
        
        if duration_col:
            output_df['duration'] = pd.to_numeric(df[duration_col], errors='coerce').fillna(0.0).astype(float)
        else:
            output_df['duration'] = 0.0
        
        # Protocol - try multiple column names
        proto_col = None
        for col in ['proto', 'protocol', 'ip.proto']:
            if col in df.columns:
                proto_col = col
                break
        
        if proto_col:
            output_df['proto'] = df[proto_col]
        else:
            output_df['proto'] = 0
        
        # Save to CSV
        output_path = Path(output_csv).name if Path(output_csv).is_absolute() else output_csv
        output_df.to_csv(output_path, index=False)
        print(f"✅ Converted {len(output_df)} flows to {output_path}")
        
        # Clean up Zeek logs (optional)
        print("  Note: Zeek generated additional log files in current directory")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Zeek: {e}")
        print(f"   stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Convert PCAP file to Flow CSV for misconfiguration detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Zeek (recommended)
  python pcap_to_csv.py realtime_capture.pcap realtime_flows.csv --method zeek
  
  # Using tshark
  python pcap_to_csv.py realtime_capture.pcap realtime_flows.csv --method tshark
  
  # Auto-detect (tries Zeek first, falls back to tshark)
  python pcap_to_csv.py realtime_capture.pcap realtime_flows.csv
        """
    )
    
    parser.add_argument('pcap_file', help='Input PCAP file path')
    parser.add_argument('output_csv', help='Output CSV file path')
    parser.add_argument(
        '--method',
        choices=['zeek', 'tshark', 'auto'],
        default='auto',
        help='Conversion method (default: auto - tries Zeek first, then tshark)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug information (column names, sample data)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.pcap_file):
        print(f"❌ Error: PCAP file not found: {args.pcap_file}")
        return 1
    
    # Convert based on method
    success = False
    
    if args.method == 'zeek':
        success = convert_with_zeek(args.pcap_file, args.output_csv, debug=args.debug)
    elif args.method == 'tshark':
        success = convert_with_tshark(args.pcap_file, args.output_csv, debug=args.debug)
    else:  # auto
        print("🔄 Auto-detecting best method...")
        # Try Zeek first (better for flow analysis)
        if convert_with_zeek(args.pcap_file, args.output_csv, debug=args.debug):
            success = True
        else:
            print("\n⚠️  Zeek conversion failed, trying tshark...")
            success = convert_with_tshark(args.pcap_file, args.output_csv, debug=args.debug)
    
    if success:
        print(f"\n✅ Success! Flow CSV saved to: {args.output_csv}")
        print(f"\n📝 Next step: Process the CSV with:")
        print(f"   python scripts/process_realtime_data.py --flow-csv {args.output_csv} --output features.csv")
        return 0
    else:
        print("\n❌ Conversion failed. Check errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())

