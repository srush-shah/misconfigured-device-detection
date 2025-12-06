#!/usr/bin/env python3
"""
Generate batch synthetic dataset with 80 devices (60% normal, 40% misconfigured).
Includes label column for easy identification.
Uses only standard library (no pandas required).
"""

import csv
import random
from datetime import datetime, timedelta

random.seed(456)  # Different seed for variety

# Generate synthetic data - 60% normal, 40% misconfigured
n_normal = 48  # 60% of 80
n_misconfig = 32  # 40% of 80
base_time = datetime(2025, 11, 22, 10, 0, 0)

devices = []

# Normal devices (60%)
for i in range(n_normal):
    device_id = f'10.0.2.{100 + i}'
    time_window = base_time + timedelta(minutes=random.randint(0, 120))
    
    # Normal traffic patterns - healthy network behavior
    flow_count = random.randint(20, 250)
    bytes_out = random.randint(15000, 600000)
    bytes_in = random.randint(8000, 400000)
    packets_out = random.randint(80, 2500)
    packets_in = random.randint(40, 1800)
    duration = round(random.uniform(1.5, 12.0), 2)
    
    devices.append({
        'device_id': device_id,
        'time_window': time_window.strftime('%Y-%m-%d %H:%M:%S'),
        'flow_bytes_out_sum': bytes_out,
        'flow_bytes_out_mean': round(bytes_out / max(flow_count, 1), 2),
        'flow_count': flow_count,
        'flow_bytes_in_sum': bytes_in,
        'flow_bytes_in_mean': round(bytes_in / max(flow_count, 1), 2),
        'flow_packets_out_sum': packets_out,
        'flow_packets_in_sum': packets_in,
        'flow_duration_mean': duration,
        'proto_<lambda>': random.choice([6, 17, 1]),
        'dhcp_discover_count': random.randint(0, 2),  # Low, normal
        'dhcp_request_count': random.randint(0, 2),
        'dhcp_ack_count': random.randint(1, 3),  # Good ACK rate
        'dns_query_count': random.randint(10, 60),
        'dns_success_count': random.randint(8, 58),  # High success rate
        'dns_failure_count': random.randint(0, 3),  # Low failures
        'label': 0  # Normal
    })

# DHCP Misconfigured devices (20% of misconfig)
for i in range(int(n_misconfig * 0.20)):
    device_id = f'10.0.2.{200 + i}'
    time_window = base_time + timedelta(minutes=random.randint(0, 120))
    
    # DHCP misconfig: high discover, low ACK - device can't get IP
    flow_count = random.randint(15, 120)
    bytes_out = random.randint(3000, 150000)
    bytes_in = random.randint(1000, 80000)
    packets_out = random.randint(25, 800)
    packets_in = random.randint(10, 400)
    duration = round(random.uniform(0.3, 4.0), 2)
    
    discover = random.randint(20, 60)  # High discover attempts
    ack = random.randint(0, 3)  # Very low ACK
    
    devices.append({
        'device_id': device_id,
        'time_window': time_window.strftime('%Y-%m-%d %H:%M:%S'),
        'flow_bytes_out_sum': bytes_out,
        'flow_bytes_out_mean': round(bytes_out / max(flow_count, 1), 2),
        'flow_count': flow_count,
        'flow_bytes_in_sum': bytes_in,
        'flow_bytes_in_mean': round(bytes_in / max(flow_count, 1), 2),
        'flow_packets_out_sum': packets_out,
        'flow_packets_in_sum': packets_in,
        'flow_duration_mean': duration,
        'proto_<lambda>': random.choice([6, 17]),
        'dhcp_discover_count': discover,  # High
        'dhcp_request_count': random.randint(15, 45),
        'dhcp_ack_count': ack,  # Very low
        'dns_query_count': random.randint(0, 15),
        'dns_success_count': random.randint(0, 12),
        'dns_failure_count': random.randint(0, 5),
        'label': 2  # DHCP Misconfig
    })

# DNS Misconfigured devices (25% of misconfig)
for i in range(int(n_misconfig * 0.25)):
    device_id = f'10.0.2.{250 + i}'
    time_window = base_time + timedelta(minutes=random.randint(0, 120))
    
    # DNS misconfig: high failure rate - DNS server issues
    flow_count = random.randint(25, 200)
    bytes_out = random.randint(10000, 350000)
    bytes_in = random.randint(5000, 180000)
    packets_out = random.randint(50, 1800)
    packets_in = random.randint(25, 1000)
    duration = round(random.uniform(2.5, 15.0), 2)
    
    queries = random.randint(30, 120)
    failures = random.randint(int(queries * 0.65), queries)  # Very high failure rate
    
    devices.append({
        'device_id': device_id,
        'time_window': time_window.strftime('%Y-%m-%d %H:%M:%S'),
        'flow_bytes_out_sum': bytes_out,
        'flow_bytes_out_mean': round(bytes_out / max(flow_count, 1), 2),
        'flow_count': flow_count,
        'flow_bytes_in_sum': bytes_in,
        'flow_bytes_in_mean': round(bytes_in / max(flow_count, 1), 2),
        'flow_packets_out_sum': packets_out,
        'flow_packets_in_sum': packets_in,
        'flow_duration_mean': duration,
        'proto_<lambda>': random.choice([6, 17]),
        'dhcp_discover_count': random.randint(0, 4),
        'dhcp_request_count': random.randint(0, 4),
        'dhcp_ack_count': random.randint(0, 4),
        'dns_query_count': queries,
        'dns_success_count': queries - failures,
        'dns_failure_count': failures,  # Very high
        'label': 1  # DNS Misconfig
    })

# Gateway Misconfigured devices (25% of misconfig)
for i in range(int(n_misconfig * 0.25)):
    device_id = f'10.0.2.{300 + i}'
    time_window = base_time + timedelta(minutes=random.randint(0, 120))
    
    # Gateway misconfig: unusual traffic patterns - multiple gateways or wrong gateway
    flow_count = random.randint(80, 400)  # Very high flow count
    bytes_out = random.randint(300000, 3000000)  # Very high traffic
    bytes_in = random.randint(150000, 1500000)
    packets_out = random.randint(2000, 8000)  # Very high packet count
    packets_in = random.randint(1000, 5000)
    duration = round(random.uniform(0.05, 2.5), 2)  # Short durations
    
    devices.append({
        'device_id': device_id,
        'time_window': time_window.strftime('%Y-%m-%d %H:%M:%S'),
        'flow_bytes_out_sum': bytes_out,
        'flow_bytes_out_mean': round(bytes_out / max(flow_count, 1), 2),
        'flow_count': flow_count,
        'flow_bytes_in_sum': bytes_in,
        'flow_bytes_in_mean': round(bytes_in / max(flow_count, 1), 2),
        'flow_packets_out_sum': packets_out,
        'flow_packets_in_sum': packets_in,
        'flow_duration_mean': duration,
        'proto_<lambda>': random.choice([6, 17, 1]),
        'dhcp_discover_count': random.randint(0, 8),
        'dhcp_request_count': random.randint(0, 8),
        'dhcp_ack_count': random.randint(0, 6),
        'dns_query_count': random.randint(15, 80),
        'dns_success_count': random.randint(12, 75),
        'dns_failure_count': random.randint(0, 8),
        'label': 3  # Gateway Misconfig
    })

# ARP Storm devices (30% of misconfig)
for i in range(int(n_misconfig * 0.30)):
    device_id = f'10.0.2.{350 + i}'
    time_window = base_time + timedelta(minutes=random.randint(0, 120))
    
    # ARP Storm: extremely high packet count, low bytes - network flooding
    flow_count = random.randint(150, 600)
    bytes_out = random.randint(2000, 60000)  # Low bytes per packet
    bytes_in = random.randint(1000, 30000)
    packets_out = random.randint(8000, 25000)  # Extremely high packet count
    packets_in = random.randint(3000, 12000)
    duration = round(random.uniform(0.005, 0.3), 2)  # Very short durations
    
    devices.append({
        'device_id': device_id,
        'time_window': time_window.strftime('%Y-%m-%d %H:%M:%S'),
        'flow_bytes_out_sum': bytes_out,
        'flow_bytes_out_mean': round(bytes_out / max(flow_count, 1), 2),
        'flow_count': flow_count,
        'flow_bytes_in_sum': bytes_in,
        'flow_bytes_in_mean': round(bytes_in / max(flow_count, 1), 2),
        'flow_packets_out_sum': packets_out,
        'flow_packets_in_sum': packets_in,
        'flow_duration_mean': duration,
        'proto_<lambda>': 17,  # Mostly UDP for ARP
        'dhcp_discover_count': random.randint(0, 3),
        'dhcp_request_count': random.randint(0, 3),
        'dhcp_ack_count': random.randint(0, 2),
        'dns_query_count': random.randint(0, 8),
        'dns_success_count': random.randint(0, 6),
        'dns_failure_count': random.randint(0, 4),
        'label': 4  # ARP Storm
    })

# Shuffle devices to mix normal and misconfigured
random.shuffle(devices)

# Write to CSV
output_file = 'synthetic_batch_test_data.csv'
fieldnames = [
    'device_id', 'time_window', 'flow_bytes_out_sum', 'flow_bytes_out_mean',
    'flow_count', 'flow_bytes_in_sum', 'flow_bytes_in_mean',
    'flow_packets_out_sum', 'flow_packets_in_sum', 'flow_duration_mean',
    'proto_<lambda>', 'dhcp_discover_count', 'dhcp_request_count',
    'dhcp_ack_count', 'dns_query_count', 'dns_success_count', 'dns_failure_count',
    'label'
]

with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(devices)

# Count by label
label_counts = {}
for device in devices:
    label = device['label']
    label_counts[label] = label_counts.get(label, 0) + 1

label_names = {
    0: 'Normal',
    1: 'DNS Misconfig',
    2: 'DHCP Misconfig',
    3: 'Gateway Misconfig',
    4: 'ARP Storm'
}

print(f'✅ Created batch synthetic dataset with {len(devices)} devices')
print(f'   Saved to: {output_file}')
print(f'\nDistribution:')
for label, count in sorted(label_counts.items()):
    print(f'   {label_names[label]} (label {label}): {count} devices ({count/len(devices)*100:.1f}%)')
print(f'\nTotal: {len(devices)} devices')
print(f'   Normal: {label_counts.get(0, 0)} devices ({label_counts.get(0, 0)/len(devices)*100:.1f}%)')
print(f'   Misconfigured: {sum(v for k, v in label_counts.items() if k != 0)} devices ({sum(v for k, v in label_counts.items() if k != 0)/len(devices)*100:.1f}%)')
print(f'\nFirst 10 rows:')
with open(output_file, 'r') as f:
    lines = f.readlines()[:11]
    for line in lines:
        print(line.strip())

