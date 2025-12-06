#!/usr/bin/env python3
"""
Main Pipeline Execution Script
Runs the complete misconfiguration detection pipeline.
"""

import sys
import os
import argparse
from pipeline.main_pipeline import MainPipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def create_labels(df):
    """Create labels based on feature patterns."""
    labels = []
    
    for idx, row in df.iterrows():
        label = 0  # Normal by default
        
        # DNS misconfig
        if row.get('dns_failure_ratio', 0) > 0.5 and row.get('dns_query_count', 0) > 5:
            label = 1
        # DHCP misconfig
        elif row.get('dhcp_discover_count', 0) > 10 and row.get('dhcp_ack_count', 0) < row.get('dhcp_discover_count', 0) * 0.5:
            label = 2
        # Gateway misconfig
        elif row.get('num_distinct_gateways', 0) > 3:
            label = 3
        # ARP storm
        elif row.get('arp_request_count', 0) > 50 or (row.get('broadcast_packet_ratio', 0) > 0.3 and row.get('arp_request_count', 0) > 20):
            label = 4
        
        labels.append(label)
    
    return np.array(labels)


def main():
    parser = argparse.ArgumentParser(description='Run misconfiguration detection pipeline')
    parser.add_argument('--dhcp-log', type=str, help='Path to DHCP log file')
    parser.add_argument('--dns-log', type=str, help='Path to DNS log file')
    parser.add_argument('--flow-csv', type=str, help='Path to Flow CSV file')
    parser.add_argument('--output', type=str, default='data/output/misconfig_report.csv',
                       help='Output report path')
    parser.add_argument('--train-only', action='store_true', help='Only train models, no prediction')
    parser.add_argument('--baseline-only', action='store_true', help='Use baseline models only')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MainPipeline(config={
        'time_window_minutes': 5,
        'sequence_length': 12,
        'device': 'cpu',
        'batch_size': 32,
        'lstm_epochs': 50,
        'multi_view_epochs': 50,
        'n_clusters': 5,
        'confidence_threshold': 0.7
    })
    
    # Step 1: Ingest data
    print("="*60)
    print("STEP 1: DATA INGESTION")
    print("="*60)
    
    dhcp_features, dns_features, flow_features = pipeline.ingest_data(
        dhcp_log_path=args.dhcp_log,
        dns_log_path=args.dns_log,
        flow_csv_path=args.flow_csv
    )
    
    if dhcp_features.empty and dns_features.empty and flow_features.empty:
        print("ERROR: No data files provided or found!")
        print("Please provide data files or run prepare_data.py first to create synthetic data.")
        return 1
    
    # Step 2: Extract features
    print("\n" + "="*60)
    print("STEP 2: FEATURE EXTRACTION")
    print("="*60)
    
    combined_features = pipeline.extract_features(dhcp_features, dns_features, flow_features)
    
    if combined_features.empty:
        print("ERROR: No features extracted!")
        return 1
    
    # Step 3: Create labels
    print("\n" + "="*60)
    print("STEP 3: LABEL CREATION")
    print("="*60)
    
    combined_features['label'] = create_labels(combined_features)
    
    print("Label Distribution:")
    print(pd.Series(combined_features['label']).value_counts().sort_index())
    
    # Step 4: Split data
    X = combined_features.copy()
    y = combined_features['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_normal = X_train[X_train['label'] == 0].copy()
    
    print(f"\nTraining set: {len(X_train)} windows")
    print(f"Test set: {len(X_test)} windows")
    print(f"Normal data: {len(X_normal)} windows")
    
    # Step 5: Train models
    print("\n" + "="*60)
    print("STEP 4: MODEL TRAINING")
    print("="*60)
    
    pipeline.train_baselines(X_train, y_train, X_normal=X_normal)
    
    if not args.baseline_only:
        pipeline.train_advanced_models(X_train, y_train, X_normal=X_normal)
    
    if args.train_only:
        print("\nTraining complete. Exiting.")
        return 0
    
    # Step 6: Generate predictions
    print("\n" + "="*60)
    print("STEP 5: PREDICTION")
    print("="*60)
    
    predictions = pipeline.predict(X_test, use_advanced=not args.baseline_only)
    
    # Step 7: Generate report
    print("\n" + "="*60)
    print("STEP 6: REPORT GENERATION")
    print("="*60)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report = pipeline.generate_report(X_test, predictions, output_path=args.output)
    
    print(f"\nPipeline complete! Report saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

