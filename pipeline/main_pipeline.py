"""
Main Pipeline for Use Case 7
Orchestrates data ingestion, feature extraction, model training, and inference.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_ingest import (DHCPParser, DNSParser, FlowParser, WestermoLoader)
from features import FeatureAggregator, SequenceBuilder
from models import (
    RuleBasedDetector, BaselineClassifier, BaselineAnomalyDetector,
    LSTMAutoencoder, LSTMAutoencoderTrainer, SequenceDataset,
    MultiViewFusionModel, OpenSetDetector,
    ImprovedLSTMAutoencoder, ImprovedLSTMAutoencoderTrainer,
    ImprovedMultiViewFusionModel, ImprovedOpenSetDetector
)
from explanations import ExplanationGenerator
from torch.utils.data import DataLoader


class MainPipeline:
    """Main pipeline for misconfiguration detection."""
    
    def __init__(self, config=None):
        """
        Initialize pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        self.feature_aggregator = FeatureAggregator()
        self.sequence_builder = SequenceBuilder(sequence_length=self.config['sequence_length'])
        self.explanation_generator = ExplanationGenerator()
        
        # Models (will be initialized during training)
        self.rule_detector = None
        self.baseline_classifier = None
        self.anomaly_detector = None
        self.lstm_autoencoder = None
        self.multi_view_model = None
        self.open_set_detector = None
        
        # Feature groups
        self.feature_groups = None
        self.feature_columns = None
    
    def _default_config(self):
        """Default configuration."""
        return {
            'time_window_minutes': 5,
            'sequence_length': 12,
            'device': 'cpu',
            'batch_size': 32,
            'lstm_epochs': 50,
            'multi_view_epochs': 50,
            'n_clusters': 5,
            'confidence_threshold': 0.7,
            'use_improved_models': True,  # Use improved models by default
            'lstm_hidden_size': 128,  # Increased from 64
            'multi_view_embedding_dim': 64,  # Increased from 32
            'use_dbscan': False  # Use KMeans by default, DBSCAN for irregular clusters
        }
    
    def ingest_data(self, dhcp_log_path=None, dns_log_path=None, flow_csv_path=None,
                   westermo_base_path=None, dataset_type='reduced', max_files=None):
        """
        Ingest and parse all data sources.
        
        Args:
            dhcp_log_path: Path to DHCP log file or directory (legacy support)
            dns_log_path: Path to DNS log file or directory (legacy support)
            flow_csv_path: Path to Flow CSV file or directory (legacy support)
            westermo_base_path: Base path to Westermo dataset (e.g., 'data/raw/westermo')
            dataset_type: 'reduced' or 'extended' (default: 'reduced')
            max_files: Maximum number of flow files to load (None for all)
            
        Returns:
            Tuple of (dhcp_features, dns_features, flow_features)
        """
        print("="*60)
        print("DATA INGESTION")
        print("="*60)
        
        # Westermo dataset loading (primary method)
        if westermo_base_path:
            # Convert relative path to absolute
            if not os.path.isabs(westermo_base_path):
                westermo_base_path = os.path.abspath(westermo_base_path)
            
            print(f"\nLoading Westermo dataset from: {westermo_base_path}")
            print(f"Dataset type: {dataset_type}")
            
            # Verify path exists
            if not os.path.exists(westermo_base_path):
                print(f"⚠️  WARNING: Base path does not exist: {westermo_base_path}")
                print("Please check the path and ensure dataset is downloaded.")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            loader = WestermoLoader(
                base_path=westermo_base_path,
                dataset_type=dataset_type,
                time_window_minutes=self.config['time_window_minutes']
            )
            
            # Load data
            data = loader.load_all(use_flows=True, max_files=max_files)
            
            # Get flow data (conn_df)
            conn_df = data.get('conn', pd.DataFrame())
            
            if conn_df.empty:
                print("⚠️  WARNING: No data loaded from Westermo dataset")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
            print(f"✓ Loaded {len(conn_df):,} flows from Westermo dataset")
            
            # Extract features from Westermo flow data
            # Westermo flows are pre-aggregated, so we group by device and time window
            if not conn_df.empty:
                # Ensure we have required columns
                if 'orig_ip' in conn_df.columns:
                    conn_df['device_id'] = conn_df['orig_ip']
                elif 'sIPs' in conn_df.columns:
                    conn_df['device_id'] = conn_df['sIPs']
                else:
                    conn_df['device_id'] = 'unknown'
                
                # Ensure timestamp column
                if 'timestamp' not in conn_df.columns:
                    if 'start' in conn_df.columns:
                        # Convert Unix timestamp to datetime
                        conn_df['timestamp'] = pd.to_datetime(conn_df['start'], unit='s')
                    else:
                        conn_df['timestamp'] = pd.Timestamp.now()
                else:
                    # Ensure timestamp is datetime
                    if conn_df['timestamp'].dtype != 'datetime64[ns]':
                        # Try to convert - check if it's Unix timestamp
                        if conn_df['timestamp'].dtype in ['int64', 'float64']:
                            conn_df['timestamp'] = pd.to_datetime(conn_df['timestamp'], unit='s')
                        else:
                            conn_df['timestamp'] = pd.to_datetime(conn_df['timestamp'])
                
                # Group by device and time window
                conn_df['time_window'] = pd.to_datetime(conn_df['timestamp']).dt.floor(f"{self.config['time_window_minutes']}min")
                
                # Aggregate flow features
                agg_dict = {}
                if 'orig_bytes' in conn_df.columns:
                    agg_dict['orig_bytes'] = ['sum', 'mean', 'count']
                elif 'sBytesSum' in conn_df.columns:
                    agg_dict['sBytesSum'] = ['sum', 'mean', 'count']
                
                if 'resp_bytes' in conn_df.columns:
                    agg_dict['resp_bytes'] = ['sum', 'mean']
                elif 'rBytesSum' in conn_df.columns:
                    agg_dict['rBytesSum'] = ['sum', 'mean']
                
                if 'orig_pkts' in conn_df.columns:
                    agg_dict['orig_pkts'] = 'sum'
                elif 'sPackets' in conn_df.columns:
                    agg_dict['sPackets'] = 'sum'
                
                if 'resp_pkts' in conn_df.columns:
                    agg_dict['resp_pkts'] = 'sum'
                elif 'rPackets' in conn_df.columns:
                    agg_dict['rPackets'] = 'sum'
                
                if 'duration' in conn_df.columns:
                    agg_dict['duration'] = 'mean'
                
                if 'label' in conn_df.columns:
                    agg_dict['label'] = 'first'  # Keep label from first flow in window
                
                if agg_dict:
                    # Count unique devices before aggregation
                    unique_devices_before = conn_df['device_id'].nunique()
                    
                    # Aggregate - use dropna=False to preserve all device/time_window combinations
                    flow_features = conn_df.groupby(['device_id', 'time_window'], dropna=False).agg(agg_dict).reset_index()
                    
                    # Flatten column names (handle MultiIndex)
                    new_columns = ['device_id', 'time_window']
                    for col in flow_features.columns[2:]:
                        if isinstance(col, tuple):
                            if col[1]:
                                new_columns.append(f"{col[0]}_{col[1]}")
                            else:
                                new_columns.append(col[0])
                        else:
                            new_columns.append(col)
                    flow_features.columns = new_columns
                    
                    # Fill NaN values with 0 to preserve all devices
                    # This ensures devices with no flows in a window still appear
                    numeric_cols = flow_features.select_dtypes(include=[np.number]).columns
                    flow_features[numeric_cols] = flow_features[numeric_cols].fillna(0)
                    
                    # Verify all devices are preserved
                    unique_devices_after = flow_features['device_id'].nunique()
                    if unique_devices_before != unique_devices_after:
                        print(f"⚠️  Warning: Device count changed from {unique_devices_before} to {unique_devices_after} during aggregation")
                    else:
                        print(f"✓ All {unique_devices_after} devices preserved in aggregated features")
                    
                    # Rename label column if it exists (from aggregation)
                    # The aggregation creates 'label_first' when using 'first' aggregation
                    if 'label_first' in flow_features.columns:
                        flow_features.rename(columns={'label_first': 'label'}, inplace=True)
                    elif 'label' not in flow_features.columns:
                        # If no label column, try to get it from the original data
                        # Use mode (most common label) in each window, preferring misconfig labels
                        if 'label' in conn_df.columns:
                            def get_label_mode(x):
                                """Get label mode, preferring non-zero (misconfig) labels."""
                                if len(x) == 0:
                                    return 0
                                # Count labels
                                label_counts = x.value_counts()
                                # Prefer misconfig labels (non-zero) if they exist
                                misconfig_labels = label_counts[label_counts.index != 0]
                                if len(misconfig_labels) > 0:
                                    return misconfig_labels.index[0]  # Most common misconfig label
                                return label_counts.index[0]  # Most common label (likely 0)
                            
                            label_agg = conn_df.groupby(['device_id', 'time_window'])['label'].agg(get_label_mode).reset_index()
                            label_agg.columns = ['device_id', 'time_window', 'label']
                            flow_features = flow_features.merge(label_agg, on=['device_id', 'time_window'], how='left')
                            flow_features['label'] = flow_features['label'].fillna(0).astype(int)
                    
                    # Ensure label is integer type
                    if 'label' in flow_features.columns:
                        flow_features['label'] = flow_features['label'].astype(int)
                        
                    # Debug: Print label distribution
                    if 'label' in flow_features.columns:
                        print(f"\n✓ Labels preserved in flow features:")
                        label_counts = flow_features['label'].value_counts().sort_index()
                        for label, count in label_counts.items():
                            label_name = {0: 'Normal', 2: 'DHCP Misconfig'}.get(label, f'Label {label}')
                            print(f"  {label_name}: {count:,} ({count/len(flow_features)*100:.1f}%)")
                else:
                    # Fallback: just group by device and time window
                    flow_features = conn_df.groupby(['device_id', 'time_window']).size().reset_index(name='flow_count')
                    if 'label' in conn_df.columns:
                        flow_features['label'] = conn_df.groupby(['device_id', 'time_window'])['label'].first().values
            else:
                flow_features = pd.DataFrame()
            
            # Westermo dataset doesn't have separate DHCP/DNS logs in the same format
            # Return empty DataFrames for these (or extract from flows if needed)
            dhcp_features = pd.DataFrame()
            dns_features = pd.DataFrame()
            
            return dhcp_features, dns_features, flow_features
        
        # Legacy single file/directory loading (for backward compatibility)
        dhcp_parser = DHCPParser(time_window_minutes=self.config['time_window_minutes'])
        dns_parser = DNSParser(time_window_minutes=self.config['time_window_minutes'])
        flow_parser = FlowParser(time_window_minutes=self.config['time_window_minutes'])
        
        dhcp_features = pd.DataFrame()
        dns_features = pd.DataFrame()
        flow_features = pd.DataFrame()
        
        if dhcp_log_path and os.path.exists(str(dhcp_log_path)):
            print(f"Parsing DHCP logs: {dhcp_log_path}")
            dhcp_features = dhcp_parser.process(dhcp_log_path)
            print(f"  Extracted {len(dhcp_features)} DHCP feature windows")
        
        if dns_log_path and os.path.exists(str(dns_log_path)):
            print(f"Parsing DNS logs: {dns_log_path}")
            dns_features = dns_parser.process(dns_log_path)
            print(f"  Extracted {len(dns_features)} DNS feature windows")
        
        if flow_csv_path and os.path.exists(str(flow_csv_path)):
            print(f"Parsing Flow/Conn data: {flow_csv_path}")
            flow_features = flow_parser.process(flow_csv_path)
            print(f"  Extracted {len(flow_features)} Flow feature windows")
        
        return dhcp_features, dns_features, flow_features
    
    def extract_features(self, dhcp_features, dns_features, flow_features):
        """
        Aggregate features from all sources.
        
        Args:
            dhcp_features: DHCP features DataFrame
            dns_features: DNS features DataFrame
            flow_features: Flow features DataFrame
            
        Returns:
            Combined feature DataFrame
        """
        print("\n" + "="*60)
        print("FEATURE EXTRACTION")
        print("="*60)
        
        combined_features = self.feature_aggregator.aggregate(
            dhcp_features, dns_features, flow_features
        )
        
        if combined_features.empty:
            print("Warning: No features extracted!")
            return combined_features
        
        # Get feature groups for multi-view fusion
        self.feature_groups = self.feature_aggregator.get_feature_groups(combined_features)
        
        # All feature columns (excluding metadata)
        exclude_cols = ['device_id', 'time_window', 'label']
        self.feature_columns = [col for col in combined_features.columns 
                               if col not in exclude_cols]
        
        print(f"Combined features: {len(combined_features)} windows")
        print(f"Total features: {len(self.feature_columns)}")
        print(f"  DHCP features: {len(self.feature_groups['dhcp'])}")
        print(f"  DNS features: {len(self.feature_groups['dns'])}")
        print(f"  Flow features: {len(self.feature_groups['flow'])}")
        
        return combined_features
    
    def train_baselines(self, X_train, y_train, X_normal=None):
        """
        Train baseline models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_normal: Normal data only (for anomaly detector)
        """
        print("\n" + "="*60)
        print("TRAINING BASELINE MODELS")
        print("="*60)
        
        # Rule-based detector (no training needed)
        self.rule_detector = RuleBasedDetector()
        print("✓ Rule-based detector initialized")
        
        # Baseline classifier
        print("\nTraining RandomForest classifier...")
        self.baseline_classifier = BaselineClassifier()
        self.baseline_classifier.fit(X_train, y_train, self.feature_columns)
        print("✓ RandomForest classifier trained")
        
        # Anomaly detector (train on normal data only)
        if X_normal is not None:
            print("\nTraining IsolationForest anomaly detector...")
            self.anomaly_detector = BaselineAnomalyDetector()
            self.anomaly_detector.fit(X_normal, self.feature_columns)
            print("✓ IsolationForest anomaly detector trained")
    
    def train_advanced_models(self, X_train, y_train, X_normal=None):
        """
        Train advanced models (LSTM autoencoder, multi-view fusion).
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_normal: Normal data only (for autoencoder)
        """
        print("\n" + "="*60)
        print("TRAINING ADVANCED MODELS")
        print("="*60)
        
        # LSTM Autoencoder (train on normal sequences only)
        if X_normal is not None:
            print("\nTraining LSTM Autoencoder on normal sequences...")
            
            # Build sequences
            sequences_dict = self.sequence_builder.build_sequences(
                X_normal, self.feature_columns
            )
            
            if sequences_dict:
                # Combine all device sequences
                all_sequences = []
                for device_id, sequences in sequences_dict.items():
                    all_sequences.extend(sequences)
                
                if all_sequences:
                    all_sequences = np.array(all_sequences)
                    print(f"  Created {len(all_sequences)} sequences")
                    
                    # Initialize model
                    input_size = all_sequences.shape[2]
                    self.lstm_autoencoder = LSTMAutoencoder(
                        input_size=input_size,
                        hidden_size=64,
                        num_layers=2
                    )
                    
                    # Train
                    dataset = SequenceDataset(all_sequences)
                    train_loader = DataLoader(
                        dataset,
                        batch_size=self.config['batch_size'],
                        shuffle=True
                    )
                    
                    trainer = LSTMAutoencoderTrainer(
                        self.lstm_autoencoder,
                        device=self.config['device']
                    )
                    trainer.train(train_loader, epochs=self.config['lstm_epochs'], verbose=True)
                    print("✓ LSTM Autoencoder trained")
        
        # Multi-view fusion model
        print("\nTraining Multi-View Fusion model...")
        
        # Split features by modality
        X_dhcp = X_train[self.feature_groups['dhcp']].values if self.feature_groups['dhcp'] else np.zeros((len(X_train), 1))
        X_dns = X_train[self.feature_groups['dns']].values if self.feature_groups['dns'] else np.zeros((len(X_train), 1))
        X_flow = X_train[self.feature_groups['flow']].values if self.feature_groups['flow'] else np.zeros((len(X_train), 1))
        
        # Handle empty feature groups
        if len(self.feature_groups['dhcp']) == 0:
            X_dhcp = np.zeros((len(X_train), 1))
        if len(self.feature_groups['dns']) == 0:
            X_dns = np.zeros((len(X_train), 1))
        if len(self.feature_groups['flow']) == 0:
            X_flow = np.zeros((len(X_train), 1))
        
        # Use improved model if configured
        # Use max class + 1 to ensure all class indices are supported
        # (e.g., if classes are [0, 2], we need num_classes=3 to support indices 0, 1, 2)
        num_classes = int(np.max(y_train)) + 1
        if self.config.get('use_improved_models', True):
            self.multi_view_model = ImprovedMultiViewFusionModel(
                dhcp_dim=X_dhcp.shape[1],
                dns_dim=X_dns.shape[1],
                flow_dim=X_flow.shape[1],
                num_classes=num_classes,
                embedding_dim=self.config.get('multi_view_embedding_dim', 64)
            )
            
            self.multi_view_model.fit(
                X_dhcp, X_dns, X_flow, y_train,
                epochs=self.config['multi_view_epochs'],
                batch_size=self.config['batch_size'],
                device=self.config['device'],
                learning_rate=0.001,
                weight_decay=1e-4,
                use_class_weights=True
            )
            print("✓ Improved Multi-View Fusion model trained")
        else:
            self.multi_view_model = MultiViewFusionModel(
                dhcp_dim=X_dhcp.shape[1],
                dns_dim=X_dns.shape[1],
                flow_dim=X_flow.shape[1],
                num_classes=num_classes
            )
            
            self.multi_view_model.fit(
                X_dhcp, X_dns, X_flow, y_train,
                epochs=self.config['multi_view_epochs'],
                batch_size=self.config['batch_size'],
                device=self.config['device']
            )
            print("✓ Multi-View Fusion model trained")
        
        # Open-set detector (fit clustering on embeddings)
        print("\nFitting clustering for open-set detection...")
        embeddings = self.multi_view_model.get_embeddings(
            X_dhcp, X_dns, X_flow, device=self.config['device']
        )
        
        # Use improved detector if configured
        if self.config.get('use_improved_models', True):
            self.open_set_detector = ImprovedOpenSetDetector(
                confidence_threshold=self.config['confidence_threshold'],
                use_dbscan=self.config.get('use_dbscan', False)
            )
        else:
            self.open_set_detector = OpenSetDetector(
                confidence_threshold=self.config['confidence_threshold']
            )
        self.open_set_detector.fit_clustering(embeddings, n_clusters=self.config['n_clusters'])
        print("✓ Open-set detector configured")
    
    def predict(self, X_test, use_advanced=True):
        """
        Generate predictions using trained models.
        
        Args:
            X_test: Test features
            use_advanced: Use advanced models (True) or baselines only (False)
            
        Returns:
            Dictionary with predictions from different models
        """
        print("\n" + "="*60)
        print("GENERATING PREDICTIONS")
        print("="*60)
        
        results = {}
        
        # Rule-based predictions
        if self.rule_detector:
            print("Running rule-based detector...")
            rule_preds = self.rule_detector.predict(X_test)
            results['rule_based'] = rule_preds
        
        # Baseline classifier predictions
        if self.baseline_classifier:
            print("Running baseline classifier...")
            baseline_preds = self.baseline_classifier.predict(X_test)
            baseline_probs = self.baseline_classifier.predict_proba(X_test)
            results['baseline_classifier'] = {
                'predictions': baseline_preds,
                'probabilities': baseline_probs
            }
        
        # Advanced models
        if use_advanced and self.multi_view_model:
            print("Running advanced models...")
            
            # Multi-view predictions
            X_dhcp = X_test[self.feature_groups['dhcp']].values if self.feature_groups['dhcp'] else np.zeros((len(X_test), 1))
            X_dns = X_test[self.feature_groups['dns']].values if self.feature_groups['dns'] else np.zeros((len(X_test), 1))
            X_flow = X_test[self.feature_groups['flow']].values if self.feature_groups['flow'] else np.zeros((len(X_test), 1))
            
            # Handle empty feature groups
            if len(self.feature_groups['dhcp']) == 0:
                X_dhcp = np.zeros((len(X_test), 1))
            if len(self.feature_groups['dns']) == 0:
                X_dns = np.zeros((len(X_test), 1))
            if len(self.feature_groups['flow']) == 0:
                X_flow = np.zeros((len(X_test), 1))
            
            multi_view_preds = self.multi_view_model.predict(
                X_dhcp, X_dns, X_flow, device=self.config['device']
            )
            multi_view_probs = self.multi_view_model.predict_proba(
                X_dhcp, X_dns, X_flow, device=self.config['device']
            )
            
            # Get embeddings
            embeddings = self.multi_view_model.get_embeddings(
                X_dhcp, X_dns, X_flow, device=self.config['device']
            )
            
            # Reconstruction errors (if autoencoder available)
            reconstruction_errors = None
            if self.lstm_autoencoder:
                print("Computing reconstruction errors...")
                sequences_dict = self.sequence_builder.build_sequences(
                    X_test, self.feature_columns
                )
                
                if sequences_dict:
                    all_sequences = []
                    sequence_indices = []  # Track which test samples have sequences
                    
                    for device_id, sequences in sequences_dict.items():
                        all_sequences.extend(sequences)
                        # Note: We can't easily map sequences back to original indices
                        # So we'll create one error per sequence and pad if needed
                    
                    if all_sequences:
                        dataset = SequenceDataset(np.array(all_sequences))
                        loader = DataLoader(dataset, batch_size=self.config['batch_size'])
                        
                        trainer = LSTMAutoencoderTrainer(
                            self.lstm_autoencoder,
                            device=self.config['device']
                        )
                        sequence_errors = trainer.compute_reconstruction_error(loader)
                        
                        # Map sequence errors back to test samples
                        # For now, use mean error per device or pad with median
                        if len(sequence_errors) != len(X_test):
                            # Create a mapping: use mean error for each device
                            # Or pad with median
                            median_error = np.median(sequence_errors)
                            reconstruction_errors = np.full(len(X_test), median_error)
                            print(f"  Note: {len(sequence_errors)} sequence errors mapped to {len(X_test)} test samples")
                        else:
                            reconstruction_errors = sequence_errors
            
            # Cluster distances
            cluster_distances = None
            if self.open_set_detector:
                cluster_distances = self.open_set_detector.compute_cluster_distances(embeddings)
            
            # Open-set detection
            open_set_preds = self.open_set_detector.predict_with_unknown(
                multi_view_probs,
                multi_view_preds,
                reconstruction_errors,
                embeddings
            )
            
            results['multi_view'] = {
                'predictions': multi_view_preds,
                'probabilities': multi_view_probs
            }
            results['open_set'] = {
                'predictions': open_set_preds,
                'reconstruction_errors': reconstruction_errors,
                'cluster_distances': cluster_distances,
                'embeddings': embeddings
            }
        
        return results
    
    def generate_report(self, X_test, predictions_dict, output_path=None):
        """
        Generate final report with explanations.
        
        Args:
            X_test: Test features
            predictions_dict: Dictionary of predictions from predict()
            output_path: Path to save report CSV
            
        Returns:
            DataFrame with final report
        """
        print("\n" + "="*60)
        print("GENERATING FINAL REPORT")
        print("="*60)
        
        # Use open-set predictions if available, otherwise multi-view, otherwise baseline
        if 'open_set' in predictions_dict:
            preds = predictions_dict['open_set']['predictions']
            probs = predictions_dict['multi_view']['probabilities']
            recon_errors = predictions_dict['open_set'].get('reconstruction_errors')
            cluster_dists = predictions_dict['open_set'].get('cluster_distances')
        elif 'multi_view' in predictions_dict:
            preds = predictions_dict['multi_view']['predictions']
            probs = predictions_dict['multi_view']['probabilities']
            recon_errors = None
            cluster_dists = None
        elif 'baseline_classifier' in predictions_dict:
            preds = predictions_dict['baseline_classifier']['predictions']
            probs = predictions_dict['baseline_classifier']['probabilities']
            recon_errors = None
            cluster_dists = None
        else:
            raise ValueError("No predictions available")
        
        # Confidence scores (max probability)
        confidence_scores = np.max(probs, axis=1)
        
        # Generate explanations
        report = self.explanation_generator.generate_report(
            X_test,
            preds,
            confidence_scores,
            reconstruction_errors=recon_errors,
            cluster_distances=cluster_dists
        )
        
        if output_path:
            report.to_csv(output_path, index=False)
            print(f"Report saved to: {output_path}")
        
        print(f"\nGenerated report for {len(report)} devices")
        print("\nMisconfiguration Summary:")
        print(report['misconfig_type'].value_counts())
        
        return report

