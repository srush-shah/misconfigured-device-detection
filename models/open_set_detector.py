"""
Open-Set Detector
Combines classifier confidence, reconstruction error, and clustering distance
to detect unknown misconfig types.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class OpenSetDetector:
    """Open-set misconfiguration detector."""
    
    def __init__(self, confidence_threshold=0.7, reconstruction_threshold=None, 
                 cluster_distance_threshold=None):
        """
        Initialize open-set detector.
        
        Args:
            confidence_threshold: Minimum classifier confidence for known classes
            reconstruction_threshold: Threshold for reconstruction error (auto if None)
            cluster_distance_threshold: Threshold for cluster distance (auto if None)
        """
        self.confidence_threshold = confidence_threshold
        self.reconstruction_threshold = reconstruction_threshold
        self.cluster_distance_threshold = cluster_distance_threshold
        
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_centers_ = None
    
    def fit_clustering(self, embeddings, n_clusters=5):
        """
        Fit K-Means clustering on embeddings.
        
        Args:
            embeddings: Embedding vectors
            n_clusters: Number of clusters
        """
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(embeddings_scaled)
        self.cluster_centers_ = self.kmeans.cluster_centers_
    
    def compute_cluster_distances(self, embeddings):
        """
        Compute distances to nearest cluster center.
        
        Args:
            embeddings: Embedding vectors
            
        Returns:
            Array of distances to nearest cluster
        """
        if self.kmeans is None:
            raise ValueError("Clustering not fitted. Call fit_clustering first.")
        
        embeddings_scaled = self.scaler.transform(embeddings)
        distances = self.kmeans.transform(embeddings_scaled)
        min_distances = np.min(distances, axis=1)
        
        return min_distances
    
    def detect_unknown(self, classifier_probs, reconstruction_errors=None, 
                      embeddings=None, auto_threshold=True):
        """
        Detect unknown misconfigurations.
        
        Args:
            classifier_probs: Classifier probability matrix
            reconstruction_errors: Reconstruction errors from autoencoder
            embeddings: Embedding vectors for clustering
            auto_threshold: Automatically set thresholds from data
            
        Returns:
            Boolean array: True for unknown, False for known
        """
        n_samples = len(classifier_probs)
        is_unknown = np.zeros(n_samples, dtype=bool)
        
        # 1. Check classifier confidence
        max_probs = np.max(classifier_probs, axis=1)
        low_confidence = max_probs < self.confidence_threshold
        
        # 2. Check reconstruction error (if available)
        high_reconstruction = np.zeros(n_samples, dtype=bool)
        if reconstruction_errors is not None:
            # Ensure reconstruction_errors has the same length as classifier_probs
            if len(reconstruction_errors) != n_samples:
                print(f"Warning: Reconstruction errors length ({len(reconstruction_errors)}) != classifier length ({n_samples})")
                print("  Padding reconstruction errors with median value")
                # Pad with median value
                median_error = np.median(reconstruction_errors)
                reconstruction_errors = np.full(n_samples, median_error)
            
            if auto_threshold and self.reconstruction_threshold is None:
                # Use 95th percentile as threshold
                self.reconstruction_threshold = np.percentile(reconstruction_errors, 95)
            
            if self.reconstruction_threshold is not None:
                high_reconstruction = reconstruction_errors > self.reconstruction_threshold
        
        # 3. Check cluster distance (if available)
        far_from_clusters = np.zeros(n_samples, dtype=bool)
        if embeddings is not None and self.kmeans is not None:
            # Ensure embeddings have the same length
            if len(embeddings) != n_samples:
                print(f"Warning: Embeddings length ({len(embeddings)}) != classifier length ({n_samples})")
                print("  Padding embeddings with mean value")
                # Pad with mean embedding
                mean_embedding = np.mean(embeddings, axis=0)
                if len(embeddings) < n_samples:
                    padding = np.tile(mean_embedding, (n_samples - len(embeddings), 1))
                    embeddings = np.vstack([embeddings, padding])
                else:
                    embeddings = embeddings[:n_samples]
            
            cluster_distances = self.compute_cluster_distances(embeddings)
            
            if auto_threshold and self.cluster_distance_threshold is None:
                # Use 95th percentile as threshold
                self.cluster_distance_threshold = np.percentile(cluster_distances, 95)
            
            if self.cluster_distance_threshold is not None:
                far_from_clusters = cluster_distances > self.cluster_distance_threshold
        
        # Combine conditions: unknown if low confidence AND (high reconstruction OR far from clusters)
        if reconstruction_errors is not None or embeddings is not None:
            is_unknown = low_confidence & (high_reconstruction | far_from_clusters)
        else:
            is_unknown = low_confidence
        
        return is_unknown
    
    def predict_with_unknown(self, classifier_probs, classifier_predictions,
                            reconstruction_errors=None, embeddings=None):
        """
        Predict with unknown class (-1) for open-set cases.
        
        Args:
            classifier_probs: Classifier probability matrix
            classifier_predictions: Classifier predictions
            reconstruction_errors: Reconstruction errors
            embeddings: Embedding vectors
            
        Returns:
            Predictions with -1 for unknown misconfigs
        """
        is_unknown = self.detect_unknown(
            classifier_probs,
            reconstruction_errors,
            embeddings
        )
        
        predictions = classifier_predictions.copy()
        predictions[is_unknown] = -1  # Unknown misconfig type
        
        return predictions

