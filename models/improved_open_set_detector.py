"""
Improved Open-Set Detector
Enhanced with DBSCAN option, adaptive thresholds, and ensemble detection.
"""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings


class ImprovedOpenSetDetector:
    """Improved open-set misconfiguration detector with multiple clustering methods."""
    
    def __init__(self, confidence_threshold=0.7, reconstruction_threshold=None, 
                 cluster_distance_threshold=None, use_dbscan=False):
        """
        Initialize improved open-set detector.
        
        Args:
            confidence_threshold: Minimum classifier confidence for known classes
            reconstruction_threshold: Threshold for reconstruction error (auto if None)
            cluster_distance_threshold: Threshold for cluster distance (auto if None)
            use_dbscan: Use DBSCAN instead of KMeans (better for irregular clusters)
        """
        self.confidence_threshold = confidence_threshold
        self.reconstruction_threshold = reconstruction_threshold
        self.cluster_distance_threshold = cluster_distance_threshold
        self.use_dbscan = use_dbscan
        
        self.kmeans = None
        self.dbscan = None
        self.scaler = StandardScaler()
        self.cluster_centers_ = None
        self.cluster_labels_ = None
    
    def fit_clustering(self, embeddings, n_clusters=5, eps=0.5, min_samples=5):
        """
        Fit clustering on embeddings (KMeans or DBSCAN).
        
        Args:
            embeddings: Embedding vectors
            n_clusters: Number of clusters for KMeans
            eps: Maximum distance for DBSCAN
            min_samples: Minimum samples for DBSCAN
        """
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        if self.use_dbscan:
            # Use DBSCAN for irregular clusters
            self.dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
            self.cluster_labels_ = self.dbscan.fit_predict(embeddings_scaled)
            
            # Compute cluster centers from core samples
            unique_labels = set(self.cluster_labels_)
            unique_labels.discard(-1)  # Remove noise label
            
            if len(unique_labels) > 0:
                centers = []
                for label in unique_labels:
                    mask = self.cluster_labels_ == label
                    center = embeddings_scaled[mask].mean(axis=0)
                    centers.append(center)
                self.cluster_centers_ = np.array(centers)
            else:
                # Fallback to KMeans if DBSCAN finds no clusters
                print("Warning: DBSCAN found no clusters, falling back to KMeans")
                self.use_dbscan = False
                self.fit_clustering(embeddings, n_clusters=n_clusters)
        else:
            # Use KMeans
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
            self.kmeans.fit(embeddings_scaled)
            self.cluster_centers_ = self.kmeans.cluster_centers_
            self.cluster_labels_ = self.kmeans.labels_
    
    def compute_cluster_distances(self, embeddings):
        """
        Compute distances to nearest cluster center.
        
        Args:
            embeddings: Embedding vectors
            
        Returns:
            Array of distances to nearest cluster
        """
        if self.cluster_centers_ is None:
            raise ValueError("Clustering not fitted. Call fit_clustering first.")
        
        embeddings_scaled = self.scaler.transform(embeddings)
        
        if self.use_dbscan:
            # For DBSCAN, compute distance to nearest core point
            distances = []
            for emb in embeddings_scaled:
                if len(self.cluster_centers_) > 0:
                    dists = np.linalg.norm(self.cluster_centers_ - emb, axis=1)
                    distances.append(np.min(dists))
                else:
                    distances.append(float('inf'))
            return np.array(distances)
        else:
            # For KMeans, use transform
            distances = self.kmeans.transform(embeddings_scaled)
            return np.min(distances, axis=1)
    
    def detect_unknown(self, classifier_probs, reconstruction_errors=None, 
                      embeddings=None, auto_threshold=True, ensemble_voting='majority'):
        """
        Detect unknown misconfigurations with improved ensemble method.
        
        Args:
            classifier_probs: Classifier probability matrix
            reconstruction_errors: Reconstruction errors from autoencoder
            embeddings: Embedding vectors for clustering
            auto_threshold: Automatically set thresholds from data
            ensemble_voting: 'majority' or 'unanimous' voting
            
        Returns:
            Boolean array: True for unknown, False for known
        """
        n_samples = len(classifier_probs)
        is_unknown = np.zeros(n_samples, dtype=bool)
        
        # Collect votes from different methods
        votes = []
        
        # 1. Classifier confidence vote
        max_probs = np.max(classifier_probs, axis=1)
        low_confidence = max_probs < self.confidence_threshold
        votes.append(low_confidence)
        
        # 2. Reconstruction error vote
        high_reconstruction = np.zeros(n_samples, dtype=bool)
        if reconstruction_errors is not None:
            if len(reconstruction_errors) != n_samples:
                # Align reconstruction errors
                median_error = np.median(reconstruction_errors)
                reconstruction_errors = np.full(n_samples, median_error)
            
            if auto_threshold and self.reconstruction_threshold is None:
                # Use adaptive threshold (95th percentile)
                self.reconstruction_threshold = np.percentile(reconstruction_errors, 95)
            
            if self.reconstruction_threshold is not None:
                high_reconstruction = reconstruction_errors > self.reconstruction_threshold
                votes.append(high_reconstruction)
        
        # 3. Cluster distance vote
        far_from_clusters = np.zeros(n_samples, dtype=bool)
        if embeddings is not None and self.cluster_centers_ is not None:
            if len(embeddings) != n_samples:
                # Align embeddings
                mean_embedding = np.mean(embeddings, axis=0)
                if len(embeddings) < n_samples:
                    padding = np.tile(mean_embedding, (n_samples - len(embeddings), 1))
                    embeddings = np.vstack([embeddings, padding])
                else:
                    embeddings = embeddings[:n_samples]
            
            cluster_distances = self.compute_cluster_distances(embeddings)
            
            if auto_threshold and self.cluster_distance_threshold is None:
                # Use adaptive threshold (95th percentile)
                self.cluster_distance_threshold = np.percentile(cluster_distances, 95)
            
            if self.cluster_distance_threshold is not None:
                far_from_clusters = cluster_distances > self.cluster_distance_threshold
                votes.append(far_from_clusters)
        
        # Ensemble voting
        if len(votes) > 1:
            votes_array = np.array(votes)
            if ensemble_voting == 'majority':
                # Unknown if majority of methods agree
                is_unknown = np.sum(votes_array, axis=0) >= (len(votes) / 2)
            elif ensemble_voting == 'unanimous':
                # Unknown if all methods agree
                is_unknown = np.all(votes_array, axis=0)
            else:
                # Default: at least one method indicates unknown
                is_unknown = np.any(votes_array, axis=0)
        else:
            is_unknown = votes[0]
        
        return is_unknown
    
    def predict_with_unknown(self, classifier_probs, classifier_predictions,
                            reconstruction_errors=None, embeddings=None,
                            ensemble_voting='majority'):
        """
        Predict with unknown class (-1) for open-set cases.
        
        Args:
            classifier_probs: Classifier probability matrix
            classifier_predictions: Classifier predictions
            reconstruction_errors: Reconstruction errors
            embeddings: Embedding vectors
            ensemble_voting: Voting method for ensemble
            
        Returns:
            Predictions with -1 for unknown misconfigs
        """
        is_unknown = self.detect_unknown(
            classifier_probs,
            reconstruction_errors,
            embeddings,
            ensemble_voting=ensemble_voting
        )
        
        predictions = classifier_predictions.copy()
        predictions[is_unknown] = -1  # Unknown misconfig type
        
        return predictions
    
    def get_cluster_quality_score(self, embeddings):
        """
        Compute clustering quality score (silhouette score).
        
        Args:
            embeddings: Embedding vectors
            
        Returns:
            Silhouette score (higher is better)
        """
        if self.cluster_labels_ is None:
            return None
        
        embeddings_scaled = self.scaler.transform(embeddings)
        
        # Remove noise points for DBSCAN
        if self.use_dbscan:
            valid_mask = self.cluster_labels_ != -1
            if np.sum(valid_mask) < 2:
                return None
            labels = self.cluster_labels_[valid_mask]
            embeddings_valid = embeddings_scaled[valid_mask]
        else:
            labels = self.cluster_labels_
            embeddings_valid = embeddings_scaled
        
        if len(set(labels)) < 2:
            return None
        
        try:
            score = silhouette_score(embeddings_valid, labels)
            return score
        except:
            return None

