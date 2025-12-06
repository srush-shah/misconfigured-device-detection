"""
Knowledge-Based Explanation Generator
Converts model outputs into human-readable explanations.
"""

import pandas as pd
import numpy as np


class ExplanationGenerator:
    """Generate human-readable explanations for misconfigurations."""
    
    def __init__(self):
        """Initialize explanation generator."""
        self.misconfig_types = {
            0: "Normal Configuration",
            1: "DNS Misconfiguration",
            2: "DHCP Misconfiguration",
            3: "Gateway Misconfiguration",
            4: "ARP Storm / Local Misconfiguration",
            -1: "Unknown Misconfiguration Type"
        }
    
    def generate_explanation(self, row, predicted_label, confidence_score, 
                            reconstruction_error=None, cluster_distance=None):
        """
        Generate explanation for a single prediction.
        
        Args:
            row: Feature row (pandas Series or dict)
            predicted_label: Predicted misconfig label
            confidence_score: Confidence score
            reconstruction_error: Reconstruction error (if available)
            cluster_distance: Cluster distance (if available)
            
        Returns:
            Human-readable explanation string
        """
        explanations = []
        
        # Base explanation
        misconfig_name = self.misconfig_types.get(predicted_label, "Unknown")
        explanations.append(f"Detected: {misconfig_name} (confidence: {confidence_score:.2f})")
        
        # DNS misconfig explanations
        if predicted_label == 1:
            if row.get('dns_failure_ratio', 0) > 0.5:
                explanations.append(f"High DNS failure ratio: {row.get('dns_failure_ratio', 0):.2%}")
            if row.get('dns_query_count', 0) > 10:
                explanations.append(f"High DNS query count: {row.get('dns_query_count', 0)}")
            if row.get('entropy_of_domains', 0) > 3.0:
                explanations.append("Suspicious domain patterns detected (high entropy)")
            explanations.append("Likely cause: Device cannot resolve hostnames properly")
        
        # DHCP misconfig explanations
        elif predicted_label == 2:
            if row.get('dhcp_discover_count', 0) > 10:
                explanations.append(f"Excessive DHCP DISCOVER messages: {row.get('dhcp_discover_count', 0)}")
            if row.get('failed_lease_ratio', 0) > 0.5:
                explanations.append(f"High lease failure ratio: {row.get('failed_lease_ratio', 0):.2%}")
            if row.get('dhcp_ack_count', 0) < row.get('dhcp_discover_count', 0) * 0.5:
                explanations.append("Device not receiving DHCP ACK responses")
            explanations.append("Likely cause: Device not getting valid IP lease from DHCP server")
        
        # Gateway misconfig explanations
        elif predicted_label == 3:
            if row.get('num_distinct_gateways', 0) > 3:
                explanations.append(f"Multiple gateways contacted: {row.get('num_distinct_gateways', 0)}")
            explanations.append("Likely cause: Device configured with incorrect or multiple gateways")
        
        # ARP storm explanations
        elif predicted_label == 4:
            if row.get('arp_request_count', 0) > 50:
                explanations.append(f"Excessive ARP requests: {row.get('arp_request_count', 0)}")
            if row.get('broadcast_packet_ratio', 0) > 0.3:
                explanations.append(f"High broadcast traffic: {row.get('broadcast_packet_ratio', 0):.2%}")
            explanations.append("Likely cause: ARP storm or local network misconfiguration")
        
        # Unknown misconfig explanations
        elif predicted_label == -1:
            explanations.append("Pattern does not match known misconfig types")
            if reconstruction_error is not None:
                explanations.append(f"High reconstruction error: {reconstruction_error:.4f} (unusual pattern)")
            if cluster_distance is not None:
                explanations.append(f"Far from known patterns: {cluster_distance:.4f}")
            explanations.append("Recommendation: Manual investigation required")
        
        # Normal configuration
        else:
            explanations.append("Device shows normal network behavior patterns")
        
        return " | ".join(explanations)
    
    def generate_report(self, df, predictions, confidence_scores, 
                      reconstruction_errors=None, cluster_distances=None):
        """
        Generate complete report with explanations.
        
        Args:
            df: Feature DataFrame
            predictions: Predicted labels
            confidence_scores: Confidence scores
            reconstruction_errors: Reconstruction errors (optional)
            cluster_distances: Cluster distances (optional)
            
        Returns:
            DataFrame with device_id, misconfig_type, confidence_score, explanation
        """
        results = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            pred_label = predictions[idx]
            conf_score = confidence_scores[idx] if isinstance(confidence_scores, np.ndarray) else confidence_scores
            
            recon_error = reconstruction_errors[idx] if reconstruction_errors is not None else None
            cluster_dist = cluster_distances[idx] if cluster_distances is not None else None
            
            explanation = self.generate_explanation(
                row, pred_label, conf_score, recon_error, cluster_dist
            )
            
            results.append({
                'device_id': row.get('device_id', 'unknown'),
                'time_window': row.get('time_window', None),
                'misconfig_type': self.misconfig_types.get(pred_label, "Unknown"),
                'misconfig_label': pred_label,
                'confidence_score': conf_score,
                'explanation': explanation
            })
        
        return pd.DataFrame(results)

