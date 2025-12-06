"""
Multi-View Fusion Model
Combines DHCP, DNS, and Flow features using separate encoders.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


class MultiViewEncoder(nn.Module):
    """Multi-view encoder with separate encoders for each modality."""
    
    def __init__(self, dhcp_dim, dns_dim, flow_dim, embedding_dim=32):
        """
        Initialize multi-view encoder.
        
        Args:
            dhcp_dim: Number of DHCP features
            dns_dim: Number of DNS features
            flow_dim: Number of Flow features
            embedding_dim: Dimension of each modality embedding
        """
        super(MultiViewEncoder, self).__init__()
        
        # Separate encoders for each modality
        self.dhcp_encoder = nn.Sequential(
            nn.Linear(dhcp_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
        self.dns_encoder = nn.Sequential(
            nn.Linear(dns_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
        # Combined embedding dimension
        self.combined_dim = embedding_dim * 3
    
    def forward(self, dhcp_features, dns_features, flow_features):
        """
        Forward pass through multi-view encoders.
        
        Args:
            dhcp_features: DHCP feature tensor
            dns_features: DNS feature tensor
            flow_features: Flow feature tensor
            
        Returns:
            Combined embedding tensor
        """
        z_dhcp = self.dhcp_encoder(dhcp_features)
        z_dns = self.dns_encoder(dns_features)
        z_flow = self.flow_encoder(flow_features)
        
        # Concatenate embeddings
        z_combined = torch.cat([z_dhcp, z_dns, z_flow], dim=1)
        
        return z_combined, z_dhcp, z_dns, z_flow


class MultiViewFusionModel:
    """Complete multi-view fusion model with classifier."""
    
    def __init__(self, dhcp_dim, dns_dim, flow_dim, num_classes=5, embedding_dim=32):
        """
        Initialize multi-view fusion model.
        
        Args:
            dhcp_dim: Number of DHCP features
            dns_dim: Number of DNS features
            flow_dim: Number of Flow features
            num_classes: Number of misconfig classes
            embedding_dim: Embedding dimension per modality
        """
        self.encoder = MultiViewEncoder(dhcp_dim, dns_dim, flow_dim, embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.dhcp_scaler = StandardScaler()
        self.dns_scaler = StandardScaler()
        self.flow_scaler = StandardScaler()
        
        self.dhcp_dim = dhcp_dim
        self.dns_dim = dns_dim
        self.flow_dim = flow_dim
    
    def fit(self, X_dhcp, X_dns, X_flow, y, epochs=50, batch_size=32, device='cpu'):
        """
        Train the model.
        
        Args:
            X_dhcp: DHCP features
            X_dns: DNS features
            X_flow: Flow features
            y: Labels
            epochs: Training epochs
            batch_size: Batch size
            device: 'cpu' or 'cuda'
        """
        # Scale features
        X_dhcp_scaled = self.dhcp_scaler.fit_transform(X_dhcp)
        X_dns_scaled = self.dns_scaler.fit_transform(X_dns)
        X_flow_scaled = self.flow_scaler.fit_transform(X_flow)
        
        # Convert to tensors
        X_dhcp_t = torch.FloatTensor(X_dhcp_scaled).to(device)
        X_dns_t = torch.FloatTensor(X_dns_scaled).to(device)
        X_flow_t = torch.FloatTensor(X_flow_scaled).to(device)
        y_t = torch.LongTensor(y).to(device)
        
        # Move models to device
        self.encoder = self.encoder.to(device)
        self.classifier = self.classifier.to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=0.001
        )
        
        # Training loop
        self.encoder.train()
        self.classifier.train()
        
        dataset_size = len(X_dhcp_t)
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for i in range(0, dataset_size, batch_size):
                end_idx = min(i + batch_size, dataset_size)
                
                batch_dhcp = X_dhcp_t[i:end_idx]
                batch_dns = X_dns_t[i:end_idx]
                batch_flow = X_flow_t[i:end_idx]
                batch_y = y_t[i:end_idx]
                
                # Forward pass
                z_combined, _, _, _ = self.encoder(batch_dhcp, batch_dns, batch_flow)
                logits = self.classifier(z_combined)
                loss = criterion(logits, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
    
    def predict(self, X_dhcp, X_dns, X_flow, device='cpu'):
        """
        Predict labels.
        
        Args:
            X_dhcp: DHCP features
            X_dns: DNS features
            X_flow: Flow features
            device: 'cpu' or 'cuda'
            
        Returns:
            Predicted labels
        """
        self.encoder.eval()
        self.classifier.eval()
        
        # Scale features
        X_dhcp_scaled = self.dhcp_scaler.transform(X_dhcp)
        X_dns_scaled = self.dns_scaler.transform(X_dns)
        X_flow_scaled = self.flow_scaler.transform(X_flow)
        
        # Convert to tensors
        X_dhcp_t = torch.FloatTensor(X_dhcp_scaled).to(device)
        X_dns_t = torch.FloatTensor(X_dns_scaled).to(device)
        X_flow_t = torch.FloatTensor(X_flow_scaled).to(device)
        
        with torch.no_grad():
            z_combined, _, _, _ = self.encoder(X_dhcp_t, X_dns_t, X_flow_t)
            logits = self.classifier(z_combined)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X_dhcp, X_dns, X_flow, device='cpu'):
        """
        Predict label probabilities.
        
        Args:
            X_dhcp: DHCP features
            X_dns: DNS features
            X_flow: Flow features
            device: 'cpu' or 'cuda'
            
        Returns:
            Probability matrix
        """
        self.encoder.eval()
        self.classifier.eval()
        
        # Scale features
        X_dhcp_scaled = self.dhcp_scaler.transform(X_dhcp)
        X_dns_scaled = self.dns_scaler.transform(X_dns)
        X_flow_scaled = self.flow_scaler.transform(X_flow)
        
        # Convert to tensors
        X_dhcp_t = torch.FloatTensor(X_dhcp_scaled).to(device)
        X_dns_t = torch.FloatTensor(X_dns_scaled).to(device)
        X_flow_t = torch.FloatTensor(X_flow_scaled).to(device)
        
        with torch.no_grad():
            z_combined, _, _, _ = self.encoder(X_dhcp_t, X_dns_t, X_flow_t)
            logits = self.classifier(z_combined)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
    
    def get_embeddings(self, X_dhcp, X_dns, X_flow, device='cpu'):
        """
        Get combined embeddings.
        
        Args:
            X_dhcp: DHCP features
            X_dns: DNS features
            X_flow: Flow features
            device: 'cpu' or 'cuda'
            
        Returns:
            Combined embeddings
        """
        self.encoder.eval()
        
        # Scale features
        X_dhcp_scaled = self.dhcp_scaler.transform(X_dhcp)
        X_dns_scaled = self.dns_scaler.transform(X_dns)
        X_flow_scaled = self.flow_scaler.transform(X_flow)
        
        # Convert to tensors
        X_dhcp_t = torch.FloatTensor(X_dhcp_scaled).to(device)
        X_dns_t = torch.FloatTensor(X_dns_scaled).to(device)
        X_flow_t = torch.FloatTensor(X_flow_scaled).to(device)
        
        with torch.no_grad():
            z_combined, _, _, _ = self.encoder(X_dhcp_t, X_dns_t, X_flow_t)
        
        return z_combined.cpu().numpy()

