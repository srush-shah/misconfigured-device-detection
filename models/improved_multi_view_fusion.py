"""
Improved Multi-View Fusion Model
Enhanced with attention, batch normalization, class weights, and learning rate scheduling.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings


class AttentionFusion(nn.Module):
    """Attention mechanism for fusing multi-view embeddings."""
    
    def __init__(self, embedding_dim, num_views=3):
        """
        Initialize attention fusion.
        
        Args:
            embedding_dim: Dimension of each view embedding
            num_views: Number of views (DHCP, DNS, Flow)
        """
        super(AttentionFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_views = num_views
        
        # Attention weights
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * num_views, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_views),
            nn.Softmax(dim=1)
        )
    
    def forward(self, embeddings_list):
        """
        Apply attention-weighted fusion.
        
        Args:
            embeddings_list: List of view embeddings [dhcp, dns, flow]
            
        Returns:
            Weighted combined embedding
        """
        # Concatenate all embeddings
        combined = torch.cat(embeddings_list, dim=1)
        
        # Compute attention weights
        attention_weights = self.attention(combined)
        
        # Weighted sum
        weighted_embeddings = []
        for i, emb in enumerate(embeddings_list):
            weight = attention_weights[:, i:i+1]
            weighted_embeddings.append(emb * weight)
        
        # Sum weighted embeddings
        fused = sum(weighted_embeddings)
        
        return fused, attention_weights


class ImprovedMultiViewEncoder(nn.Module):
    """Improved multi-view encoder with batch normalization and deeper networks."""
    
    def __init__(self, dhcp_dim, dns_dim, flow_dim, embedding_dim=64):
        """
        Initialize improved multi-view encoder.
        
        Args:
            dhcp_dim: Number of DHCP features
            dns_dim: Number of DNS features
            flow_dim: Number of Flow features
            embedding_dim: Dimension of each modality embedding (increased from 32)
        """
        super(ImprovedMultiViewEncoder, self).__init__()
        
        # DHCP encoder with layer normalization (more stable for small batches)
        self.dhcp_encoder = nn.Sequential(
            nn.Linear(dhcp_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
        # DNS encoder with layer normalization
        self.dns_encoder = nn.Sequential(
            nn.Linear(dns_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
        # Flow encoder with layer normalization
        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embedding_dim)
        )
        
        # Attention fusion
        self.attention_fusion = AttentionFusion(embedding_dim, num_views=3)
    
    def forward(self, dhcp_features, dns_features, flow_features):
        """
        Forward pass through improved multi-view encoders.
        
        Args:
            dhcp_features: DHCP feature tensor
            dns_features: DNS feature tensor
            flow_features: Flow feature tensor
            
        Returns:
            Fused embedding and attention weights
        """
        z_dhcp = self.dhcp_encoder(dhcp_features)
        z_dns = self.dns_encoder(dns_features)
        z_flow = self.flow_encoder(flow_features)
        
        # Attention-weighted fusion
        z_fused, attention_weights = self.attention_fusion([z_dhcp, z_dns, z_flow])
        
        return z_fused, z_dhcp, z_dns, z_flow, attention_weights


class ImprovedMultiViewFusionModel:
    """Improved multi-view fusion model with better architecture and training."""
    
    def __init__(self, dhcp_dim, dns_dim, flow_dim, num_classes=5, embedding_dim=64):
        """
        Initialize improved multi-view fusion model.
        
        Args:
            dhcp_dim: Number of DHCP features
            dns_dim: Number of DNS features
            flow_dim: Number of Flow features
            num_classes: Number of misconfig classes
            embedding_dim: Embedding dimension per modality
        """
        self.encoder = ImprovedMultiViewEncoder(dhcp_dim, dns_dim, flow_dim, embedding_dim)
        
        # Store embedding_dim for later use
        self.embedding_dim = embedding_dim
        
        # Improved classifier with layer normalization
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
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
        self.num_classes = num_classes
    
    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data."""
        from collections import Counter
        import numpy as np
        
        counts = Counter(y)
        total = len(y)
        unique_classes = sorted(counts.keys())
        n_unique = len(unique_classes)
        
        # Compute weights only for classes that exist
        weights = {}
        for class_id in unique_classes:
            weights[class_id] = total / (n_unique * counts[class_id])
        
        # Create weight tensor with proper indexing
        # Map class IDs to indices in the weight tensor
        weight_list = []
        for i in range(self.num_classes):
            if i in weights:
                weight_list.append(weights[i])
            else:
                # For missing classes, use a small weight
                weight_list.append(1.0)
        
        weight_tensor = torch.FloatTensor(weight_list)
        return weight_tensor
    
    def fit(self, X_dhcp, X_dns, X_flow, y, epochs=100, batch_size=32, device='cpu',
            learning_rate=0.001, weight_decay=1e-4, use_class_weights=True):
        """
        Train the improved model.
        
        Args:
            X_dhcp: DHCP features
            X_dns: DNS features
            X_flow: Flow features
            y: Labels
            epochs: Training epochs (increased from 50)
            batch_size: Batch size
            device: 'cpu' or 'cuda'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            use_class_weights: Use class weights for imbalanced data
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
        
        # Training setup with class weights
        # Get unique classes in the data
        unique_classes = sorted(np.unique(y))
        max_class = max(unique_classes)
        
        # Ensure num_classes is at least max_class + 1
        if max_class >= self.num_classes:
            print(f"Warning: Data contains class {max_class} but model expects {self.num_classes} classes.")
            print(f"  Expanding model to {max_class + 1} classes.")
            # Rebuild classifier with correct number of classes
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, max_class + 1)
            ).to(device)
            self.num_classes = max_class + 1
        
        if use_class_weights:
            class_weights = self._compute_class_weights(y).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Adam optimizer with weight decay
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Training loop
        self.encoder.train()
        self.classifier.train()
        
        dataset_size = len(X_dhcp_t)
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        best_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 20
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            # Shuffle data
            indices = torch.randperm(dataset_size)
            X_dhcp_shuffled = X_dhcp_t[indices]
            X_dns_shuffled = X_dns_t[indices]
            X_flow_shuffled = X_flow_t[indices]
            y_shuffled = y_t[indices]
            
            for i in range(0, dataset_size, batch_size):
                end_idx = min(i + batch_size, dataset_size)
                
                batch_dhcp = X_dhcp_shuffled[i:end_idx]
                batch_dns = X_dns_shuffled[i:end_idx]
                batch_flow = X_flow_shuffled[i:end_idx]
                batch_y = y_shuffled[i:end_idx]
                
                # Forward pass
                z_fused, _, _, _, _ = self.encoder(batch_dhcp, batch_dns, batch_flow)
                logits = self.classifier(z_fused)
                loss = criterion(logits, batch_y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.classifier.parameters()),
                    max_norm=1.0
                )
                
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                scheduler_info = ""
                if hasattr(scheduler, 'num_bad_epochs') and scheduler.num_bad_epochs > 0:
                    scheduler_info = f" (LR reduced {scheduler.num_bad_epochs} times)"
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {lr:.6f}{scheduler_info}")
    
    def predict(self, X_dhcp, X_dns, X_flow, device='cpu'):
        """Predict labels."""
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
            z_fused, _, _, _, _ = self.encoder(X_dhcp_t, X_dns_t, X_flow_t)
            logits = self.classifier(z_fused)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X_dhcp, X_dns, X_flow, device='cpu'):
        """Predict label probabilities."""
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
            z_fused, _, _, _, _ = self.encoder(X_dhcp_t, X_dns_t, X_flow_t)
            logits = self.classifier(z_fused)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()
    
    def get_embeddings(self, X_dhcp, X_dns, X_flow, device='cpu'):
        """Get fused embeddings."""
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
            z_fused, _, _, _, _ = self.encoder(X_dhcp_t, X_dns_t, X_flow_t)
        
        return z_fused.cpu().numpy()

