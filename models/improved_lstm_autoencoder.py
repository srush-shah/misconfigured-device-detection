"""
Improved LSTM Autoencoder for Temporal Anomaly Detection
Enhanced with attention, bidirectional encoding, learning rate scheduling, and early stopping.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import warnings


class ImprovedLSTMAutoencoder(nn.Module):
    """Improved LSTM-based autoencoder with attention and bidirectional encoding."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True, use_attention=True):
        """
        Initialize improved LSTM autoencoder.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden state size (increased from 64)
            num_layers: Number of LSTM layers
            dropout: Dropout rate (increased from 0.2)
            bidirectional: Use bidirectional LSTM
            use_attention: Use attention mechanism
        """
        super(ImprovedLSTMAutoencoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.directions = 2 if bidirectional else 1
        
        # Encoder: Bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size * self.directions,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Decoder: Bidirectional LSTM
        self.decoder = nn.LSTM(
            hidden_size * self.directions,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output projection with residual connection
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Reconstructed tensor of same shape as input
        """
        # Normalize input
        x_norm = self.layer_norm(x)
        
        # Encode
        encoded, (hidden, cell) = self.encoder(x_norm)
        
        # Apply attention if enabled
        if self.use_attention:
            encoded, _ = self.attention(encoded, encoded, encoded)
        
        # Decode
        decoded, _ = self.decoder(encoded)
        
        # Project to original feature space
        output = self.fc(decoded)
        
        # Residual connection
        output = output + x  # Residual connection
        
        return output
    
    def encode(self, x):
        """
        Encode input to hidden representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded representation and hidden state
        """
        x_norm = self.layer_norm(x)
        encoded, (hidden, cell) = self.encoder(x_norm)
        
        if self.use_attention:
            encoded, _ = self.attention(encoded, encoded, encoded)
        
        return encoded, hidden


class ImprovedLSTMAutoencoderTrainer:
    """Improved trainer with learning rate scheduling, early stopping, and gradient clipping."""
    
    def __init__(self, model, device='cpu', learning_rate=0.001, weight_decay=1e-5):
        """
        Initialize improved trainer.
        
        Args:
            model: ImprovedLSTMAutoencoder model
            device: 'cpu' or 'cuda'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
        """
        self.model = model.to(device)
        self.device = device
        
        # Use Huber loss (more robust than MSE)
        self.criterion = nn.HuberLoss(delta=1.0)
        
        # Adam optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.training_history = []
    
    def train(self, train_loader, epochs=100, early_stopping_patience=15, 
              gradient_clip=1.0, verbose=True, validation_loader=None):
        """
        Train the autoencoder with improved techniques.
        
        Args:
            train_loader: DataLoader with training sequences
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            gradient_clip: Gradient clipping threshold
            verbose: Print training progress
            validation_loader: Optional validation DataLoader for early stopping
        """
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                reconstructed = self.model(batch)
                loss = self.criterion(reconstructed, batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            self.training_history.append(avg_loss)
            
            # Validation loss for early stopping
            val_loss = None
            if validation_loader is not None:
                val_loss = self._validate(validation_loader)
                self.model.train()  # Set back to training mode
            
            # Learning rate scheduling
            loss_for_scheduler = val_loss if val_loss is not None else avg_loss
            self.scheduler.step(loss_for_scheduler)
            
            # Early stopping
            if val_loss is not None:
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                    # Save best model state
                    self.best_model_state = self.model.state_dict().copy()
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"\nEarly stopping at epoch {epoch+1}")
                        # Load best model
                        self.model.load_state_dict(self.best_model_state)
                        break
            
            if verbose and (epoch + 1) % 10 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                scheduler_info = ""
                if hasattr(self.scheduler, 'last_epoch') and self.scheduler.last_epoch > 0:
                    scheduler_info = f" (LR reduced {self.scheduler.num_bad_epochs} times)"
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {lr:.6f}" + 
                      (f", Val Loss: {val_loss:.6f}" if val_loss is not None else "") + scheduler_info)
    
    def _validate(self, validation_loader):
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(self.device)
                reconstructed = self.model(batch)
                loss = self.criterion(reconstructed, batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def compute_reconstruction_error(self, data_loader):
        """
        Compute reconstruction errors for sequences.
        
        Args:
            data_loader: DataLoader with sequences
            
        Returns:
            numpy array of reconstruction errors
        """
        self.model.eval()
        errors = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                reconstructed = self.model(batch)
                
                # Compute MSE per sample (more interpretable than Huber loss)
                mse = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
                errors.extend(mse.cpu().numpy())
        
        return np.array(errors)
    
    def get_embeddings(self, data_loader):
        """
        Get encoded embeddings for sequences.
        
        Args:
            data_loader: DataLoader with sequences
            
        Returns:
            numpy array of embeddings
        """
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                encoded, _ = self.model.encode(batch)
                # Use mean pooling over sequence length
                pooled = torch.mean(encoded, dim=1)
                embeddings.extend(pooled.cpu().numpy())
        
        return np.array(embeddings)

