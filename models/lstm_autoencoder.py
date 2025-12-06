"""
LSTM/GRU Autoencoder for Temporal Anomaly Detection
Detects misconfigurations by learning normal temporal patterns.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """Dataset for sequence data."""
    
    def __init__(self, sequences):
        """
        Initialize dataset.
        
        Args:
            sequences: numpy array of shape (N, T, F) where N=samples, T=time steps, F=features
        """
        self.sequences = torch.FloatTensor(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


class LSTMAutoencoder(nn.Module):
    """LSTM-based autoencoder for sequence reconstruction."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        """
        Initialize LSTM autoencoder.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden state size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMAutoencoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Reconstructed tensor of same shape as input
        """
        # Encode
        encoded, (hidden, cell) = self.encoder(x)
        
        # Use last hidden state as context
        # Repeat hidden state for decoder
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Decoder input: use encoded output
        decoded, _ = self.decoder(encoded)
        
        # Project to original feature space
        output = self.fc(decoded)
        
        return output
    
    def encode(self, x):
        """
        Encode input to hidden representation.
        
        Args:
            x: Input tensor
            
        Returns:
            Encoded representation
        """
        encoded, (hidden, cell) = self.encoder(x)
        return encoded, hidden


class LSTMAutoencoderTrainer:
    """Trainer for LSTM autoencoder."""
    
    def __init__(self, model, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model: LSTMAutoencoder model
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    def train(self, train_loader, epochs=50, verbose=True):
        """
        Train the autoencoder.
        
        Args:
            train_loader: DataLoader with training sequences
            epochs: Number of training epochs
            verbose: Print training progress
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
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
    
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
                
                # Compute MSE per sample
                mse = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
                errors.extend(mse.cpu().numpy())
        
        return np.array(errors)

