import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset import load_all_csv_files_separated

window_size = 5
X_train, y_train, X_test, y_test = load_all_csv_files_separated(modified_dir='modified', window_size=window_size)

# Assume X_train, y_train, X_test, y_test are loaded from your dataset building step.
# For example, you might load them from .npz files or simply run the previous script in the same runtime.
# For now, we assume they are available as numpy arrays.

# ---------------------------
# 1. Define PyTorch Dataset
# ---------------------------
class RoastDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to torch tensors (float32)
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (N, window_size)
        self.y = torch.tensor(y, dtype=torch.float32)  # shape: (N,)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ---------------------------
# 2. Define the Transformer Model
# ---------------------------
class TransformerRegressor(nn.Module):
    def __init__(self, window_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        """
        window_size: length of the input sequence (here 5)
        d_model: dimension for the transformer
        nhead: number of attention heads
        num_layers: number of transformer encoder layers
        dropout: dropout rate
        """
        super(TransformerRegressor, self).__init__()
        # Project each scalar input (1 value) to a d_model-dimensional vector
        self.input_linear = nn.Linear(1, d_model)
        
        # Define Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pool across the time (sequence) dimension to get a single vector per example.
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Final fully connected layer to predict a single value
        self.fc_out = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x shape: (batch_size, window_size)
        # Add a feature dimension: (batch_size, window_size, 1)
        x = x.unsqueeze(-1)
        # Linear projection: (batch_size, window_size, d_model)
        x = self.input_linear(x)
        # Transformer expects shape (sequence_length, batch_size, d_model)
        x = x.transpose(0, 1)  # shape: (window_size, batch_size, d_model)
        x = self.transformer_encoder(x)  # shape: (window_size, batch_size, d_model)
        # Transpose back to (batch_size, d_model, window_size)
        x = x.transpose(0, 1).transpose(1, 2)
        # Pool across the sequence dimension (window_size)
        x = self.pool(x).squeeze(-1)  # shape: (batch_size, d_model)
        # Final regression output: (batch_size, 1)
        output = self.fc_out(x)
        return output.squeeze(-1)  # shape: (batch_size)

# ---------------------------
# 3. Training Loop
# ---------------------------
def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)
        test_loss /= len(test_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    return model

# ---------------------------
# 4. Create Datasets and DataLoaders
# ---------------------------
def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=128):
    train_dataset = RoastDataset(X_train, y_train)
    test_dataset = RoastDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ---------------------------
# 5. Main: Initialize and Train the Model
# ---------------------------
if __name__ == "__main__":
    # Replace these with your actual data from the dataset building step.
    # Here, we assume X_train, y_train, X_test, y_test are available.
    # For example, you could load them from saved .npz files.
    # For demonstration, I'll assume they are already defined.
    # If not, please ensure to run the previous script that builds these arrays.
    
    # Example shapes printed earlier:
    # X_train shape: (217654, 5) and X_test shape: (37716, 5)
    # (Replace these with your actual data loading if needed.)
    
    # Create data loaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=128)
    
    # Initialize the TransformerRegressor model with window_size=5
    model = TransformerRegressor(window_size=5, d_model=64, nhead=4, num_layers=2, dropout=0.1)
    
    # Choose device
    device = 'cuda'
    print("Training on device:", device)
    
    # Train the model
    model = train_model(model, train_loader, test_loader, epochs=20, lr=1e-3, device=device)
    
    # Save the model to disk
    torch.save(model.state_dict(), "transformer_regressor.pt")
    print("Model saved as transformer_regressor.pt")
