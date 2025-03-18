import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataset import load_all_csv_files_separated  # Import your dataset-building function

# Define the PyTorch Dataset (same as before)
class RoastDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the Transformer model with batch_first=True
class TransformerRegressorBatchFirst(nn.Module):
    def __init__(self, window_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerRegressorBatchFirst, self).__init__()
        # Project each scalar input (1 value) to a d_model-dimensional vector
        self.input_linear = nn.Linear(1, d_model)
        # Transformer Encoder layer with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Pooling layer: average across the time dimension
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x shape: (batch_size, window_size)
        # Add feature dimension: (batch_size, window_size, 1)
        x = x.unsqueeze(-1)
        # Linear projection: (batch_size, window_size, d_model)
        x = self.input_linear(x)
        # Pass through the transformer: stays as (batch_size, window_size, d_model)
        x = self.transformer_encoder(x)
        # Transpose for pooling: (batch_size, d_model, window_size)
        x = x.transpose(1, 2)
        # Adaptive average pooling to get a fixed-size vector (batch_size, d_model, 1)
        x = self.pool(x).squeeze(-1)
        # Final regression output: (batch_size, 1) -> squeeze to (batch_size)
        output = self.fc_out(x)
        return output.squeeze(-1)

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=128):
    train_dataset = RoastDataset(X_train, y_train)
    test_dataset = RoastDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                test_loss += loss.item() * X_batch.size(0)
        test_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    return model

if __name__ == "__main__":
    window_size = 5
    X_train, y_train, X_test, y_test = load_all_csv_files_separated(modified_dir='modified', window_size=window_size)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=128)
    
    model = TransformerRegressorBatchFirst(window_size=window_size, d_model=64, nhead=4, num_layers=2, dropout=0.1)
    device = 'cuda'
    print("Training on device:", device)
    
    model = train_model(model, train_loader, test_loader, epochs=20, lr=1e-3, device=device)
    torch.save(model.state_dict(), "transformer_regressor_batch_first.pt")
    print("Model saved as transformer_regressor_batch_first.pt")
