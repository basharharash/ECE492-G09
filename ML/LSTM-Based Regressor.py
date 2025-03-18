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

# Define the LSTM-based model
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1):
        super(LSTMRegressor, self).__init__()
        # LSTM expects input shape: (batch, seq_len, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len); add a feature dimension to get (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_dim)
        # Use the output from the final time step
        out = out[:, -1, :]  # (batch, hidden_dim)
        output = self.fc(out)  # (batch, 1)
        return output.squeeze(-1)

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=128):
    train_dataset = RoastDataset(X_train, y_train)
    test_dataset = RoastDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=40, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # You can experiment with nn.L1Loss() or Huber loss if desired
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
    # Load the dataset from dataset.py
    window_size = 5
    X_train, y_train, X_test, y_test = load_all_csv_files_separated(modified_dir='ML\modified', window_size=window_size)
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=128)
    
    # Initialize the LSTM model
    model = LSTMRegressor(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1)
    device = 'cuda'
    print("Training on device:", device)
    
    # Train the model
    model = train_model(model, train_loader, test_loader, epochs=40, lr=1e-3, device=device)
    
    # Save the model
    torch.save(model.state_dict(), "lstm_regressor_v4.pt")
    print("Model saved as lstm_regressor.pt")
