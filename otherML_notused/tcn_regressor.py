import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from glob import glob

# Import your dataset-building function from dataset.py
from dataset import load_all_csv_files_separated

#############################################
#           Define the PyTorch Dataset      #
#############################################
class RoastDataset(Dataset):
    def __init__(self, X, y):
        # Convert numpy arrays to torch tensors (float32)
        self.X = torch.tensor(X, dtype=torch.float32)  # shape: (N, window_size)
        self.y = torch.tensor(y, dtype=torch.float32)  # shape: (N,)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#############################################
#          Define Chomp1d Layer             #
#############################################
class Chomp1d(nn.Module):
    """
    Chomps off the last `chomp_size` elements along the time dimension.
    This is used to ensure that the output of the convolution has the same length as the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

#############################################
#          Define the TCN Modules           #
#############################################
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2
        )
        # If the input and output channels differ, downsample the input with a 1x1 convolution.
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # The addition works now because both out and res have the same shape.
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        """
        input_size: number of input features (here 1)
        output_size: number of output features (here 1)
        num_channels: list of channel sizes for each TCN layer (e.g., [64, 64])
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation_size = 2 ** i
            # Compute padding so that output length equals input length.
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, features) --> need to transpose to (batch, features, sequence_length)
        x = x.transpose(1, 2)
        y = self.network(x)
        # Global average pooling over time dimension
        y = torch.mean(y, dim=2)
        out = self.linear(y)
        return out.squeeze()

# Wrapper regressor that uses the TCN
class TCNRegressor(nn.Module):
    def __init__(self, window_size, num_channels=[64, 64], kernel_size=2, dropout=0.2):
        super(TCNRegressor, self).__init__()
        # Our input is a single feature per time step.
        self.tcn = TCN(input_size=1, output_size=1, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
    
    def forward(self, x):
        # x shape: (batch, window_size)
        x = x.unsqueeze(-1)  # add a feature dimension: (batch, window_size, 1)
        return self.tcn(x)

#############################################
#          DataLoader and Training          #
#############################################
def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=128):
    train_dataset = RoastDataset(X_train, y_train)
    test_dataset = RoastDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=20, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # You can experiment with alternative losses if desired
    
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

#############################################
#                  Main                   #
#############################################
if __name__ == "__main__":
    # Define the window size (here, 5 seconds)
    window_size = 5
    
    # Load training and test data using your dataset-building function from dataset.py
    X_train, y_train, X_test, y_test = load_all_csv_files_separated(modified_dir='modified', window_size=window_size)
    print("Training set shapes: X:", X_train.shape, "y:", y_train.shape)
    print("Test set shapes: X:", X_test.shape, "y:", y_test.shape)
    
    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test, batch_size=128)
    
    # Initialize the TCN regressor
    model = TCNRegressor(window_size=window_size, num_channels=[64, 64], kernel_size=2, dropout=0.2)
    device = 'cuda'
    print("Training on device:", device)
    
    # Train the model
    model = train_model(model, train_loader, test_loader, epochs=20, lr=1e-3, device=device)
    
    # Save the model to disk
    torch.save(model.state_dict(), "tcn_regressor.pt")
    print("Model saved as tcn_regressor.pt")
