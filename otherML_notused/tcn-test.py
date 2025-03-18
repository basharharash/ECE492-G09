import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Import your dataset-building function from dataset.py
from dataset import load_all_csv_files_separated

#######################################################
#      Define the same TCN model used for training    #
#######################################################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

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
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        # Global average pooling over time dimension
        y = torch.mean(y, dim=2)
        out = self.linear(y)
        return out.squeeze()

class TCNRegressor(nn.Module):
    def __init__(self, window_size, num_channels=[64, 64], kernel_size=2, dropout=0.2):
        super(TCNRegressor, self).__init__()
        self.tcn = TCN(input_size=1, output_size=1, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
    
    def forward(self, x):
        # x shape: (batch, window_size)
        x = x.unsqueeze(-1)  # add feature dimension -> (batch, window_size, 1)
        return self.tcn(x)

#######################################################
#           PyTorch Dataset for Test Data             #
#######################################################
class RoastDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#######################################################
#           Evaluation Function & Main Script         #
#######################################################
def evaluate_model(model, test_loader, device):
    model.eval()
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = mse_loss(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    mae = np.mean(np.abs(all_preds - all_targets))
    return avg_loss, mae, all_preds, all_targets

if __name__ == "__main__":
    # 1. Load the test data
    window_size = 5
    _, _, X_test, y_test = load_all_csv_files_separated(modified_dir='modified', window_size=window_size)
    print("Test set shapes: X:", X_test.shape, "y:", y_test.shape)
    
    # 2. Create test DataLoader
    test_dataset = RoastDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 3. Initialize your TCN model (same hyperparams used during training)
    model = TCNRegressor(window_size=window_size, num_channels=[64, 64], kernel_size=2, dropout=0.2)
    
    # 4. Load the trained weights
    model.load_state_dict(torch.load("tcn_regressor.pt"))
    
    # 5. Evaluate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    test_mse, test_mae, predictions, targets = evaluate_model(model, test_loader, device)
    
    print(f"Test MSE Loss: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # 6. Plot a sample of predictions vs. actual values
    plt.figure(figsize=(10, 6))
    plt.plot(targets[:100], label="Actual")
    plt.plot(predictions[:100], label="Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Gas Setting")
    plt.title("TCN: Actual vs. Predicted Gas Setting (first 100 test samples)")
    plt.legend()
    plt.show()
