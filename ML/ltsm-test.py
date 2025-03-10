import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Import your dataset-building function from dataset.py
from dataset import load_all_csv_files_separated

#############################################
#          Define the PyTorch Dataset       #
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
#        Define Your LSTM Model             #
#############################################
# (Make sure this definition matches the one used during training)
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

#############################################
#              Evaluation Steps             #
#############################################
def evaluate_model(model, test_loader, device):
    model.eval()  # set model to evaluation mode
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = mse_loss(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(all_preds - all_targets))
    
    return avg_loss, mae, all_preds, all_targets

#############################################
#               Main Script                 #
#############################################
if __name__ == "__main__":
    # Define the window size (same as used during training)
    window_size = 5
    
    # Load your dataset (we use the test portion)
    _, _, X_test, y_test = load_all_csv_files_separated(modified_dir='modified', window_size=window_size)
    print("Test set shapes: X:", X_test.shape, "y:", y_test.shape)
    
    # Create a DataLoader for the test set
    test_dataset = RoastDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize your model and load trained weights
    model = LSTMRegressor(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1)
    model.load_state_dict(torch.load("lstm_regressor.pt"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print("Evaluating model on device:", device)
    
    # Evaluate the model
    test_mse, test_mae, predictions, targets = evaluate_model(model, test_loader, device)
    print(f"Test MSE Loss: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Plot a sample of predictions vs. actual values
    plt.figure(figsize=(10, 6))
    plt.plot(targets[:100], label="Actual")
    plt.plot(predictions[:100], label="Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Gas Setting")
    plt.title("Actual vs. Predicted Gas Setting (first 100 test samples)")
    plt.legend()
    plt.show()
