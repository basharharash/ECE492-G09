import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Import your dataset-building function from dataset.py
from dataset import load_all_csv_files_separated

#############################################
#         Define the PyTorch Dataset        #
#############################################
class RoastDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#############################################
#     Define the Transformer Model          #
#############################################
class TransformerRegressorBatchFirst(nn.Module):
    def __init__(self, window_size, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        """
        window_size: length of the input sequence (here, 5)
        d_model: dimension for the transformer
        nhead: number of attention heads
        num_layers: number of transformer encoder layers
        dropout: dropout rate
        """
        super(TransformerRegressorBatchFirst, self).__init__()
        # Project each scalar input to a d_model-dimensional vector.
        self.input_linear = nn.Linear(1, d_model)
        
        # Transformer encoder layer with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Adaptive average pooling over the time dimension.
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, 1)
    
    def forward(self, x):
        # x: (batch_size, window_size)
        # Add feature dimension: (batch_size, window_size, 1)
        x = x.unsqueeze(-1)
        # Linear projection: (batch_size, window_size, d_model)
        x = self.input_linear(x)
        # Pass through the transformer encoder (batch_first=True keeps shape as (batch, seq, d_model))
        x = self.transformer_encoder(x)
        # Transpose to (batch, d_model, window_size) for pooling.
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)  # shape: (batch, d_model)
        output = self.fc_out(x)       # shape: (batch, 1)
        return output.squeeze(-1)     # shape: (batch)

#############################################
#            Evaluation Function            #
#############################################
def evaluate_model(model, test_loader, device):
    model.eval()
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
    mae = np.mean(np.abs(all_preds - all_targets))
    return avg_loss, mae, all_preds, all_targets

#############################################
#               Main Script                 #
#############################################
if __name__ == "__main__":
    # Set the window size (same as training)
    window_size = 5
    
    # Load test data from dataset.py; this loads files from Valid_test and Invalid_test folders.
    _, _, X_test, y_test = load_all_csv_files_separated(modified_dir='modified', window_size=window_size)
    print("Test set shapes: X:", X_test.shape, "y:", y_test.shape)
    
    # Create the test DataLoader
    test_dataset = RoastDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize the Transformer model and load saved weights.
    model = TransformerRegressorBatchFirst(window_size=window_size, d_model=64, nhead=4, num_layers=2, dropout=0.1)
    model.load_state_dict(torch.load("transformer_regressor_batch_first.pt"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print("Evaluating model on device:", device)
    
    # Evaluate model
    test_mse, test_mae, predictions, targets = evaluate_model(model, test_loader, device)
    print(f"Test MSE Loss: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Plot first 100 test samples: Actual vs. Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(targets[:100], label="Actual")
    plt.plot(predictions[:100], label="Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Gas Setting")
    plt.title("Transformer Batch First: Actual vs. Predicted (first 100 test samples)")
    plt.legend()
    plt.show()
