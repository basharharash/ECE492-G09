import torch
import torch.nn as nn
import time
import numpy as np

# Assume LSTMRegressor is defined as in training
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, sequence_length)
        # Add feature dimension -> (batch, sequence_length, 1)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        # Use the output from the last time step
        out = out[:, -1, :]
        output = self.fc(out)
        return output.squeeze(-1)

# Load the saved model
model = LSTMRegressor(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1)
model.load_state_dict(torch.load("lstm_regressor.pt"))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# Define the window size used in training (e.g., 5 seconds)
window_size = 5
current_window = []  # This list will store the latest 'window_size' bean temperature readings

# Function to simulate getting a new reading from your sensor.
def get_new_reading():
    # In a real scenario, replace this with code that reads from your sensor.
    # For example, read from an ADC or serial port.
    # Here we simulate by returning a random value near 25°C.
    return np.random.normal(loc=25, scale=2)

print("Starting real-time prediction. Press Ctrl+C to stop.")
try:
    while True:
        new_reading = get_new_reading()
        current_window.append(new_reading)
        # Keep only the last 'window_size' readings
        if len(current_window) > window_size:
            current_window.pop(0)
        
        if len(current_window) == window_size:
            # Convert the current window into a tensor of shape (1, window_size)
            input_tensor = torch.tensor([current_window], dtype=torch.float32).to(device)
            with torch.no_grad():
                prediction = model(input_tensor)
            print(f"New reading: {new_reading:.2f}°C | Current window: {current_window} | Predicted gas setting: {prediction.item():.2f}")
        
        # Wait for 1 second before getting the next reading
        time.sleep(1)
except KeyboardInterrupt:
    print("Real-time prediction stopped.")
