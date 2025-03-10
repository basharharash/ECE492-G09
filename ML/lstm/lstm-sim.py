import pandas as pd
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt

# Define the LSTM model (same as during training)
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch, sequence_length)
        # Add feature dimension: (batch, sequence_length, 1)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        # Use the output from the last time step
        out = out[:, -1, :]
        output = self.fc(out)
        return output.squeeze(-1)

# 1. Load the saved model weights
model = LSTMRegressor(input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.1)
model.load_state_dict(torch.load("lstm/lstm_regressor.pt"))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()

# 2. Load the CSV file simulating a real-time roast run.
csv_file = "lstm/real_time_roast.csv"  # Adjust the path as needed.
df = pd.read_csv(csv_file)

# 3. Set up a sliding window buffer and simulation parameters.
window_size = 5
current_window = []

# For storing predictions and actual values (if available)
predicted_list = []
actual_list = []
time_list = []

print("Starting fast real-time simulation from CSV...")
# Adjust sleep_time for faster simulation (e.g., 0.1 seconds per reading).
sleep_time = 0.1

for index, row in df.iterrows():
    # Get the current bean temperature reading
    bean_temp_value = row["bean_temp"]
    current_window.append(bean_temp_value)
    
    # Keep only the last 'window_size' readings
    if len(current_window) > window_size:
        current_window.pop(0)
    
    # If we have a full window, make a prediction
    if len(current_window) == window_size:
        # Convert the window to tensor shape (1, window_size)
        input_tensor = torch.tensor([current_window], dtype=torch.float32).to(device)
        with torch.no_grad():
            prediction = model(input_tensor)
        predicted_value = prediction.item()
        predicted_list.append(predicted_value)
        # Append the corresponding actual gas setting if available in CSV
        # (Assuming the CSV contains a column "gas_setting")
        actual_value = row["gas_setting"] if "gas_setting" in row else None
        actual_list.append(actual_value)
        time_list.append(row["Time"])
        print(f"Time: {row['Time']} | Window: {current_window} | Predicted Gas Setting: {predicted_value:.2f}")
    else:
        print(f"Time: {row['Time']} | Collecting data... ({len(current_window)}/{window_size})")
    
    # Wait a short time to simulate fast real-time processing
    time.sleep(sleep_time)

print("Real-time simulation finished.")

# Calculate custom accuracy if actual values are available.
# Define tolerance (e.g., ±5 units).
tolerance = 5
if actual_list and any(val is not None for val in actual_list):
    predicted_array = np.array(predicted_list)
    actual_array = np.array(actual_list, dtype=float)  # Ensure these are floats
    custom_accuracy = np.mean(np.abs(predicted_array - actual_array) <= tolerance)
    print(f"Custom Accuracy (within ±{tolerance} units): {custom_accuracy * 100:.2f}%")
else:
    print("Actual gas setting values not available; cannot compute custom accuracy.")

# Plot the original vs predicted curves
plt.figure(figsize=(12, 6))
if actual_list and any(val is not None for val in actual_list):
    plt.plot(time_list, actual_list, label="Actual Gas Setting", linewidth=2)
plt.plot(time_list, predicted_list, label="Predicted Gas Setting", linestyle="--", linewidth=2)
plt.xlabel("Time")
plt.ylabel("Gas Setting")
plt.title("Real-Time Simulation: Actual vs. Predicted Gas Setting")
plt.legend()
plt.grid(True)
plt.show()
