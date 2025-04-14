import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------
# 1. Load and Preprocess Data
# -------------------------
# Read CSV file (ensure the file path is correct)
df = pd.read_csv('../data/preprocessed/dataframe_Preprocessed.csv', parse_dates=['date'])

# Example: filter for one customer (e.g., Customer == 0) and sort by date
customer_id = 145
df_customer = df[df['Customer'] == customer_id].copy()
df_customer.sort_values('date', inplace=True)
df_customer.reset_index(drop=True, inplace=True)

# Select features:
# Use 'consumption_daily_normalized' as both a feature and the forecasting target.
# Also use 'is_holiday_or_weekend' and 'saison' as additional features.
features = ['consumption_daily_normalized', 'is_holiday_or_weekend', 'saison']
data = df_customer[features].values.astype(np.float32)

# -------------------------
# 2. Define a Sliding Window Dataset for Time Series Forecasting
# -------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, forecast_horizon):
        """
        data: numpy array of shape (num_time_steps, num_features)
        window_size: number of past steps used as input
        forecast_horizon: number of future steps to predict
        """
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_horizon + 1

    def __getitem__(self, idx):
        # Input sequence of shape (window_size, num_features)
        x = self.data[idx : idx + self.window_size, :]
        # Forecast target: we predict only the normalized consumption (assumed to be the first feature)
        y = self.data[idx + self.window_size : idx + self.window_size + self.forecast_horizon, 0]
        return x, y

# Set hyperparameters for the dataset and model
window_size = 30          # Use the past 30 days as input
forecast_horizon = 7      # Forecast the next 7 days
feature_size = len(features)

# -------------------------
# 3. Split Data into Training and Test Sets
# -------------------------
# We use a contiguous split to preserve temporal order.
# Use the first 80% of the data for training and the rest for testing.
split_idx = int(0.8 * len(data))
# For the test set, include extra past data to allow a full window at the very beginning.
train_data = data[:split_idx]
test_data = data[split_idx - window_size:]  # include window_size extra rows

train_dataset = TimeSeriesDataset(train_data, window_size, forecast_horizon)
test_dataset = TimeSeriesDataset(test_data, window_size, forecast_horizon)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------
# 4. Define the Transformer Model for Time Series Forecasting
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size, d_model=64, nhead=4, num_encoder_layers=2, 
                 dim_feedforward=128, dropout=0.1, forecast_horizon=7):
        """
        feature_size: number of features per time step
        forecast_horizon: number of future time steps to predict
        """
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model

        # Project input features to d_model dimensions
        self.input_linear = nn.Linear(feature_size, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Forecasting head: project encoder output to forecast horizon predictions
        self.forecasting_head = nn.Linear(d_model, forecast_horizon)

    def forward(self, src):
        """
        src: shape (batch_size, seq_length, feature_size)
        """
        # Project input features
        src = self.input_linear(src)  # (batch_size, seq_length, d_model)

        # Add positional encoding
        src = self.positional_encoding(src)

        # Transformer expects input as (seq_length, batch_size, d_model)
        src = src.permute(1, 0, 2)
        encoder_output = self.transformer_encoder(src)
        
        # Use the output of the last time step for forecasting
        last_output = encoder_output[-1, :, :]  # (batch_size, d_model)

        # Forecast the next forecast_horizon steps (target: normalized consumption)
        prediction = self.forecasting_head(last_output)  # (batch_size, forecast_horizon)
        return prediction

# Instantiate the model
model = TimeSeriesTransformer(feature_size=feature_size, forecast_horizon=forecast_horizon)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# -------------------------
# 5. Define Loss, Optimizer, and Training Loop
# -------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20

for epoch in range(1, num_epochs + 1):
    model.train()
    train_losses = []
    for x, y in train_loader:
        x = x.to(device)  # (batch, window_size, feature_size)
        y = y.to(device)  # (batch, forecast_horizon)
        
        optimizer.zero_grad()
        outputs = model(x)  # (batch, forecast_horizon)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)
    print(f"Epoch {epoch}/{num_epochs}, Training Loss: {avg_train_loss:.6f}")

# -------------------------
# 6. Evaluate the Model on the Test Set
# -------------------------
model.eval()
test_losses = []
predictions = []
ground_truths = []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = criterion(output, y)
        test_losses.append(loss.item())
        predictions.append(output.cpu().numpy())
        ground_truths.append(y.cpu().numpy())

avg_test_loss = np.mean(test_losses)
print(f"Test Loss: {avg_test_loss:.6f}")

# Optionally, concatenate predictions and ground truths for further analysis.
predictions = np.concatenate(predictions, axis=0)
ground_truths = np.concatenate(ground_truths, axis=0)

# -------------------------
# 7. Plot Real Values vs. Predicted Values for the First Test Sample
# -------------------------
# For the first sample in the test set:
first_pred = predictions[0]
first_truth = ground_truths[0]

plt.figure(figsize=(10, 5))
days = np.arange(1, forecast_horizon+1)
plt.plot(days, first_truth, marker='o', label='Ground Truth')
plt.plot(days, first_pred, marker='x', label='Predicted')
plt.xlabel("Forecast Horizon (Days)")
plt.ylabel("Normalized Consumption")
plt.title("7-Day Forecast: Ground Truth vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

# Optionally, print the values for reference.
print("Example prediction (7 days forecast):", first_pred)
print("Ground truth (7 days):", first_truth)
