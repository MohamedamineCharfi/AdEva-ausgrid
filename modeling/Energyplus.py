import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Skorch and scikit-learn imports
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

# Assume the following classes have been defined, similar to the previous code:
#  - PositionalEncoding
#  - TimeSeriesTransformer

# For brevity, we'll just use the TimeSeriesTransformer defined earlier:
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
        
class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size, d_model=64, nhead=4, num_encoder_layers=2, 
                 dim_feedforward=128, dropout=0.1, forecast_horizon=7):
        super(TimeSeriesTransformer, self).__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(feature_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.forecasting_head = nn.Linear(d_model, forecast_horizon)
        
    def forward(self, src):
        # src: (batch_size, seq_length, feature_size)
        src = self.input_linear(src)
        src = self.positional_encoding(src)
        src = src.permute(1, 0, 2)
        encoder_output = self.transformer_encoder(src)
        last_output = encoder_output[-1, :, :]
        prediction = self.forecasting_head(last_output)
        return prediction

# Data Loading & Preprocessing (similar to previous example)
df = pd.read_csv('../data/preprocessed/dataframe_Preprocessed.csv', parse_dates=['date'])
customer_id = 0
df_customer = df[df['Customer'] == customer_id].copy()
df_customer.sort_values('date', inplace=True)
df_customer.reset_index(drop=True, inplace=True)
features = ['consumption_daily_normalized', 'is_holiday_or_weekend', 'saison']
data = df_customer[features].values.astype(np.float32)

# Create sliding window sequences
def create_sequences(data, window_size, forecast_horizon):
    xs, ys = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        x = data[i : i + window_size, :]
        y = data[i + window_size : i + window_size + forecast_horizon, 0]  # target is first feature
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

window_size = 30
forecast_horizon = 7
X, y = create_sequences(data, window_size, forecast_horizon)

# Train-Test Split (using a fixed split for hyperparameter tuning)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Define the device (ensure compatibility with Skorch)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Wrap the PyTorch model using Skorch
net = NeuralNetRegressor(
    module=TimeSeriesTransformer,
    module__feature_size=len(features),
    module__forecast_horizon=forecast_horizon,
    max_epochs=20,
    lr=0.001,
    batch_size=32,
    optimizer=torch.optim.Adam,
    iterator_train__shuffle=False,
    device=device,
)

# Define a grid for hyperparameter tuning
params = {
    'module__d_model': [64, 128],
    'module__num_encoder_layers': [2, 3],
    'lr': [0.001, 0.0005],
    'batch_size': [32, 64],
}

# Set up GridSearchCV
gs = GridSearchCV(net, params, refit=True, cv=3, scoring='neg_mean_squared_error', verbose=2)
gs.fit(X_train, y_train)

print("Best parameters found:", gs.best_params_)
print("Best CV score:", -gs.best_score_)

# Evaluate on the validation set with the tuned model
y_pred = gs.predict(X_val)
mse = np.mean((y_val - y_pred)**2)
print("Validation MSE:", mse)

# Plot the first validation sample's predictions vs. actual values
first_pred = y_pred[0]
first_truth = y_val[0]

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
days = np.arange(1, forecast_horizon + 1)
plt.plot(days, first_truth, marker='o', label='Ground Truth')
plt.plot(days, first_pred, marker='x', label='Predicted')
plt.xlabel("Forecast Horizon (Days)")
plt.ylabel("Normalized Consumption")
plt.title("7-Day Forecast: Ground Truth vs Predicted (Tuned Model)")
plt.legend()
plt.grid(True)
plt.show()
