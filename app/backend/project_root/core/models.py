# api/forecast_model.py
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_size, d_model=64, nhead=4, num_encoder_layers=2,
                 dim_feedforward=128, dropout=0.1, forecast_horizon=7):
        super().__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(feature_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_encoder_layers)
        self.forecasting_head = nn.Linear(d_model, forecast_horizon)

    def forward(self, src):
        # src: (batch, seq_len, feature_size)
        src = self.input_linear(src)
        src = self.positional_encoding(src)
        src = src.permute(1, 0, 2)               # (seq_len, batch, d_model)
        enc_out = self.transformer_encoder(src)
        last_output = enc_out[-1]                # (batch, d_model)
        return self.forecasting_head(last_output)