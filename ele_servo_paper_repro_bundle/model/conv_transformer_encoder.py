import math
import torch
from torch import nn


def sinusoidal_positional_encoding(length, dim, device):
    position = torch.arange(length, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class ConvTransformerEncoder(nn.Module):
    def __init__(
        self,
        channels,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=2048,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, d_model, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.max_len = max_len

    def _encode(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        bsz, seq_len, dim = x.shape

        if seq_len > self.max_len:
            x = x[:, : self.max_len, :]
            seq_len = x.size(1)
        pos = sinusoidal_positional_encoding(seq_len, dim, x.device)
        x = x + pos.unsqueeze(0)
        return self.encoder(x)

    def forward_tokens(self, x):
        # Returns sequence tokens without CLS. Shape: [B, S, D]
        return self._encode(x)

    def forward(self, x):
        # Returns CLS-like pooled embedding. Shape: [B, D]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        bsz, seq_len, dim = x.shape

        cls_token = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        if x.size(1) > self.max_len:
            x = x[:, : self.max_len, :]
        pos = sinusoidal_positional_encoding(x.size(1), dim, x.device)
        x = x + pos.unsqueeze(0)

        x = self.encoder(x)
        return x[:, 0, :]
