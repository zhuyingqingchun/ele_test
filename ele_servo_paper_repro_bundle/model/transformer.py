import torch
from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(self, channels, num_classes, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(channels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.2, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.proj(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)
