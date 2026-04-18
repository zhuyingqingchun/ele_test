import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(self, channels, num_classes, hidden=128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden * 2, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)
