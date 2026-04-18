import torch
from torch import nn


class TCNClassifier(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x).squeeze(-1)
        return self.fc(x)
