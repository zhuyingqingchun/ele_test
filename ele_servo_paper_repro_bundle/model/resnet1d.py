import torch
from torch import nn


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class ResNet1DClassifier(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer1 = nn.Sequential(ResBlock1D(64), ResBlock1D(64))
        self.layer2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResBlock1D(128),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
