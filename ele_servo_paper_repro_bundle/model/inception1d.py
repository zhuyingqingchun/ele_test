import torch
from torch import nn


class InceptionBlock1D(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(channels, out_channels, kernel_size=1)
        self.branch3 = nn.Conv1d(channels, out_channels, kernel_size=3, padding=1)
        self.branch5 = nn.Conv1d(channels, out_channels, kernel_size=5, padding=2)
        self.act = nn.ReLU()

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        out = torch.cat([b1, b3, b5], dim=1)
        return self.act(out)


class Inception1DClassifier(nn.Module):
    def __init__(self, channels, num_classes):
        super().__init__()
        self.block1 = InceptionBlock1D(channels, 32)
        self.block2 = InceptionBlock1D(96, 32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
