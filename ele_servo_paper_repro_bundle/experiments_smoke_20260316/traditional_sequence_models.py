from __future__ import annotations

import torch
from torch import nn

from model.lstm import LSTMClassifier
from model.transformer import TransformerClassifier


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.downsample(x)


class CNNTCNClassifier(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.tcn = nn.Sequential(
            TemporalBlock(64, 64, dilation=1),
            TemporalBlock(64, 96, dilation=2),
            TemporalBlock(96, 128, dilation=4),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.tcn(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class FCNBranch(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(x)).squeeze(-1)


class ResNetFCNTimeSeriesClassifier(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.res_stem = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.res_body = nn.Sequential(
            ResidualBlock1D(64),
            ResidualBlock1D(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock1D(128),
        )
        self.res_pool = nn.AdaptiveAvgPool1d(1)
        self.fcn = FCNBranch(channels)
        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        res = self.res_pool(self.res_body(self.res_stem(x))).squeeze(-1)
        fcn = self.fcn(x)
        fused = torch.cat([res, fcn], dim=1)
        return self.head(fused)


class ConvBiLSTMTransformerBranch(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, lstm_hidden: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        model_dim = lstm_hidden * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.bilstm(x)
        return self.transformer(x)


class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, model_dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.time_to_freq = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.freq_to_time = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.time_norm = nn.LayerNorm(model_dim)
        self.freq_norm = nn.LayerNorm(model_dim)

    def forward(self, time_tokens: torch.Tensor, freq_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        time_ctx, _ = self.time_to_freq(time_tokens, freq_tokens, freq_tokens)
        freq_ctx, _ = self.freq_to_time(freq_tokens, time_tokens, time_tokens)
        return self.time_norm(time_tokens + time_ctx), self.freq_norm(freq_tokens + freq_ctx)


class DualBranchCrossAttentionClassifier(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        hidden_dim = 96
        lstm_hidden = 96
        model_dim = lstm_hidden * 2
        heads = 4

        self.time_branch = ConvBiLSTMTransformerBranch(channels, hidden_dim, lstm_hidden, heads)
        self.freq_branch = ConvBiLSTMTransformerBranch(channels, hidden_dim, lstm_hidden, heads)
        self.cross_fusion = CrossAttentionFusionBlock(model_dim, heads)
        self.output_refine = nn.Sequential(
            nn.Conv1d(model_dim * 2, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_tokens = self.time_branch(x)

        freq_x = torch.fft.rfft(x, dim=1).abs()
        freq_tokens = self.freq_branch(freq_x)

        time_tokens, freq_tokens = self.cross_fusion(time_tokens, freq_tokens)

        freq_summary = freq_tokens.mean(dim=1, keepdim=True).expand(-1, time_tokens.size(1), -1)
        fused = torch.cat([time_tokens, freq_summary], dim=-1)
        fused = self.output_refine(fused.transpose(1, 2))
        fused = self.pool(fused).squeeze(-1)
        return self.head(fused)


def build_baseline_model(model_name: str, channels: int, num_classes: int) -> nn.Module:
    if model_name == "cnn_tcn":
        return CNNTCNClassifier(channels, num_classes)
    if model_name == "bilstm":
        return LSTMClassifier(channels, num_classes)
    if model_name == "resnet_fcn":
        return ResNetFCNTimeSeriesClassifier(channels, num_classes)
    if model_name == "transformer":
        return TransformerClassifier(channels, num_classes)
    if model_name == "dual_branch_xattn":
        return DualBranchCrossAttentionClassifier(channels, num_classes)
    raise ValueError(f"Unsupported model: {model_name}")
