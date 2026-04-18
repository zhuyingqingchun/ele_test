import math
import torch
from torch import nn

from .conv_transformer_encoder import ConvTransformerEncoder, sinusoidal_positional_encoding


def prune_tokens(tokens, keep_ratio):
    if keep_ratio >= 1.0:
        return tokens
    bsz, seq_len, dim = tokens.shape
    keep_k = max(1, int(seq_len * keep_ratio))
    scores = tokens.norm(dim=2)
    topk = scores.topk(keep_k, dim=1).indices
    idx = topk.unsqueeze(-1).expand(-1, -1, dim)
    return torch.gather(tokens, 1, idx)


class PhysiTokenEncoder(nn.Module):
    def __init__(
        self,
        channels,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=2048,
        use_freq=False,
        use_phys=False,
        freq_bins=64,
        prune_ratio=1.0,
    ):
        super().__init__()
        self.use_freq = use_freq
        self.use_phys = use_phys
        self.freq_bins = freq_bins
        self.prune_ratio = prune_ratio
        self.max_len = max_len

        self.time_encoder = ConvTransformerEncoder(
            channels=channels,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
        )

        if use_freq:
            self.freq_proj = nn.Linear(channels, d_model)
        else:
            self.freq_proj = None

        if use_phys:
            self.phys_proj = nn.Linear(4, d_model)
        else:
            self.phys_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def _freq_tokens(self, x):
        # x: [B, T, C]
        freq = torch.fft.rfft(x, dim=1)
        mag = torch.abs(freq)
        if mag.size(1) > self.freq_bins:
            mag = mag[:, : self.freq_bins, :]
        return self.freq_proj(mag)

    def _phys_tokens(self, x):
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        rms = torch.sqrt((x ** 2).mean(dim=1) + 1e-12)
        ptp = x.max(dim=1).values - x.min(dim=1).values
        feats = torch.stack([mean, std, rms, ptp], dim=2)
        return self.phys_proj(feats)

    def forward_tokens(self, x):
        # time tokens already encoded
        time_tokens = self.time_encoder.forward_tokens(x)
        tokens = [time_tokens]

        if self.use_freq:
            tokens.append(self._freq_tokens(x))
        if self.use_phys:
            tokens.append(self._phys_tokens(x))

        tokens = torch.cat(tokens, dim=1)
        tokens = prune_tokens(tokens, self.prune_ratio)

        bsz, seq_len, dim = tokens.shape
        if seq_len > self.max_len:
            tokens = tokens[:, : self.max_len, :]
            seq_len = tokens.size(1)
        pos = sinusoidal_positional_encoding(seq_len + 1, dim, tokens.device)
        cls = self.cls_token.expand(bsz, -1, -1)
        xcat = torch.cat([cls, tokens], dim=1)
        xcat = xcat + pos.unsqueeze(0)
        fused = self.fusion(xcat)
        return fused[:, 1:, :]

    def forward(self, x):
        tokens = self.forward_tokens(x)
        bsz, seq_len, dim = tokens.shape
        cls = self.cls_token.expand(bsz, -1, -1)
        pos = sinusoidal_positional_encoding(seq_len + 1, dim, tokens.device)
        xcat = torch.cat([cls, tokens], dim=1)
        xcat = xcat + pos.unsqueeze(0)
        fused = self.fusion(xcat)
        return fused[:, 0, :]
