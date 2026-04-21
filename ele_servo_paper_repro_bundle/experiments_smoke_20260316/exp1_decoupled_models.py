from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from servo_diagnostic.multimodal_method import (
    SpectralStateEncoder,
    TemporalExplainableEncoder,
    ThermalStateEncoder,
    VibrationStateEncoder,
)


def info_nce(signal_emb: Tensor, text_emb: Tensor, temperature: float) -> Tensor:
    sim = (signal_emb @ text_emb.t()) / temperature
    targets = torch.arange(sim.shape[0], device=sim.device)
    return 0.5 * (F.cross_entropy(sim, targets) + F.cross_entropy(sim.t(), targets))


@dataclass
class StageOutputs:
    logits: Tensor
    align_loss: Tensor | None
    quality_loss: Tensor | None = None
    quality_gates: Tensor | None = None
    branch_energies: Tensor | None = None


class SequencePooling(nn.Module):
    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        if mode not in {"cls", "mean", "attn", "text"}:
            raise ValueError(f"Unsupported pooling mode: {mode}")
        self.mode = mode
        self.score = nn.Linear(dim, 1) if mode == "attn" else None

    def forward(self, seq: Tensor, *, text_index: int | None = None) -> Tensor:
        if self.mode == "cls":
            return seq[:, 0, :]
        if self.mode == "text":
            if text_index is None:
                raise ValueError("text_index is required for text pooling")
            return seq[:, text_index, :]
        if self.mode == "mean":
            return seq.mean(dim=1)
        weights = torch.softmax(self.score(seq).squeeze(-1), dim=1)
        return torch.einsum("bs,bsd->bd", weights, seq)


class ModalityQualityEstimator(nn.Module):
    def __init__(self, token_dim: int, summary_dim: int = 4, hidden_dim: int = 128, min_gate: float = 0.10) -> None:
        super().__init__()
        self.min_gate = float(min_gate)
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim + summary_dim),
            nn.Linear(token_dim + summary_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pooled_tokens: Tensor, summaries: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        logits = self.net(torch.cat([pooled_tokens, summaries], dim=-1)).squeeze(-1)
        scores = torch.sigmoid(logits)
        gates = self.min_gate + (1.0 - self.min_gate) * scores
        loss = F.binary_cross_entropy_with_logits(logits, targets) if targets is not None else None
        return gates, loss


class DecoupledSignalTokenBackbone(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        model_dim: int,
        token_dim: int,
        *,
        quality_aware_fusion: bool = False,
        quality_hidden_dim: int = 128,
        modality_drop_prob: float = 0.0,
        quality_min_gate: float = 0.10,
    ) -> None:
        super().__init__()
        self.quality_aware_fusion = bool(quality_aware_fusion)
        self.modality_drop_prob = float(modality_drop_prob)
        self.pos_encoder = TemporalExplainableEncoder(input_dims["pos"], model_dim, token_count=20, hidden_dim=48, dilations=(1, 2, 4, 8))
        self.electrical_encoder = TemporalExplainableEncoder(input_dims["electrical"], model_dim, token_count=24, hidden_dim=56, dilations=(1, 2, 4, 8))
        self.thermal_encoder = ThermalStateEncoder(input_dims["thermal"], model_dim, token_count=8, hidden_dim=32)
        self.vibration_encoder = VibrationStateEncoder(input_dims["vibration"], model_dim, token_count=16, hidden_dim=64)
        self.proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, token_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.quality_estimator = (
            ModalityQualityEstimator(token_dim=token_dim, hidden_dim=quality_hidden_dim, min_gate=quality_min_gate)
            if self.quality_aware_fusion
            else None
        )

    def _corrupt_modality(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if (not self.training) or self.modality_drop_prob <= 0.0:
            targets = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            return x, targets
        drop_mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < self.modality_drop_prob).to(x.dtype)
        clean_target = 1.0 - drop_mask.squeeze(-1).squeeze(-1)
        return x * (1.0 - drop_mask), clean_target

    def _summary_stats(self, x: Tensor) -> Tensor:
        delta = torch.diff(x, dim=1, prepend=x[:, :1, :])
        mean_abs = x.abs().mean(dim=(1, 2))
        std = x.std(dim=(1, 2), unbiased=False)
        delta_rms = torch.sqrt(torch.clamp(delta.pow(2).mean(dim=(1, 2)), min=1.0e-8))
        near_zero = (x.abs() < 1.0e-6).float().mean(dim=(1, 2))
        return torch.stack([mean_abs, std, delta_rms, near_zero], dim=-1)

    def _encode_project(self, encoder: nn.Module, x: Tensor) -> Tensor:
        tokens, _, _ = encoder(x)
        return self.proj(tokens)

    def forward(self, batch) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None, Tensor]:
        pos_in, pos_target = self._corrupt_modality(batch.pos)
        electrical_in, electrical_target = self._corrupt_modality(batch.electrical)
        thermal_in, thermal_target = self._corrupt_modality(batch.thermal)
        vibration_in, vibration_target = self._corrupt_modality(batch.vibration)

        pos_tokens = self._encode_project(self.pos_encoder, pos_in)
        electrical_tokens = self._encode_project(self.electrical_encoder, electrical_in)
        thermal_tokens = self._encode_project(self.thermal_encoder, thermal_in)
        vibration_tokens = self._encode_project(self.vibration_encoder, vibration_in)

        quality_loss = None
        gates = None
        if self.quality_estimator is not None:
            pooled_stack = torch.stack(
                [
                    pos_tokens.mean(dim=1),
                    electrical_tokens.mean(dim=1),
                    thermal_tokens.mean(dim=1),
                    vibration_tokens.mean(dim=1),
                ],
                dim=1,
            )
            summary_stack = torch.stack(
                [
                    self._summary_stats(pos_in),
                    self._summary_stats(electrical_in),
                    self._summary_stats(thermal_in),
                    self._summary_stats(vibration_in),
                ],
                dim=1,
            )
            target_stack = torch.stack([pos_target, electrical_target, thermal_target, vibration_target], dim=1)
            gates, quality_loss = self.quality_estimator(pooled_stack, summary_stack, targets=target_stack)
            pos_tokens = pos_tokens * gates[:, 0].view(-1, 1, 1)
            electrical_tokens = electrical_tokens * gates[:, 1].view(-1, 1, 1)
            thermal_tokens = thermal_tokens * gates[:, 2].view(-1, 1, 1)
            vibration_tokens = vibration_tokens * gates[:, 3].view(-1, 1, 1)

        branch_energies = torch.stack(
            [
                pos_tokens.pow(2).mean(dim=(1, 2)),
                electrical_tokens.pow(2).mean(dim=(1, 2)),
                thermal_tokens.pow(2).mean(dim=(1, 2)),
                vibration_tokens.pow(2).mean(dim=(1, 2)),
            ],
            dim=1,
        )

        tokens = torch.cat(
            [pos_tokens, electrical_tokens, thermal_tokens, vibration_tokens],
            dim=1,
        )
        pooled = F.normalize(tokens.mean(dim=1), dim=-1)
        return tokens, pooled, quality_loss, gates, branch_energies


class Stage1DecoupledClassifier(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        model_dim: int = 128,
        token_dim: int = 256,
        *,
        quality_aware_fusion: bool = False,
        quality_hidden_dim: int = 128,
        modality_drop_prob: float = 0.0,
        quality_min_gate: float = 0.10,
    ) -> None:
        super().__init__()
        self.backbone = DecoupledSignalTokenBackbone(
            input_dims,
            model_dim=model_dim,
            token_dim=token_dim,
            quality_aware_fusion=quality_aware_fusion,
            quality_hidden_dim=quality_hidden_dim,
            modality_drop_prob=modality_drop_prob,
            quality_min_gate=quality_min_gate,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, batch) -> StageOutputs:
        _tokens, pooled, quality_loss, quality_gates, branch_energies = self.backbone(batch)
        return StageOutputs(
            logits=self.head(pooled),
            align_loss=None,
            quality_loss=quality_loss,
            quality_gates=quality_gates,
            branch_energies=branch_energies,
        )


class Stage2DecoupledClassifier(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        model_dim: int = 128,
        token_dim: int = 256,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 768,
        dropout: float = 0.10,
        pool: str = "cls",
        quality_aware_fusion: bool = False,
        quality_hidden_dim: int = 128,
        modality_drop_prob: float = 0.0,
        quality_min_gate: float = 0.10,
    ) -> None:
        super().__init__()
        self.backbone = DecoupledSignalTokenBackbone(
            input_dims,
            model_dim=model_dim,
            token_dim=token_dim,
            quality_aware_fusion=quality_aware_fusion,
            quality_hidden_dim=quality_hidden_dim,
            modality_drop_prob=modality_drop_prob,
            quality_min_gate=quality_min_gate,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.norm = nn.LayerNorm(token_dim)
        self.pool = SequencePooling(token_dim, pool)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, batch) -> StageOutputs:
        tokens, _pooled, quality_loss, quality_gates, branch_energies = self.backbone(batch)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = self.encoder(seq)
        seq = self.norm(seq)
        return StageOutputs(
            logits=self.head(self.pool(seq)),
            align_loss=None,
            quality_loss=quality_loss,
            quality_gates=quality_gates,
            branch_energies=branch_energies,
        )


class Stage3DecoupledClassifier(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        text_dim: int,
        model_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.10,
        temperature: float = 0.07,
        quality_aware_fusion: bool = False,
        quality_hidden_dim: int = 128,
        modality_drop_prob: float = 0.0,
        quality_min_gate: float = 0.10,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.backbone = DecoupledSignalTokenBackbone(
            input_dims,
            model_dim=model_dim,
            token_dim=token_dim,
            quality_aware_fusion=quality_aware_fusion,
            quality_hidden_dim=quality_hidden_dim,
            modality_drop_prob=modality_drop_prob,
            quality_min_gate=quality_min_gate,
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, batch, text_embeddings: Tensor) -> StageOutputs:
        _tokens, pooled, quality_loss, quality_gates, branch_energies = self.backbone(batch)
        logits = self.head(pooled)
        text_vec = F.normalize(self.text_proj(text_embeddings), dim=-1)
        return StageOutputs(
            logits=logits,
            align_loss=info_nce(pooled, text_vec, temperature=self.temperature),
            quality_loss=quality_loss,
            quality_gates=quality_gates,
            branch_energies=branch_energies,
        )


class Stage4DecoupledClassifier(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        text_dim: int,
        model_dim: int = 128,
        token_dim: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 768,
        dropout: float = 0.10,
        temperature: float = 0.07,
        pool: str = "cls",
        quality_aware_fusion: bool = False,
        quality_hidden_dim: int = 128,
        modality_drop_prob: float = 0.0,
        quality_min_gate: float = 0.10,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.backbone = DecoupledSignalTokenBackbone(
            input_dims,
            model_dim=model_dim,
            token_dim=token_dim,
            quality_aware_fusion=quality_aware_fusion,
            quality_hidden_dim=quality_hidden_dim,
            modality_drop_prob=modality_drop_prob,
            quality_min_gate=quality_min_gate,
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.norm = nn.LayerNorm(token_dim)
        self.pool = SequencePooling(token_dim, pool)
        self.head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, batch, text_embeddings: Tensor) -> StageOutputs:
        signal_tokens, pooled, quality_loss, quality_gates, branch_energies = self.backbone(batch)
        text_token = F.normalize(self.text_proj(text_embeddings), dim=-1).unsqueeze(1)
        cls = self.cls.expand(signal_tokens.shape[0], -1, -1)
        seq = torch.cat([cls, signal_tokens, text_token], dim=1)
        seq = self.encoder(seq)
        seq = self.norm(seq)
        pooled_seq = self.pool(seq, text_index=seq.shape[1] - 1)
        logits = self.head(pooled_seq)
        align_loss = info_nce(pooled, text_token.squeeze(1), temperature=self.temperature)
        return StageOutputs(
            logits=logits,
            align_loss=align_loss,
            quality_loss=quality_loss,
            quality_gates=quality_gates,
            branch_energies=branch_energies,
        )
