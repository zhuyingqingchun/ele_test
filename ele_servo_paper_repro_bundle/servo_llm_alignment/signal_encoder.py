from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from servo_diagnostic.multimodal_method import (
    ContextEncoder,
    PhysicsGraphFusion,
    SpectralStateEncoder,
    TemporalExplainableEncoder,
    ThermalStateEncoder,
    VibrationStateEncoder,
)


@dataclass
class SignalAlignmentOutputs:
    embedding: Tensor
    fused: Tensor
    modality_importance: Tensor
    edge_attention: Tensor
    physics_loss: Tensor
    alignment_loss: Tensor


class SignalAlignmentEncoder(nn.Module):
    def __init__(self, input_dims: dict[str, int], model_dim: int = 128, align_dim: int = 768) -> None:
        super().__init__()
        self.pos_encoder = TemporalExplainableEncoder(input_dims["pos"], model_dim, token_count=24, hidden_dim=48, dilations=(1, 2, 4, 8))
        self.elec_encoder = TemporalExplainableEncoder(input_dims["elec"], model_dim, token_count=24, hidden_dim=64, dilations=(1, 2, 4, 8))
        self.therm_encoder = ThermalStateEncoder(input_dims["therm"], model_dim, token_count=12, hidden_dim=32)
        self.vib_encoder = VibrationStateEncoder(input_dims["vib"], model_dim, token_count=16, hidden_dim=32)
        self.res_encoder = TemporalExplainableEncoder(input_dims["res"], model_dim, token_count=20, hidden_dim=64, dilations=(1, 2, 4, 8))
        self.freq_encoder = SpectralStateEncoder(input_dims["freq"], model_dim, token_count=12, hidden_dim=48)
        self.ctx_encoder = ContextEncoder(input_dims["ctx"], model_dim)
        self.fusion = PhysicsGraphFusion(model_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(model_dim, align_dim),
        )

    def forward(self, batch) -> SignalAlignmentOutputs:
        _, pos_node, _ = self.pos_encoder(batch.pos)
        _, elec_node, _ = self.elec_encoder(batch.elec)
        _, therm_node, _ = self.therm_encoder(batch.therm)
        _, vib_node, _ = self.vib_encoder(batch.vib)
        _, res_node, _ = self.res_encoder(batch.res)
        _, freq_node, _ = self.freq_encoder(batch.freq)
        ctx_node = self.ctx_encoder(batch.ctx)

        nodes = torch.stack([pos_node, elec_node, therm_node, vib_node, res_node, freq_node, ctx_node], dim=1)
        fused, _, modality_importance, edge_attention, physics_loss, alignment_loss = self.fusion(nodes)
        embedding = F.normalize(self.proj(fused), dim=-1)
        return SignalAlignmentOutputs(
            embedding=embedding,
            fused=fused,
            modality_importance=modality_importance,
            edge_attention=edge_attention,
            physics_loss=physics_loss,
            alignment_loss=alignment_loss,
        )
