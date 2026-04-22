from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from experiments_smoke_20260316.exp1_decoupled_models import DecoupledSignalTokenBackbone, SequencePooling
from experiments_smoke_20260316.exp1_topk_duallevel_align_v2_models import TextGuidedFaultGate
from experiments_smoke_20260316.token_evidence_alignment_v1_utils import TOKEN_GROUP_LENGTHS, token_transport_plan


@dataclass
class Stage4TokenEvidenceOutputs:
    logits: Tensor
    signal_embedding: Tensor
    signal_tokens: Tensor
    modality_scores: Tensor
    modality_embeddings: Tensor
    combined_text_embedding: Tensor
    evidence_text_embedding: Tensor
    mechanism_text_embedding: Tensor
    contrast_text_embedding: Tensor
    token_plan: Tensor
    quality_loss: Tensor | None = None
    quality_gates: Tensor | None = None
    branch_energies: Tensor | None = None


class Stage4TokenEvidenceAlignmentV1(nn.Module):
    lengths = TOKEN_GROUP_LENGTHS

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        text_dim: int,
        *,
        model_dim: int = 128,
        token_dim: int = 256,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 768,
        pool: str = "attn",
        quality_aware_fusion: bool = True,
        quality_hidden_dim: int = 128,
        modality_drop_prob: float = 0.10,
        quality_min_gate: float = 0.10,
        fault_gate_hidden_dim: int = 128,
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
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, token_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.fault_gate = TextGuidedFaultGate(token_dim, hidden_dim=fault_gate_hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.10,
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

    def _split_modality_embeddings(self, tokens: Tensor) -> Tensor:
        outs = []
        start = 0
        for ln in self.lengths:
            outs.append(tokens[:, start:start + ln, :].mean(dim=1))
            start += ln
        return torch.stack(outs, dim=1)

    def forward(self, batch, text_views: dict[str, Tensor]) -> Stage4TokenEvidenceOutputs:
        backbone_out = self.backbone(batch)
        if isinstance(backbone_out, tuple) and len(backbone_out) == 6:
            signal_tokens, pooled, quality_loss, quality_gates, branch_energies, _ = backbone_out
        else:
            signal_tokens, pooled, quality_loss, quality_gates, branch_energies = backbone_out
        modality_embeddings = self._split_modality_embeddings(signal_tokens)

        combined = F.normalize(self.text_proj(text_views["combined"]), dim=-1)
        evidence = F.normalize(self.text_proj(text_views["evidence"]), dim=-1)
        mechanism = F.normalize(self.text_proj(text_views["mechanism"]), dim=-1)
        contrast = F.normalize(self.text_proj(text_views["contrast"]), dim=-1)

        fault_gates = self.fault_gate(modality_embeddings, pooled, evidence, mechanism)

        repeats = torch.tensor(self.lengths, device=signal_tokens.device)
        gated_tokens = signal_tokens * torch.repeat_interleave(fault_gates, repeats, dim=1).unsqueeze(-1)

        token_plan = token_transport_plan(gated_tokens, evidence, mechanism, temperature=0.07)

        cls = self.cls.expand(gated_tokens.shape[0], -1, -1)
        seq = torch.cat(
            [cls, gated_tokens, evidence.unsqueeze(1), mechanism.unsqueeze(1), combined.unsqueeze(1)],
            dim=1,
        )
        seq = self.norm(self.encoder(seq))
        pooled_seq = self.pool(seq)
        logits = self.head(pooled_seq)

        return Stage4TokenEvidenceOutputs(
            logits=logits,
            signal_embedding=F.normalize(pooled_seq, dim=-1),
            signal_tokens=gated_tokens,
            modality_scores=fault_gates,
            modality_embeddings=modality_embeddings,
            combined_text_embedding=combined,
            evidence_text_embedding=evidence,
            mechanism_text_embedding=mechanism,
            contrast_text_embedding=contrast,
            token_plan=token_plan,
            quality_loss=quality_loss,
            quality_gates=quality_gates,
            branch_energies=branch_energies,
        )
