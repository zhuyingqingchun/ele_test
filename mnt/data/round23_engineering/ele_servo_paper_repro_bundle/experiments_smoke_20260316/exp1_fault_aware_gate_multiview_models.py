from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from servo_diagnostic.multimodal_method import TemporalExplainableEncoder, ThermalStateEncoder, VibrationStateEncoder

from experiments_smoke_20260316.modality_evidence_priors import BRANCH_NAMES


def info_nce(signal_emb: Tensor, text_emb: Tensor, temperature: float) -> Tensor:
    sim = (signal_emb @ text_emb.t()) / temperature
    targets = torch.arange(sim.shape[0], device=sim.device)
    return 0.5 * (F.cross_entropy(sim, targets) + F.cross_entropy(sim.t(), targets))


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


class FaultAwareTextGuidedGate(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int = 128, min_gate: float = 0.15) -> None:
        super().__init__()
        self.min_gate = float(min_gate)
        self.net = nn.Sequential(
            nn.LayerNorm(token_dim * 4),
            nn.Linear(token_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, modality_pooled: Tensor, pooled_signal: Tensor, evidence_vec: Tensor, mechanism_vec: Tensor) -> Tensor:
        pooled_signal = pooled_signal.unsqueeze(1).expand_as(modality_pooled)
        evidence_vec = evidence_vec.unsqueeze(1).expand_as(modality_pooled)
        mechanism_vec = mechanism_vec.unsqueeze(1).expand_as(modality_pooled)
        feat = torch.cat([modality_pooled, pooled_signal, evidence_vec, mechanism_vec], dim=-1)
        logits = self.net(feat).squeeze(-1)
        scores = torch.softmax(logits, dim=1)
        return self.min_gate + (1.0 - self.min_gate) * scores


class PrototypeBank(nn.Module):
    def __init__(self, num_classes: int, dim: int) -> None:
        super().__init__()
        self.evidence_proto = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.mechanism_proto = nn.Parameter(torch.randn(num_classes, dim) * 0.02)

    def normalized(self) -> tuple[Tensor, Tensor]:
        return F.normalize(self.evidence_proto, dim=-1), F.normalize(self.mechanism_proto, dim=-1)

    def prototype_losses(self, signal_vec: Tensor, labels: Tensor, margin: float = 0.15) -> tuple[Tensor, Tensor]:
        evidence_proto, mechanism_proto = self.normalized()
        signal_vec = F.normalize(signal_vec, dim=-1)
        logits_e = signal_vec @ evidence_proto.t()
        logits_m = signal_vec @ mechanism_proto.t()
        ce_loss = 0.5 * (F.cross_entropy(logits_e, labels) + F.cross_entropy(logits_m, labels))
        pos_e = logits_e.gather(1, labels.unsqueeze(1)).squeeze(1)
        pos_m = logits_m.gather(1, labels.unsqueeze(1)).squeeze(1)
        mask = F.one_hot(labels, num_classes=evidence_proto.shape[0]).bool()
        neg_e = logits_e.masked_fill(mask, -1.0e9).max(dim=1).values
        neg_m = logits_m.masked_fill(mask, -1.0e9).max(dim=1).values
        hard_negative = 0.5 * (
            torch.relu(margin - (pos_e - neg_e)).mean() + torch.relu(margin - (pos_m - neg_m)).mean()
        )
        return ce_loss, hard_negative


class DecoupledSignalTokenBackboneV2(nn.Module):
    branch_names = list(BRANCH_NAMES)

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
        self.quality_estimator = ModalityQualityEstimator(token_dim=token_dim, hidden_dim=quality_hidden_dim, min_gate=quality_min_gate) if self.quality_aware_fusion else None

    def _corrupt_modality(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if (not self.training) or self.modality_drop_prob <= 0.0:
            return x, torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        drop_mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < self.modality_drop_prob).to(x.dtype)
        atten_mask = (torch.rand(x.shape[0], 1, 1, device=x.device) < self.modality_drop_prob).to(x.dtype)
        out = x * (1.0 - drop_mask) * (1.0 - 0.75 * atten_mask)
        target = (1.0 - drop_mask.squeeze(-1).squeeze(-1) - 0.7 * atten_mask.squeeze(-1).squeeze(-1)).clamp(0.0, 1.0)
        return out, target

    def _summary_stats(self, x: Tensor) -> Tensor:
        delta = torch.diff(x, dim=1, prepend=x[:, :1, :])
        mean_abs = x.abs().mean(dim=(1, 2))
        std = x.std(dim=(1, 2), unbiased=False)
        delta_rms = torch.sqrt(torch.clamp(delta.pow(2).mean(dim=(1, 2)), min=1.0e-8))
        near_zero = (x.abs() < 1.0e-6).float().mean(dim=(1, 2))
        return torch.stack([mean_abs, std, delta_rms, near_zero], dim=-1)

    def _encode_project(self, encoder: nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
        tokens, _, _ = encoder(x)
        tokens = self.proj(tokens)
        pooled = F.normalize(tokens.mean(dim=1), dim=-1)
        return tokens, pooled

    def forward(self, batch):
        pos_in, pos_t = self._corrupt_modality(batch.pos)
        ele_in, ele_t = self._corrupt_modality(batch.electrical)
        thm_in, thm_t = self._corrupt_modality(batch.thermal)
        vib_in, vib_t = self._corrupt_modality(batch.vibration)

        pos_tokens, pos_pooled = self._encode_project(self.pos_encoder, pos_in)
        ele_tokens, ele_pooled = self._encode_project(self.electrical_encoder, ele_in)
        thm_tokens, thm_pooled = self._encode_project(self.thermal_encoder, thm_in)
        vib_tokens, vib_pooled = self._encode_project(self.vibration_encoder, vib_in)

        modality_pooled = torch.stack([pos_pooled, ele_pooled, thm_pooled, vib_pooled], dim=1)
        quality_loss = None
        quality_gates = None
        if self.quality_estimator is not None:
            summary = torch.stack([
                self._summary_stats(pos_in), self._summary_stats(ele_in), self._summary_stats(thm_in), self._summary_stats(vib_in)
            ], dim=1)
            targets = torch.stack([pos_t, ele_t, thm_t, vib_t], dim=1)
            quality_gates, quality_loss = self.quality_estimator(modality_pooled, summary, targets=targets)
            pos_tokens = pos_tokens * quality_gates[:, 0].view(-1, 1, 1)
            ele_tokens = ele_tokens * quality_gates[:, 1].view(-1, 1, 1)
            thm_tokens = thm_tokens * quality_gates[:, 2].view(-1, 1, 1)
            vib_tokens = vib_tokens * quality_gates[:, 3].view(-1, 1, 1)
            modality_pooled = torch.stack([
                F.normalize(pos_tokens.mean(dim=1), dim=-1),
                F.normalize(ele_tokens.mean(dim=1), dim=-1),
                F.normalize(thm_tokens.mean(dim=1), dim=-1),
                F.normalize(vib_tokens.mean(dim=1), dim=-1),
            ], dim=1)
        branch_energies = torch.stack([
            pos_tokens.pow(2).mean(dim=(1, 2)), ele_tokens.pow(2).mean(dim=(1, 2)), thm_tokens.pow(2).mean(dim=(1, 2)), vib_tokens.pow(2).mean(dim=(1, 2))
        ], dim=1)
        tokens = torch.cat([pos_tokens, ele_tokens, thm_tokens, vib_tokens], dim=1)
        pooled = F.normalize(tokens.mean(dim=1), dim=-1)
        token_slices = {
            "position": (0, pos_tokens.shape[1]),
            "electrical": (pos_tokens.shape[1], pos_tokens.shape[1] + ele_tokens.shape[1]),
            "thermal": (pos_tokens.shape[1] + ele_tokens.shape[1], pos_tokens.shape[1] + ele_tokens.shape[1] + thm_tokens.shape[1]),
            "vibration": (pos_tokens.shape[1] + ele_tokens.shape[1] + thm_tokens.shape[1], tokens.shape[1]),
        }
        return tokens, pooled, quality_loss, quality_gates, branch_energies, modality_pooled, token_slices


@dataclass
class FaultAwareStageOutputs:
    logits: Tensor
    align_loss: Tensor | None
    quality_loss: Tensor | None = None
    quality_gates: Tensor | None = None
    branch_energies: Tensor | None = None
    modality_gates: Tensor | None = None
    evidence_scores: Tensor | None = None
    mechanism_scores: Tensor | None = None
    contrast_scores: Tensor | None = None
    prototype_loss: Tensor | None = None
    hard_negative_loss: Tensor | None = None


class _CommonMultiView(nn.Module):
    def __init__(self, num_classes: int, text_dim: int, token_dim: int, fault_gate_hidden_dim: int, fault_gate_min: float, dropout: float) -> None:
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fault_gate = FaultAwareTextGuidedGate(token_dim=token_dim, hidden_dim=fault_gate_hidden_dim, min_gate=fault_gate_min)
        self.prototypes = PrototypeBank(num_classes=num_classes, dim=token_dim)
        self.branch_names = list(BRANCH_NAMES)

    def _project_views(self, text_views: dict[str, Tensor]) -> dict[str, Tensor]:
        return {k: F.normalize(self.text_proj(v), dim=-1) for k, v in text_views.items()}

    def _alignment_losses(self, pooled_signal: Tensor, text_vecs: dict[str, Tensor], evidence_scores: Tensor, mechanism_scores: Tensor, primary_targets: Tensor | None, support_targets: Tensor | None, labels: Tensor | None, temperature: float, weights: dict[str, float]) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor]:
        contrast_scores = torch.einsum("bd,bd->b", pooled_signal, text_vecs["contrast"])
        total = (
            weights.get("lambda_combined", 1.0) * info_nce(pooled_signal, text_vecs["combined"], temperature)
            + weights.get("lambda_evidence", 1.0) * info_nce(pooled_signal, text_vecs["evidence"], temperature)
            + weights.get("lambda_mechanism", 1.0) * info_nce(pooled_signal, text_vecs["mechanism"], temperature)
            + weights.get("lambda_contrast", 0.35) * torch.relu(contrast_scores - 0.15).mean()
        )
        if primary_targets is not None:
            total = total + weights.get("lambda_modality_align", 0.20) * F.cross_entropy(evidence_scores, primary_targets)
        if support_targets is not None:
            total = total + weights.get("lambda_modality_align", 0.20) * F.binary_cross_entropy_with_logits(mechanism_scores, support_targets.to(mechanism_scores.dtype))
        proto_loss = None
        hard_loss = None
        if labels is not None:
            proto_loss, hard_loss = self.prototypes.prototype_losses(pooled_signal, labels, margin=weights.get("hard_negative_margin", 0.15))
            total = total + weights.get("lambda_proto", 0.10) * proto_loss + weights.get("lambda_hard_negative", 0.10) * hard_loss
        return total, proto_loss, hard_loss, contrast_scores


class Stage3FaultAwareMultiViewClassifier(_CommonMultiView):
    def __init__(self, input_dims: dict[str, int], num_classes: int, text_dim: int, model_dim: int = 128, token_dim: int = 256, dropout: float = 0.10, temperature: float = 0.07, quality_aware_fusion: bool = False, quality_hidden_dim: int = 128, modality_drop_prob: float = 0.0, quality_min_gate: float = 0.10, fault_gate_hidden_dim: int = 128, fault_gate_min: float = 0.15) -> None:
        super().__init__(num_classes, text_dim, token_dim, fault_gate_hidden_dim, fault_gate_min, dropout)
        self.temperature = float(temperature)
        self.backbone = DecoupledSignalTokenBackboneV2(input_dims, model_dim=model_dim, token_dim=token_dim, quality_aware_fusion=quality_aware_fusion, quality_hidden_dim=quality_hidden_dim, modality_drop_prob=modality_drop_prob, quality_min_gate=quality_min_gate)
        self.head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(token_dim, num_classes))

    def forward(self, batch, text_views: dict[str, Tensor], labels: Tensor | None = None, primary_targets: Tensor | None = None, support_targets: Tensor | None = None, weights: dict[str, float] | None = None) -> FaultAwareStageOutputs:
        weights = weights or {}
        _, pooled, quality_loss, quality_gates, branch_energies, modality_pooled, _ = self.backbone(batch)
        text_vecs = self._project_views(text_views)
        modality_gates = self.fault_gate(modality_pooled, pooled, text_vecs["evidence"], text_vecs["mechanism"])
        gated_modality = modality_pooled * modality_gates.unsqueeze(-1)
        pooled_signal = F.normalize(gated_modality.sum(dim=1), dim=-1)
        logits = self.head(pooled_signal)
        evidence_scores = torch.einsum("bmd,bd->bm", F.normalize(gated_modality, dim=-1), text_vecs["evidence"])
        mechanism_scores = torch.einsum("bmd,bd->bm", F.normalize(gated_modality, dim=-1), text_vecs["mechanism"])
        align_total, proto_loss, hard_loss, contrast_scores = self._alignment_losses(pooled_signal, text_vecs, evidence_scores, mechanism_scores, primary_targets, support_targets, labels, self.temperature, weights)
        return FaultAwareStageOutputs(logits, align_total, quality_loss, quality_gates, branch_energies, modality_gates, evidence_scores, mechanism_scores, contrast_scores, proto_loss, hard_loss)


class Stage4FaultAwareMultiViewClassifier(_CommonMultiView):
    def __init__(self, input_dims: dict[str, int], num_classes: int, text_dim: int, model_dim: int = 128, token_dim: int = 256, num_layers: int = 4, nhead: int = 8, dim_feedforward: int = 768, dropout: float = 0.10, temperature: float = 0.07, pool: str = "text", quality_aware_fusion: bool = False, quality_hidden_dim: int = 128, modality_drop_prob: float = 0.0, quality_min_gate: float = 0.10, fault_gate_hidden_dim: int = 128, fault_gate_min: float = 0.15) -> None:
        super().__init__(num_classes, text_dim, token_dim, fault_gate_hidden_dim, fault_gate_min, dropout)
        self.temperature = float(temperature)
        self.backbone = DecoupledSignalTokenBackboneV2(input_dims, model_dim=model_dim, token_dim=token_dim, quality_aware_fusion=quality_aware_fusion, quality_hidden_dim=quality_hidden_dim, modality_drop_prob=modality_drop_prob, quality_min_gate=quality_min_gate)
        layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.norm = nn.LayerNorm(token_dim)
        self.pool_mode = pool
        self.head = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(token_dim, num_classes))

    def _apply_modality_gates(self, signal_tokens: Tensor, token_slices: dict[str, tuple[int, int]], modality_gates: Tensor) -> Tensor:
        out = signal_tokens.clone()
        for idx, name in enumerate(self.branch_names):
            start, end = token_slices[name]
            out[:, start:end, :] = out[:, start:end, :] * modality_gates[:, idx].view(-1, 1, 1)
        return out

    def forward(self, batch, text_views: dict[str, Tensor], labels: Tensor | None = None, primary_targets: Tensor | None = None, support_targets: Tensor | None = None, weights: dict[str, float] | None = None) -> FaultAwareStageOutputs:
        weights = weights or {}
        signal_tokens, pooled, quality_loss, quality_gates, branch_energies, modality_pooled, token_slices = self.backbone(batch)
        text_vecs = self._project_views(text_views)
        modality_gates = self.fault_gate(modality_pooled, pooled, text_vecs["evidence"], text_vecs["mechanism"])
        signal_tokens = self._apply_modality_gates(signal_tokens, token_slices, modality_gates)
        gated_modality = modality_pooled * modality_gates.unsqueeze(-1)
        pooled_signal = F.normalize(gated_modality.sum(dim=1), dim=-1)
        cls = self.cls.expand(signal_tokens.shape[0], -1, -1)
        seq = torch.cat([cls, signal_tokens, text_vecs["evidence"].unsqueeze(1), text_vecs["mechanism"].unsqueeze(1), text_vecs["contrast"].unsqueeze(1)], dim=1)
        seq = self.norm(self.encoder(seq))
        pooled_seq = seq[:, -3, :] if self.pool_mode == "text" else seq[:, 0, :]
        logits = self.head(pooled_seq)
        evidence_scores = torch.einsum("bmd,bd->bm", F.normalize(gated_modality, dim=-1), text_vecs["evidence"])
        mechanism_scores = torch.einsum("bmd,bd->bm", F.normalize(gated_modality, dim=-1), text_vecs["mechanism"])
        align_total, proto_loss, hard_loss, contrast_scores = self._alignment_losses(pooled_signal, text_vecs, evidence_scores, mechanism_scores, primary_targets, support_targets, labels, self.temperature, weights)
        return FaultAwareStageOutputs(logits, align_total, quality_loss, quality_gates, branch_energies, modality_gates, evidence_scores, mechanism_scores, contrast_scores, proto_loss, hard_loss)
