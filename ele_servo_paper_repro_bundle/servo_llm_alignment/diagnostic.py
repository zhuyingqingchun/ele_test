from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .signal_encoder import SignalAlignmentEncoder


@dataclass
class DiagnosticOutputs:
    scenario_logits: Tensor
    family_logits: Tensor | None
    location_logits: Tensor | None
    prototype_logits: Tensor
    fused_embedding: Tensor
    aligned_embedding: Tensor
    semantic_token: Tensor | None = None
    semantic_gate: Tensor | None = None
    semantic_attention: Tensor | None = None
    semantic_fused_token: Tensor | None = None


@dataclass
class DiagnosticPrototypeBank:
    scenario_embeddings: Tensor
    family_embeddings: Tensor | None = None
    location_embeddings: Tensor | None = None


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int | None = None, dropout: float = 0.10) -> None:
        super().__init__()
        out_dim = output_dim or input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.gate = nn.Linear(input_dim, out_dim)
        self.residual = nn.Identity() if input_dim == out_dim else nn.Linear(input_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.residual(x)
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return residual + torch.sigmoid(self.gate(x)) * h


class FeatureRecalibration(nn.Module):
    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(dim // reduction, 32)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.net(x)


class PrototypeContextAttention(nn.Module):
    def __init__(self, query_dim: int, prototype_dim: int, token_dim: int, num_heads: int = 4, top_k: int = 6) -> None:
        super().__init__()
        self.top_k = top_k
        self.query_proj = nn.Linear(query_dim, token_dim)
        self.prototype_proj = nn.Linear(prototype_dim, token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, batch_first=True, dropout=0.10)

    def forward(self, query: Tensor, prototype_bank: Tensor, prototype_logits: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = query.shape[0]
        k = min(self.top_k, prototype_bank.shape[0])
        topk_idx = prototype_logits.topk(k=k, dim=1).indices
        projected_bank = self.prototype_proj(prototype_bank)
        selected = projected_bank[topk_idx]
        attn_query = self.query_proj(query).unsqueeze(1)
        context, attn_weights = self.attn(attn_query, selected, selected)
        return context.squeeze(1), attn_weights.squeeze(1)


class SemanticPrototypePooling(nn.Module):
    def __init__(self, query_dim: int, prototype_dim: int, token_dim: int) -> None:
        super().__init__()
        self.query_proj = nn.Linear(query_dim, token_dim)
        self.prototype_proj = nn.Linear(prototype_dim, token_dim)

    def forward(self, query: Tensor, prototype_bank: Tensor, logits: Tensor) -> Tensor:
        projected_query = self.query_proj(query)
        projected_prototypes = self.prototype_proj(prototype_bank)
        weights = torch.softmax(logits, dim=1)
        pooled = weights @ projected_prototypes
        return pooled + projected_query


def resolve_diagnostic_prototype_bank(prototype_bank: Any) -> DiagnosticPrototypeBank:
    if isinstance(prototype_bank, DiagnosticPrototypeBank):
        return prototype_bank
    if isinstance(prototype_bank, Tensor):
        return DiagnosticPrototypeBank(scenario_embeddings=prototype_bank)
    if hasattr(prototype_bank, "scenario_embeddings"):
        return DiagnosticPrototypeBank(
            scenario_embeddings=prototype_bank.scenario_embeddings,
            family_embeddings=getattr(prototype_bank, "family_embeddings", None),
            location_embeddings=getattr(prototype_bank, "location_embeddings", None),
        )
    raise TypeError(f"Unsupported prototype bank type: {type(prototype_bank)!r}")


class LegacyStage1DiagnosticModel(nn.Module):
    def __init__(
        self,
        encoder: SignalAlignmentEncoder,
        fused_dim: int,
        align_dim: int,
        num_scenarios: int,
        num_families: int,
        num_locations: int,
        freeze_encoder: bool = True,
        use_family_aux: bool = True,
        use_location_aux: bool = True,
        use_prototype_fusion: bool = True,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.use_family_aux = use_family_aux
        self.use_location_aux = use_location_aux
        self.use_prototype_fusion = use_prototype_fusion

        self.scenario_head = nn.Sequential(
            nn.LayerNorm(fused_dim + align_dim),
            nn.Linear(fused_dim + align_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_scenarios),
        )
        self.family_head = None
        self.location_head = None
        if use_family_aux:
            self.family_head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim // 2, num_families),
            )
        if use_location_aux:
            self.location_head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim // 2, num_locations),
            )
        self.prototype_scale = nn.Parameter(torch.tensor(0.75))

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, batch, prototype_bank) -> DiagnosticOutputs:
        if self.freeze_encoder:
            with torch.no_grad():
                enc = self.encoder(batch)
        else:
            enc = self.encoder(batch)
        resolved_bank = resolve_diagnostic_prototype_bank(prototype_bank)
        fused = enc.fused
        aligned = enc.embedding
        features = torch.cat([fused, aligned], dim=-1)
        scenario_logits = self.scenario_head(features)
        scenario_prototypes = resolved_bank.scenario_embeddings.to(aligned.device)
        prototype_logits = aligned @ scenario_prototypes.T
        if self.use_prototype_fusion:
            scenario_logits = scenario_logits + self.prototype_scale * prototype_logits
        family_logits = self.family_head(fused) if self.family_head is not None else None
        location_logits = self.location_head(fused) if self.location_head is not None else None
        return DiagnosticOutputs(
            scenario_logits=scenario_logits,
            family_logits=family_logits,
            location_logits=location_logits,
            prototype_logits=prototype_logits,
            fused_embedding=fused,
            aligned_embedding=aligned,
            semantic_token=None,
            semantic_gate=None,
            semantic_attention=None,
            semantic_fused_token=None,
        )


class HardCoreStage1DiagnosticModel(nn.Module):
    def __init__(
        self,
        encoder: SignalAlignmentEncoder,
        fused_dim: int,
        align_dim: int,
        num_scenarios: int,
        num_families: int,
        num_locations: int,
        freeze_encoder: bool = True,
        use_family_aux: bool = True,
        use_location_aux: bool = True,
        use_prototype_fusion: bool = True,
        hidden_dim: int = 256,
        token_dim: int = 256,
        prototype_topk: int = 6,
        text_semantic_mode: str = "none",
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        self.use_family_aux = use_family_aux
        self.use_location_aux = use_location_aux
        self.use_prototype_fusion = use_prototype_fusion
        if text_semantic_mode not in {"none", "aux", "full", "token"}:
            raise ValueError(f"Unsupported text_semantic_mode: {text_semantic_mode}")
        self.text_semantic_mode = text_semantic_mode
        self.use_text_semantic_fusion = text_semantic_mode != "none"
        self.use_full_text_semantic_fusion = text_semantic_mode == "full"
        self.use_semantic_token_fusion = text_semantic_mode == "token"

        graph_dim = 14
        self.fused_proj = nn.Sequential(nn.LayerNorm(fused_dim), nn.Linear(fused_dim, token_dim), nn.GELU())
        self.aligned_proj = nn.Sequential(nn.LayerNorm(align_dim), nn.Linear(align_dim, token_dim), nn.GELU())
        self.modality_proj = nn.Sequential(nn.LayerNorm(7), nn.Linear(7, token_dim), nn.GELU())
        self.graph_proj = nn.Sequential(nn.LayerNorm(graph_dim), nn.Linear(graph_dim, token_dim), nn.GELU())
        self.query_proj = nn.Sequential(
            nn.LayerNorm(fused_dim + token_dim + 7 + graph_dim),
            nn.Linear(fused_dim + token_dim + 7 + graph_dim, token_dim),
            nn.GELU(),
        )
        self.prototype_context = PrototypeContextAttention(
            query_dim=token_dim,
            prototype_dim=align_dim,
            token_dim=token_dim,
            num_heads=4,
            top_k=prototype_topk,
        )
        if self.use_text_semantic_fusion:
            self.family_context_pool = SemanticPrototypePooling(query_dim=token_dim, prototype_dim=align_dim, token_dim=token_dim)
            self.location_context_pool = SemanticPrototypePooling(query_dim=token_dim, prototype_dim=align_dim, token_dim=token_dim)
        if self.use_full_text_semantic_fusion or self.use_semantic_token_fusion:
            self.semantic_context_merge = nn.Sequential(
                nn.LayerNorm(token_dim * 3),
                nn.Linear(token_dim * 3, token_dim),
                nn.GELU(),
            )
            self.semantic_gate = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_mixer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        fusion_input_dim = (token_dim * 5 + 7 + graph_dim) if (self.use_full_text_semantic_fusion or self.use_semantic_token_fusion) else (token_dim * 4 + 7 + graph_dim)
        self.fusion_grn = GatedResidualNetwork(fusion_input_dim, hidden_dim * 2, output_dim=hidden_dim * 2, dropout=0.12)
        self.recalibration = FeatureRecalibration(hidden_dim * 2, reduction=4)
        self.scenario_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_scenarios),
        )
        self.family_head = None
        self.location_head = None
        if use_family_aux:
            self.family_head = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim // 2, num_families),
            )
        if use_location_aux:
            self.location_head = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(hidden_dim // 2, num_locations),
            )
        self.prototype_scale = nn.Parameter(torch.tensor(0.65))
        self.prototype_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.prototype_residual_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_scenarios),
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, batch, prototype_bank) -> DiagnosticOutputs:
        if self.freeze_encoder:
            with torch.no_grad():
                enc = self.encoder(batch)
        else:
            enc = self.encoder(batch)
        resolved_bank = resolve_diagnostic_prototype_bank(prototype_bank)
        fused = enc.fused
        aligned = enc.embedding
        scenario_prototypes = resolved_bank.scenario_embeddings.to(aligned.device)
        family_prototypes = (
            resolved_bank.family_embeddings.to(aligned.device) if resolved_bank.family_embeddings is not None else None
        )
        location_prototypes = (
            resolved_bank.location_embeddings.to(aligned.device) if resolved_bank.location_embeddings is not None else None
        )
        prototype_logits = aligned @ scenario_prototypes.T

        modality_importance = enc.modality_importance
        edge_attention = enc.edge_attention
        edge_row_mean = edge_attention.mean(dim=2)
        edge_diag = torch.diagonal(edge_attention, dim1=1, dim2=2)
        graph_summary = torch.cat([edge_row_mean, edge_diag], dim=1)

        fused_token = self.fused_proj(fused)
        aligned_token = self.aligned_proj(aligned)
        modality_token = self.modality_proj(modality_importance)
        graph_token = self.graph_proj(graph_summary)

        query_seed = self.query_proj(torch.cat([fused, aligned_token, modality_importance, graph_summary], dim=1))
        prototype_context, attn_weights = self.prototype_context(query_seed, scenario_prototypes, prototype_logits)
        semantic_context = prototype_context
        family_text_logits = None
        location_text_logits = None
        if self.use_text_semantic_fusion and family_prototypes is not None and location_prototypes is not None:
            family_text_logits = aligned @ family_prototypes.T
            location_text_logits = aligned @ location_prototypes.T
            family_context = self.family_context_pool(query_seed, family_prototypes, family_text_logits)
            location_context = self.location_context_pool(query_seed, location_prototypes, location_text_logits)
            if self.use_full_text_semantic_fusion or self.use_semantic_token_fusion:
                semantic_context = self.semantic_context_merge(
                    torch.cat([prototype_context, family_context, location_context], dim=1)
                )
        if self.use_semantic_token_fusion:
            semantic_gate_val = self.semantic_gate(query_seed)
            semantic_context = semantic_gate_val * semantic_context
        else:
            semantic_gate_val = None

        if self.use_full_text_semantic_fusion:
            tokens = torch.stack(
                [
                    fused_token,
                    aligned_token,
                    modality_token,
                    graph_token,
                    prototype_context,
                    semantic_context,
                    fused_token + semantic_context,
                ],
                dim=1,
            )
        elif self.use_semantic_token_fusion:
            tokens = torch.stack(
                [
                    fused_token,
                    aligned_token,
                    modality_token,
                    graph_token,
                    prototype_context,
                    semantic_context,
                    fused_token + semantic_context,
                ],
                dim=1,
            )
        else:
            tokens = torch.stack(
                [
                    fused_token,
                    aligned_token,
                    modality_token,
                    graph_token,
                    prototype_context,
                    fused_token + prototype_context,
                ],
                dim=1,
            )
        mixed = self.token_mixer(tokens)
        pooled = torch.cat([mixed[:, 0], mixed.mean(dim=1), mixed.amax(dim=1)], dim=1)
        if self.use_full_text_semantic_fusion or self.use_semantic_token_fusion:
            fused_features = torch.cat([pooled, semantic_context, prototype_context, modality_importance, graph_summary], dim=1)
        else:
            fused_features = torch.cat([pooled, prototype_context, modality_importance, graph_summary], dim=1)
        fused_features = self.fusion_grn(fused_features)
        fused_features = self.recalibration(fused_features)

        scenario_logits = self.scenario_head(fused_features)
        prototype_bias = self.prototype_residual_head(prototype_context)
        if self.use_prototype_fusion:
            scenario_logits = scenario_logits + self.prototype_gate(fused_features) * (
                self.prototype_scale * prototype_logits + 0.20 * prototype_bias
            )
        if self.use_full_text_semantic_fusion and family_text_logits is not None and location_text_logits is not None:
            family_summary = family_text_logits.max(dim=1, keepdim=True).values
            location_summary = location_text_logits.max(dim=1, keepdim=True).values
            full_semantic_gate_val = self.prototype_gate(fused_features)
            scenario_logits = scenario_logits + full_semantic_gate_val * 0.08 * (
                self.prototype_scale * prototype_logits + family_summary + location_summary
            )
        family_logits = self.family_head(fused_features) if self.family_head is not None else None
        location_logits = self.location_head(fused_features) if self.location_head is not None else None
        if family_logits is not None and family_text_logits is not None and self.text_semantic_mode in {"aux", "full"}:
            family_logits = family_logits + 0.15 * family_text_logits
        if location_logits is not None and location_text_logits is not None and self.text_semantic_mode in {"aux", "full"}:
            location_logits = location_logits + 0.15 * location_text_logits
        return DiagnosticOutputs(
            scenario_logits=scenario_logits,
            family_logits=family_logits,
            location_logits=location_logits,
            prototype_logits=prototype_logits,
            fused_embedding=fused_features,
            aligned_embedding=aligned,
            semantic_token=semantic_context if self.use_text_semantic_fusion else None,
            semantic_gate=semantic_gate_val,
            semantic_attention=attn_weights if attn_weights is not None else None,
            semantic_fused_token=(fused_token + semantic_context) if self.use_text_semantic_fusion else None,
        )


class Stage1DiagnosticModel(nn.Module):
    def __init__(
        self,
        encoder: SignalAlignmentEncoder,
        fused_dim: int,
        align_dim: int,
        num_scenarios: int,
        num_families: int,
        num_locations: int,
        freeze_encoder: bool = True,
        use_family_aux: bool = True,
        use_location_aux: bool = True,
        use_prototype_fusion: bool = True,
        hidden_dim: int = 256,
        head_type: str = "hardcore",
        token_dim: int = 256,
        prototype_topk: int = 6,
        use_text_semantic_fusion: bool = False,
    ) -> None:
        super().__init__()
        if head_type not in {"hardcore", "legacy", "semantic_hardcore", "semantic_aux_hardcore", "semantic_token_hardcore"}:
            raise ValueError(f"Unsupported head_type: {head_type}")
        self.head_type = head_type
        if head_type == "legacy":
            self.impl = LegacyStage1DiagnosticModel(
                encoder=encoder,
                fused_dim=fused_dim,
                align_dim=align_dim,
                num_scenarios=num_scenarios,
                num_families=num_families,
                num_locations=num_locations,
                freeze_encoder=freeze_encoder,
                use_family_aux=use_family_aux,
                use_location_aux=use_location_aux,
                use_prototype_fusion=use_prototype_fusion,
                hidden_dim=hidden_dim,
            )
        else:
            self.impl = HardCoreStage1DiagnosticModel(
                encoder=encoder,
                fused_dim=fused_dim,
                align_dim=align_dim,
                num_scenarios=num_scenarios,
                num_families=num_families,
                num_locations=num_locations,
                freeze_encoder=freeze_encoder,
                use_family_aux=use_family_aux,
                use_location_aux=use_location_aux,
                use_prototype_fusion=use_prototype_fusion,
                hidden_dim=hidden_dim,
                token_dim=token_dim,
                prototype_topk=prototype_topk,
                text_semantic_mode=(
                    "token"
                    if head_type == "semantic_token_hardcore"
                    else "full"
                    if (use_text_semantic_fusion or head_type == "semantic_hardcore")
                    else "aux" if head_type == "semantic_aux_hardcore" else "none"
                ),
            )

    def forward(self, batch, scenario_prototypes: Tensor) -> DiagnosticOutputs:
        return self.impl(batch, scenario_prototypes)


def _hard_negative_index_map(scenario_names: list[str] | tuple[str, ...] | None) -> dict[str, int]:
    if scenario_names is None:
        return {}
    return {name: idx for idx, name in enumerate(scenario_names)}


def _mean_or_zero(tensor: Tensor) -> Tensor:
    if tensor.numel() == 0:
        return tensor.new_zeros(())
    return tensor.mean()


def _targeted_hard_negative_loss(
    outputs: DiagnosticOutputs,
    y_cls: Tensor,
    target_idx: int | None,
    negative_indices: list[int],
    margin: float,
    similarity_margin: float,
) -> tuple[Tensor, Tensor]:
    if target_idx is None or not negative_indices:
        zero = outputs.scenario_logits.new_zeros(())
        return zero, zero
    positive_mask = y_cls == target_idx
    negative_mask = torch.zeros_like(positive_mask)
    for idx in negative_indices:
        negative_mask |= y_cls == idx
    margin_loss = outputs.scenario_logits.new_zeros(())
    contrastive_loss = outputs.scenario_logits.new_zeros(())
    if positive_mask.any():
        positive_target = outputs.scenario_logits[positive_mask, target_idx]
        positive_negatives = outputs.scenario_logits[positive_mask][:, negative_indices].max(dim=1).values
        margin_loss = margin_loss + _mean_or_zero(torch.relu(margin - (positive_target - positive_negatives)))
    if negative_mask.any():
        negative_true = outputs.scenario_logits[negative_mask].gather(1, y_cls[negative_mask].unsqueeze(1)).squeeze(1)
        negative_target = outputs.scenario_logits[negative_mask, target_idx]
        margin_loss = margin_loss + _mean_or_zero(torch.relu(margin - (negative_true - negative_target)))
    if positive_mask.any() and negative_mask.any():
        pos_emb = F.normalize(outputs.aligned_embedding[positive_mask], dim=1)
        neg_emb = F.normalize(outputs.aligned_embedding[negative_mask], dim=1)
        sim = pos_emb @ neg_emb.T
        contrastive_loss = _mean_or_zero(torch.relu(sim - similarity_margin))
    return margin_loss, contrastive_loss


def diagnostic_loss(
    outputs: DiagnosticOutputs,
    y_cls: Tensor,
    y_family: Tensor,
    y_loc: Tensor,
    scenario_names: list[str] | tuple[str, ...] | None = None,
    family_weight: float = 0.20,
    location_weight: float = 0.15,
    prototype_consistency_weight: float = 0.10,
    prototype_agreement_weight: float = 0.05,
    label_smoothing: float = 0.03,
    hard_negative_margin_weight: float = 0.045,
    hard_negative_contrastive_weight: float = 0.035,
    hard_negative_margin: float = 0.34,
    hard_negative_similarity_margin: float = 0.19,
) -> Tensor:
    loss = F.cross_entropy(outputs.scenario_logits, y_cls, label_smoothing=label_smoothing)
    if outputs.family_logits is not None:
        loss = loss + family_weight * F.cross_entropy(outputs.family_logits, y_family)
    if outputs.location_logits is not None:
        loss = loss + location_weight * F.cross_entropy(outputs.location_logits, y_loc)
    loss = loss + prototype_consistency_weight * F.cross_entropy(outputs.prototype_logits, y_cls)
    loss = loss + prototype_agreement_weight * F.kl_div(
        F.log_softmax(outputs.scenario_logits, dim=1),
        F.softmax(outputs.prototype_logits.detach(), dim=1),
        reduction="batchmean",
    )
    idx_map = _hard_negative_index_map(scenario_names)
    targeted_specs = {
        "load_disturbance_severe": ["normal", "backlash_growth", "speed_sensor_scale", "position_sensor_bias", "friction_wear_severe"],
        "backlash_growth": ["load_disturbance_severe", "normal", "speed_sensor_scale", "position_sensor_bias", "motor_encoder_freeze"],
        "speed_sensor_scale": ["position_sensor_bias", "motor_encoder_freeze", "normal", "backlash_growth", "current_sensor_bias"],
    }
    margin_total = outputs.scenario_logits.new_zeros(())
    contrastive_total = outputs.scenario_logits.new_zeros(())
    active_targets = 0
    for target_name, negative_names in targeted_specs.items():
        target_idx = idx_map.get(target_name)
        negative_indices = [idx_map[name] for name in negative_names if name in idx_map]
        margin_loss, contrastive_loss = _targeted_hard_negative_loss(
            outputs=outputs,
            y_cls=y_cls,
            target_idx=target_idx,
            negative_indices=negative_indices,
            margin=hard_negative_margin,
            similarity_margin=hard_negative_similarity_margin,
        )
        if target_idx is not None and negative_indices:
            margin_total = margin_total + margin_loss
            contrastive_total = contrastive_total + contrastive_loss
            active_targets += 1
    if active_targets > 0:
        loss = loss + hard_negative_margin_weight * (margin_total / active_targets)
        loss = loss + hard_negative_contrastive_weight * (contrastive_total / active_targets)
    return loss
