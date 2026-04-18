from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .multimodal_method import (
    BEARING_COMPETITOR_INDICES,
    BEARING_SCENARIO_INDEX,
    BEARING_TARGET_FEATURE_DIM,
    BOUNDARY_LABEL_ORDER,
    ELECTRICAL_BOUNDARY_INDEX,
    ELECTRICAL_FAMILY_INDEX,
    ELECTRICAL_SCENARIO_INDICES,
    ExplainabilityArtifacts,
    GRAPH_NODE_NAMES,
    LOAD_COMPETITOR_INDICES,
    LOAD_FAMILY_INDEX,
    LOAD_SCENARIO_INDICES,
    LOAD_TARGET_FEATURE_DIM,
    ModelOutputs,
    MultiTaskDiagnosticHead,
    MultimodalBatch,
    PhysicsGraphFusion,
    condition_scenario_logits,
    extract_bearing_targeted_features,
    extract_load_targeted_features,
)


@dataclass
class GateInspection:
    modality_scores: Tensor
    bearing_gate: Tensor
    load_gate: Tensor
    electrical_gate: Tensor
    top_modalities: list[list[str]]
    dominant_channels: dict[str, list[list[int]]]
    per_sample_summary: list[dict[str, Any]]


class ChannelGateWithStats(nn.Module):
    """A drop-in gate with inspectable summary statistics."""

    def __init__(self, channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        hidden = max(hidden_dim, channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels * 5, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        max_abs = x.abs().amax(dim=1)
        last = x[:, -1, :]
        drift = x[:, -1, :] - x[:, 0, :]
        stats = torch.cat([mean, std, max_abs, last, drift], dim=1)
        gates = torch.sigmoid(self.mlp(stats))
        gated = x * gates.unsqueeze(1)
        return gated, gates, {
            "mean": mean,
            "std": std,
            "max_abs": max_abs,
            "last": last,
            "drift": drift,
        }


class AttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        pooled = torch.einsum("bs,bsd->bd", weights, tokens)
        return pooled, weights


class GLUResidualBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.dw = nn.Conv1d(dim, dim, kernel_size, padding=padding, dilation=dilation, groups=dim)
        self.pw = nn.Conv1d(dim, dim * 2, kernel_size=1)
        self.norm = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.dw(x)
        value, gate = self.pw(x).chunk(2, dim=1)
        x = value * torch.sigmoid(gate)
        x = self.norm(self.dropout(x))
        return F.gelu(x + residual)


class SharedTemporalEncoder(nn.Module):
    """V1: one unified encoder class reused across multiple modalities with separate parameters."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        token_count: int,
        hidden_dim: int,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channel_gate = ChannelGateWithStats(input_dim, hidden_dim)
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1)
        self.blocks = nn.ModuleList([
            GLUResidualBlock(model_dim, kernel_size=5, dilation=d, dropout=dropout) for d in dilations
        ])
        self.token_pool = nn.AdaptiveAvgPool1d(token_count)
        self.token_norm = nn.LayerNorm(model_dim)
        self.pool = AttentionPooling(model_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        x, gates, stats = self.channel_gate(x)
        x = self.input_proj(x.transpose(1, 2))
        for block in self.blocks:
            x = block(x)
        tokens = self.token_pool(x).transpose(1, 2)
        tokens = self.token_norm(tokens)
        pooled, _ = self.pool(tokens)
        return tokens, pooled, gates, stats


class SharedContextEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 5, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        last = x[:, -1, :]
        drift = x[:, -1, :] - x[:, 0, :]
        max_abs = x.abs().amax(dim=1)
        return self.mlp(torch.cat([mean, std, last, drift, max_abs], dim=1))


class UnifiedSignalEncoderV1(nn.Module):
    """V1 backbone: still distributed by modality, but uses one unified encoder family."""

    def __init__(self, input_dims: dict[str, int], model_dim: int = 128) -> None:
        super().__init__()
        specs = {
            "pos": (24, 48),
            "elec": (24, 64),
            "therm": (12, 32),
            "vib": (16, 32),
            "res": (20, 64),
            "freq": (12, 48),
        }
        self.encoders = nn.ModuleDict({
            name: SharedTemporalEncoder(
                input_dim=input_dims[name],
                model_dim=model_dim,
                token_count=specs[name][0],
                hidden_dim=specs[name][1],
            )
            for name in specs
        })
        self.ctx_encoder = SharedContextEncoder(input_dims["ctx"], model_dim)

    def forward(self, batch: MultimodalBatch) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, dict[str, Tensor]]]:
        nodes: dict[str, Tensor] = {}
        gates: dict[str, Tensor] = {}
        tokens: dict[str, Tensor] = {}
        stats: dict[str, dict[str, Tensor]] = {}
        for name in ("pos", "elec", "therm", "vib", "res", "freq"):
            t, p, g, s = self.encoders[name](getattr(batch, name))
            tokens[name] = t
            nodes[name] = p
            gates[name] = g
            stats[name] = s
        nodes["ctx"] = self.ctx_encoder(batch.ctx)
        return tokens, nodes, gates, stats


class GlobalPatchTokenizer(nn.Module):
    """V2 helper: turns all modalities into a single token stream."""

    def __init__(self, input_dims: dict[str, int], model_dim: int, tokens_per_modality: int = 8) -> None:
        super().__init__()
        self.modality_names = ["pos", "elec", "therm", "vib", "res", "freq", "ctx"]
        self.tokens_per_modality = tokens_per_modality
        self.input_proj = nn.ModuleDict({
            name: nn.Linear(input_dims[name], model_dim) for name in self.modality_names
        })
        self.modality_embed = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, 1, model_dim) * 0.02) for name in self.modality_names
        })
        self.token_pool = nn.ModuleDict({
            name: nn.AdaptiveAvgPool1d(tokens_per_modality) for name in self.modality_names if name != "ctx"
        })
        self.ctx_token_count = 2
        self.ctx_mlp = nn.Sequential(
            nn.Linear(input_dims["ctx"] * 3, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim * self.ctx_token_count),
        )
        self.positional = nn.Parameter(torch.randn(1, tokens_per_modality * 6 + self.ctx_token_count + 1, model_dim) * 0.02)

    def forward(self, batch: MultimodalBatch, cls_token: Tensor) -> tuple[Tensor, dict[str, slice]]:
        streams: list[Tensor] = [cls_token]
        slices: dict[str, slice] = {}
        cursor = 1
        for name in ("pos", "elec", "therm", "vib", "res", "freq"):
            x = getattr(batch, name)
            x = self.input_proj[name](x)
            x = self.token_pool[name](x.transpose(1, 2)).transpose(1, 2)
            x = x + self.modality_embed[name]
            streams.append(x)
            slices[name] = slice(cursor, cursor + x.shape[1])
            cursor += x.shape[1]
        ctx = batch.ctx
        ctx_stats = torch.cat([ctx.mean(dim=1), ctx.std(dim=1, unbiased=False), ctx[:, -1, :]], dim=1)
        ctx_tokens = self.ctx_mlp(ctx_stats).view(ctx.shape[0], self.ctx_token_count, -1)
        ctx_tokens = ctx_tokens + self.modality_embed["ctx"]
        streams.append(ctx_tokens)
        slices["ctx"] = slice(cursor, cursor + ctx_tokens.shape[1])
        seq = torch.cat(streams, dim=1)
        seq = seq + self.positional[:, : seq.shape[1], :]
        return seq, slices


class UnifiedTokenEncoderV2(nn.Module):
    """V2 backbone: a single transformer over one unified token sequence."""

    def __init__(
        self,
        input_dims: dict[str, int],
        model_dim: int = 128,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        tokens_per_modality: int = 8,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.tokenizer = GlobalPatchTokenizer(input_dims, model_dim, tokens_per_modality=tokens_per_modality)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.norm = nn.LayerNorm(model_dim)
        self.inspect_proj = nn.Linear(model_dim, 1)

    def forward(self, batch: MultimodalBatch) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, dict[str, Tensor]]]:
        cls = self.cls.expand(batch.pos.shape[0], -1, -1)
        seq, slices = self.tokenizer(batch, cls)
        seq = self.norm(self.encoder(seq))

        tokens: dict[str, Tensor] = {}
        nodes: dict[str, Tensor] = {}
        gates: dict[str, Tensor] = {}
        stats: dict[str, dict[str, Tensor]] = {}
        for name, sl in slices.items():
            t = seq[:, sl, :]
            tokens[name] = t
            node = t.mean(dim=1)
            nodes[name] = node
            score = torch.sigmoid(self.inspect_proj(t).mean(dim=1)).squeeze(-1)
            if name != "ctx":
                channel_dim = getattr(batch, name).shape[-1]
                gate = score.unsqueeze(1).expand(-1, channel_dim)
                gates[name] = gate
                stats[name] = {"global_score": score}
        return tokens, nodes, gates, stats


class SqueezeExciteTemporalMixer(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = dim * expansion
        self.pre = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.mix = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden)
        self.se = nn.Sequential(
            nn.Linear(hidden, max(hidden // 4, 8)),
            nn.GELU(),
            nn.Linear(max(hidden // 4, 8), hidden),
            nn.Sigmoid(),
        )
        self.post = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.pre(x)
        x = self.mix(x.transpose(1, 2)).transpose(1, 2)
        gate = self.se(x.mean(dim=1)).unsqueeze(1)
        x = self.post(x * gate)
        return F.gelu(x + residual)


class ImprovedTemporalEncoderV3(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, token_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.channel_gate = ChannelGateWithStats(input_dim, hidden_dim)
        self.proj = nn.Linear(input_dim, model_dim)
        self.blocks = nn.ModuleList([SqueezeExciteTemporalMixer(model_dim) for _ in range(3)])
        self.token_pool = nn.AdaptiveAvgPool1d(token_count)
        self.norm = nn.LayerNorm(model_dim)
        self.pool = AttentionPooling(model_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        x, gates, stats = self.channel_gate(x)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        tokens = self.token_pool(x.transpose(1, 2)).transpose(1, 2)
        tokens = self.norm(tokens)
        pooled, _ = self.pool(tokens)
        return tokens, pooled, gates, stats


class ImprovedThermalEncoderV3(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, token_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.channel_gate = ChannelGateWithStats(input_dim, hidden_dim)
        self.gru = nn.GRU(input_dim, model_dim // 2, num_layers=1, batch_first=True, bidirectional=True)
        self.trend_proj = nn.Linear(input_dim * 2, model_dim)
        self.token_pool = nn.AdaptiveAvgPool1d(token_count)
        self.norm = nn.LayerNorm(model_dim)
        self.pool = AttentionPooling(model_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        x, gates, stats = self.channel_gate(x)
        seq, _ = self.gru(x)
        trend = torch.cat([x.mean(dim=1), x[:, -1, :] - x[:, 0, :]], dim=1)
        trend_token = self.trend_proj(trend).unsqueeze(1)
        seq = seq + trend_token
        tokens = self.token_pool(seq.transpose(1, 2)).transpose(1, 2)
        tokens = self.norm(tokens)
        pooled, _ = self.pool(tokens)
        return tokens, pooled, gates, stats


class ImprovedVibrationEncoderV3(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, token_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.channel_gate = ChannelGateWithStats(input_dim, hidden_dim)
        self.time_proj = nn.Linear(input_dim, model_dim)
        self.time_blocks = nn.ModuleList([SqueezeExciteTemporalMixer(model_dim) for _ in range(2)])
        self.spec_proj = nn.Linear(input_dim, model_dim)
        self.token_pool = nn.AdaptiveAvgPool1d(token_count)
        self.norm = nn.LayerNorm(model_dim)
        self.pool = AttentionPooling(model_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        x, gates, stats = self.channel_gate(x)
        time_tokens = self.time_proj(x)
        for block in self.time_blocks:
            time_tokens = block(time_tokens)
        freq = torch.fft.rfft(x, dim=1)
        spec_tokens = self.spec_proj(torch.log1p(freq.abs()))
        spec_tokens = F.adaptive_avg_pool1d(spec_tokens.transpose(1, 2), time_tokens.shape[1]).transpose(1, 2)
        tokens = time_tokens + spec_tokens
        tokens = self.token_pool(tokens.transpose(1, 2)).transpose(1, 2)
        tokens = self.norm(tokens)
        pooled, _ = self.pool(tokens)
        return tokens, pooled, gates, stats


class ImprovedSpectralEncoderV3(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, token_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.channel_gate = ChannelGateWithStats(input_dim, hidden_dim)
        self.local_proj = nn.Linear(input_dim, model_dim)
        self.global_proj = nn.Linear(input_dim * 3, model_dim)
        self.blocks = nn.ModuleList([SqueezeExciteTemporalMixer(model_dim) for _ in range(2)])
        self.token_pool = nn.AdaptiveAvgPool1d(token_count)
        self.norm = nn.LayerNorm(model_dim)
        self.pool = AttentionPooling(model_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        x, gates, stats = self.channel_gate(x)
        local = self.local_proj(x)
        for block in self.blocks:
            local = block(local)
        summary = torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False), x.abs().amax(dim=1)], dim=1)
        global_token = self.global_proj(summary).unsqueeze(1)
        local = local + global_token
        tokens = self.token_pool(local.transpose(1, 2)).transpose(1, 2)
        tokens = self.norm(tokens)
        pooled, _ = self.pool(tokens)
        return tokens, pooled, gates, stats


class ImprovedContextEncoderV3(nn.Module):
    def __init__(self, input_dim: int, model_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 5, model_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        stats = torch.cat(
            [x.mean(dim=1), x.std(dim=1, unbiased=False), x[:, -1, :], x[:, -1, :] - x[:, 0, :], x.abs().amax(dim=1)],
            dim=1,
        )
        return self.net(stats)


class ImprovedHeterogeneousEncoderV3(nn.Module):
    """V3: still one encoder per modality type, but every modality gets a redesigned encoder."""

    def __init__(self, input_dims: dict[str, int], model_dim: int = 128) -> None:
        super().__init__()
        self.pos_encoder = ImprovedTemporalEncoderV3(input_dims["pos"], model_dim, token_count=24, hidden_dim=48)
        self.elec_encoder = ImprovedTemporalEncoderV3(input_dims["elec"], model_dim, token_count=24, hidden_dim=64)
        self.therm_encoder = ImprovedThermalEncoderV3(input_dims["therm"], model_dim, token_count=12, hidden_dim=32)
        self.vib_encoder = ImprovedVibrationEncoderV3(input_dims["vib"], model_dim, token_count=16, hidden_dim=32)
        self.res_encoder = ImprovedTemporalEncoderV3(input_dims["res"], model_dim, token_count=20, hidden_dim=64)
        self.freq_encoder = ImprovedSpectralEncoderV3(input_dims["freq"], model_dim, token_count=12, hidden_dim=48)
        self.ctx_encoder = ImprovedContextEncoderV3(input_dims["ctx"], model_dim)

    def forward(self, batch: MultimodalBatch) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], dict[str, dict[str, Tensor]]]:
        tokens: dict[str, Tensor] = {}
        nodes: dict[str, Tensor] = {}
        gates: dict[str, Tensor] = {}
        stats: dict[str, dict[str, Tensor]] = {}
        for name, encoder in (
            ("pos", self.pos_encoder),
            ("elec", self.elec_encoder),
            ("therm", self.therm_encoder),
            ("vib", self.vib_encoder),
            ("res", self.res_encoder),
            ("freq", self.freq_encoder),
        ):
            t, p, g, s = encoder(getattr(batch, name))
            tokens[name] = t
            nodes[name] = p
            gates[name] = g
            stats[name] = s
        nodes["ctx"] = self.ctx_encoder(batch.ctx)
        return tokens, nodes, gates, stats


class ComparativeServoReasoningNet(nn.Module):
    """A comparative wrapper that preserves the original fusion/head logic.

    Variants:
      - v1_shared: distributed multimodal branches, same encoder family for each modality.
      - v2_unified: single unified token encoder over all modalities.
      - v3_improved: heterogeneous modality-specific encoders with new architectures.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        num_families: int,
        num_locations: int,
        scenario_to_family: list[int] | np.ndarray | Tensor,
        variant: str = "v1_shared",
        num_boundary_classes: int = len(BOUNDARY_LABEL_ORDER),
        model_dim: int = 128,
    ) -> None:
        super().__init__()
        if variant not in {"v1_shared", "v2_unified", "v3_improved"}:
            raise ValueError(f"Unsupported variant: {variant}")
        self.variant = variant

        scenario_to_family_tensor = torch.as_tensor(scenario_to_family, dtype=torch.long)
        family_to_scenario_mask = torch.zeros(num_families, num_classes, dtype=torch.bool)
        family_to_scenario_mask[scenario_to_family_tensor, torch.arange(num_classes, dtype=torch.long)] = True
        self.register_buffer("scenario_to_family", scenario_to_family_tensor, persistent=True)
        self.register_buffer("family_to_scenario_mask", family_to_scenario_mask, persistent=True)

        if variant == "v1_shared":
            self.encoder = UnifiedSignalEncoderV1(input_dims, model_dim=model_dim)
        elif variant == "v2_unified":
            self.encoder = UnifiedTokenEncoderV2(input_dims, model_dim=model_dim)
        else:
            self.encoder = ImprovedHeterogeneousEncoderV3(input_dims, model_dim=model_dim)

        self.fusion = PhysicsGraphFusion(model_dim)
        self.head = MultiTaskDiagnosticHead(model_dim, num_classes, num_families, num_locations, num_boundary_classes)
        self.bearing_specialist = nn.Sequential(
            nn.LayerNorm(model_dim * 2 + BEARING_TARGET_FEATURE_DIM),
            nn.Linear(model_dim * 2 + BEARING_TARGET_FEATURE_DIM, model_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(model_dim, 1),
        )
        load_specialist_dim = model_dim * 4 + LOAD_TARGET_FEATURE_DIM
        self.load_presence_specialist = nn.Sequential(
            nn.LayerNorm(load_specialist_dim),
            nn.Linear(load_specialist_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(model_dim, 1),
        )
        self.load_severity_specialist = nn.Sequential(
            nn.LayerNorm(load_specialist_dim),
            nn.Linear(load_specialist_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(model_dim, 2),
        )
        self.last_gate_stats: dict[str, dict[str, Tensor]] | None = None
        self.last_gate_inspection: GateInspection | None = None

    def _compute_gate_inspection(
        self,
        gates: dict[str, Tensor],
        modality_importance: Tensor,
        bearing_gate: Tensor,
        load_gate: Tensor,
        electrical_gate: Tensor,
        topk: int = 3,
    ) -> GateInspection:
        modality_names = GRAPH_NODE_NAMES
        k = min(topk, len(modality_names))
        top_mod_idx = modality_importance.topk(k=k, dim=1).indices.tolist()
        top_modalities = [[modality_names[j] for j in row] for row in top_mod_idx]

        dominant_channels: dict[str, list[list[int]]] = {}
        for name, gate in gates.items():
            kk = min(topk, gate.shape[1])
            dominant_channels[name] = gate.topk(k=kk, dim=1).indices.tolist()

        per_sample_summary: list[dict[str, Any]] = []
        batch_size = modality_importance.shape[0]
        for i in range(batch_size):
            summary = {
                "top_modalities": top_modalities[i],
                "modality_scores": {
                    modality_names[j]: float(modality_importance[i, j].detach().cpu().item()) for j in range(len(modality_names))
                },
                "bearing_gate": float(bearing_gate[i].detach().cpu().item()),
                "load_gate": float(load_gate[i].detach().cpu().item()),
                "electrical_gate": float(electrical_gate[i].detach().cpu().item()),
                "dominant_channels": {
                    name: dominant_channels[name][i] for name in dominant_channels
                },
            }
            if summary["bearing_gate"] > 0.55:
                summary["active_decisions"] = "bearing_specialist"
            elif summary["load_gate"] > 0.55:
                summary["active_decisions"] = "load_specialist"
            elif summary["electrical_gate"] > 0.45:
                summary["active_decisions"] = "electrical_specialist"
            else:
                summary["active_decisions"] = "base_fusion"
            per_sample_summary.append(summary)

        return GateInspection(
            modality_scores=modality_importance,
            bearing_gate=bearing_gate,
            load_gate=load_gate,
            electrical_gate=electrical_gate,
            top_modalities=top_modalities,
            dominant_channels=dominant_channels,
            per_sample_summary=per_sample_summary,
        )

    def forward(self, batch: MultimodalBatch) -> ModelOutputs:
        tokens, nodes, gates, gate_stats = self.encoder(batch)
        self.last_gate_stats = gate_stats

        node_tensor = torch.stack([
            nodes["pos"],
            nodes["elec"],
            nodes["therm"],
            nodes["vib"],
            nodes["res"],
            nodes["freq"],
            nodes["ctx"],
        ], dim=1)
        fused, _, modality_importance, edge_attention, physics_loss, alignment_loss = self.fusion(node_tensor)

        elec_node = nodes["elec"]
        therm_node = nodes["therm"]
        res_node = nodes["res"]
        freq_node = nodes["freq"]
        vib_node = nodes["vib"]
        pos_node = nodes["pos"]
        ctx_node = nodes["ctx"]

        boundary_features = torch.cat([elec_node, therm_node, res_node, freq_node], dim=1)
        det_logits, normality_logits, cls_logits, family_logits, loc_logits, boundary_logits, electro_thermal_logits, severity, rul = self.head(
            fused,
            boundary_features,
        )

        bearing_target_features = extract_bearing_targeted_features(batch.freq, batch.vib)
        bearing_specialist_input = torch.cat([vib_node, freq_node, bearing_target_features], dim=1)
        bearing_specialist_logit = self.bearing_specialist(bearing_specialist_input).squeeze(-1)
        bearing_gate = torch.sigmoid(bearing_specialist_logit)

        load_target_features = extract_load_targeted_features(batch.pos, batch.elec, batch.res, batch.ctx)
        load_specialist_input = torch.cat([pos_node, elec_node, res_node, ctx_node, load_target_features], dim=1)
        load_presence_logit = self.load_presence_specialist(load_specialist_input).squeeze(-1)
        load_gate = torch.sigmoid(load_presence_logit)
        load_severity_logits = self.load_severity_specialist(load_specialist_input)

        family_probs = torch.softmax(family_logits, dim=1)
        boundary_probs = torch.softmax(boundary_logits, dim=1)
        electrical_gate = torch.sqrt(
            family_probs[:, ELECTRICAL_FAMILY_INDEX].clamp_min(1.0e-6)
            * boundary_probs[:, ELECTRICAL_BOUNDARY_INDEX].clamp_min(1.0e-6)
        )
        load_family_gate = torch.sqrt(
            family_probs[:, LOAD_FAMILY_INDEX].clamp_min(1.0e-6)
            * load_gate.clamp_min(1.0e-6)
        )

        pred_family = family_logits.argmax(dim=1)
        cls_logits_conditioned = condition_scenario_logits(cls_logits, pred_family, self.family_to_scenario_mask)
        cls_logits_fused = cls_logits.clone()

        centered_electrical_logits = electro_thermal_logits - electro_thermal_logits.mean(dim=1, keepdim=True)
        for local_index, global_index in enumerate(ELECTRICAL_SCENARIO_INDICES):
            cls_logits_fused[:, global_index] = (
                cls_logits_fused[:, global_index] + 0.85 * electrical_gate * centered_electrical_logits[:, local_index]
            )

        centered_load_logits = load_severity_logits - load_severity_logits.mean(dim=1, keepdim=True)
        for local_index, global_index in enumerate(LOAD_SCENARIO_INDICES):
            cls_logits_fused[:, global_index] = (
                cls_logits_fused[:, global_index]
                + 0.38 * load_family_gate
                + 0.55 * load_family_gate * centered_load_logits[:, local_index]
            )

        cls_logits_fused[:, BEARING_SCENARIO_INDEX] = (
            cls_logits_fused[:, BEARING_SCENARIO_INDEX] + 1.50 * bearing_specialist_logit - 0.20 * (1.0 - bearing_gate)
        )
        for competitor_index in BEARING_COMPETITOR_INDICES:
            cls_logits_fused[:, competitor_index] = cls_logits_fused[:, competitor_index] - 0.35 * bearing_gate
        for competitor_index in LOAD_COMPETITOR_INDICES:
            cls_logits_fused[:, competitor_index] = cls_logits_fused[:, competitor_index] - 0.12 * load_family_gate

        explainability = ExplainabilityArtifacts(
            channel_weights=gates,
            modality_importance=modality_importance,
            edge_attention=edge_attention,
            physics_loss=physics_loss,
            alignment_loss=alignment_loss,
        )

        self.last_gate_inspection = self._compute_gate_inspection(
            gates=gates,
            modality_importance=modality_importance,
            bearing_gate=bearing_gate,
            load_gate=load_gate,
            electrical_gate=electrical_gate,
        )

        return ModelOutputs(
            det_logits=det_logits,
            normality_logits=normality_logits,
            cls_logits=cls_logits,
            cls_logits_conditioned=cls_logits_conditioned,
            cls_logits_fused=cls_logits_fused,
            bearing_specialist_logit=bearing_specialist_logit,
            load_presence_logit=load_presence_logit,
            load_severity_logits=load_severity_logits,
            family_logits=family_logits,
            loc_logits=loc_logits,
            boundary_logits=boundary_logits,
            electro_thermal_logits=electro_thermal_logits,
            severity=severity,
            rul=rul,
            explainability=explainability,
        )


def summarize_gate_inspection(model: ComparativeServoReasoningNet) -> list[dict[str, Any]]:
    if model.last_gate_inspection is None:
        raise RuntimeError("No gate inspection available. Run a forward pass first.")
    return model.last_gate_inspection.per_sample_summary


def build_comparative_model(
    input_dims: dict[str, int],
    num_classes: int,
    num_families: int,
    num_locations: int,
    scenario_to_family: list[int] | np.ndarray | Tensor,
    variant: str = "v1_shared",
    num_boundary_classes: int = len(BOUNDARY_LABEL_ORDER),
    model_dim: int = 128,
) -> ComparativeServoReasoningNet:
    return ComparativeServoReasoningNet(
        input_dims=input_dims,
        num_classes=num_classes,
        num_families=num_families,
        num_locations=num_locations,
        scenario_to_family=scenario_to_family,
        variant=variant,
        num_boundary_classes=num_boundary_classes,
        model_dim=model_dim,
    )
