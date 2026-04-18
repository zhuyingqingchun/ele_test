from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from .feature_engineering import (
    FAULT_ACTIVE_RATIO_THRESHOLD,
    WindowConfig,
    assign_window_label,
    build_fault_active_mask,
    group_rows_by_run,
)

MODALITY_COLUMN_GROUPS: dict[str, list[str]] = {
    "pos": [
        "theta_ref_deg",
        "theta_meas_deg",
        "theta_motor_meas_deg",
        "omega_motor_meas_deg_s",
        "omega_load_est_deg_s",
        "load_accel_est_deg_s2",
        "encoder_count",
        "motor_encoder_count",
    ],
    "elec": [
        "current_meas_a",
        "current_d_meas_a",
        "current_q_meas_a",
        "phase_current_a_est_a",
        "phase_current_b_est_a",
        "phase_current_c_est_a",
        "phase_current_imbalance_est_a",
        "voltage_meas_v",
        "vd_cmd_v",
        "vq_cmd_v",
        "phase_voltage_a_meas_v",
        "phase_voltage_b_meas_v",
        "phase_voltage_c_meas_v",
        "bus_current_meas_a",
        "pwm_duty",
        "available_bus_voltage_v",
    ],
    "therm": [
        "winding_temp_c",
        "housing_temp_c",
        "winding_temp_rate_c_s",
        "housing_temp_rate_c_s",
    ],
    "vib": [
        "vibration_accel_mps2",
        "vibration_band_mps2",
        "vibration_envelope_mps2",
        "vibration_shock_index",
    ],
    "res": [
        "position_error_deg",
        "speed_residual_deg_s",
        "current_residual_a",
        "shaft_twist_deg",
        "torque_shaft_nm",
        "torque_load_nm",
        "torque_friction_nm",
        "back_emf_v",
        "electrical_power_w",
        "mechanical_power_w",
        "copper_loss_w",
        "iron_loss_w",
    ],
}

CONTEXT_COLUMNS = [
    "theta_ref_deg",
    "speed_ref_deg_s",
    "motor_speed_ref_deg_s",
    "current_ref_a",
    "current_limit_a",
    "voltage_cmd_v",
    "available_bus_voltage_v",
    "airspeed_actual_mps",
    "ambient_temp_c",
    "pwm_duty",
]

FREQUENCY_SOURCE_COLUMNS = [
    "theta_ref_deg",
    "theta_meas_deg",
    "omega_motor_meas_deg_s",
    "current_meas_a",
    "current_q_meas_a",
    "voltage_meas_v",
    "back_emf_v",
    "position_error_deg",
    "vibration_accel_mps2",
    "vibration_band_mps2",
    "vibration_envelope_mps2",
    "phase_current_imbalance_est_a",
    "shaft_twist_deg",
]
FREQUENCY_FEATURE_NAMES = [
    f"dct__{name}" for name in FREQUENCY_SOURCE_COLUMNS
] + [
    f"stft__{name}" for name in FREQUENCY_SOURCE_COLUMNS
]

MODALITY_NAMES = ["pos", "elec", "therm", "vib", "res", "freq"]
GRAPH_NODE_NAMES = MODALITY_NAMES + ["ctx"]
GRAPH_ADJACENCY = torch.tensor(
    [
        [1, 1, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=torch.float32,
)

DEFAULT_LOCATION_ORDER = [
    "normal",
    "sensor_chain",
    "electrical_drive",
    "mechanical_transmission",
    "thermal_management",
    "load_path",
]

SEVERITY_BY_SCENARIO = {
    "normal": 0.0,
    "load_disturbance_mild": 0.42,
    "load_disturbance_severe": 0.80,
    "friction_wear_mild": 0.40,
    "friction_wear_severe": 0.78,
    "jam_fault": 0.95,
    "intermittent_jam_fault": 0.78,
    "current_sensor_bias": 0.42,
    "speed_sensor_scale": 0.46,
    "position_sensor_bias": 0.45,
    "winding_resistance_rise": 0.68,
    "bus_voltage_sag_fault": 0.65,
    "backlash_growth": 0.58,
    "thermal_saturation": 0.82,
    "motor_encoder_freeze": 0.52,
    "partial_demagnetization": 0.72,
    "inverter_voltage_loss": 0.74,
    "bearing_defect": 0.63,
}

FAMILY_BY_SCENARIO = {
    "normal": "normal",
    "load_disturbance_mild": "load_disturbance",
    "load_disturbance_severe": "load_disturbance",
    "friction_wear_mild": "friction_increase",
    "friction_wear_severe": "friction_increase",
    "jam_fault": "jam",
    "intermittent_jam_fault": "jam",
    "current_sensor_bias": "sensor_fault",
    "speed_sensor_scale": "sensor_fault",
    "position_sensor_bias": "sensor_fault",
    "winding_resistance_rise": "electrical_fault",
    "bus_voltage_sag_fault": "electrical_fault",
    "backlash_growth": "mechanical_fault",
    "thermal_saturation": "electrical_fault",
    "motor_encoder_freeze": "sensor_fault",
    "partial_demagnetization": "electrical_fault",
    "inverter_voltage_loss": "electrical_fault",
    "bearing_defect": "mechanical_fault",
}

BOUNDARY_BY_SCENARIO = {
    "normal": "other",
    "load_disturbance_mild": "other",
    "load_disturbance_severe": "other",
    "friction_wear_mild": "other",
    "friction_wear_severe": "other",
    "jam_fault": "other",
    "intermittent_jam_fault": "other",
    "current_sensor_bias": "sensor_side",
    "speed_sensor_scale": "sensor_side",
    "position_sensor_bias": "sensor_side",
    "winding_resistance_rise": "electrical_side",
    "bus_voltage_sag_fault": "electrical_side",
    "backlash_growth": "other",
    "thermal_saturation": "electrical_side",
    "motor_encoder_freeze": "sensor_side",
    "partial_demagnetization": "electrical_side",
    "inverter_voltage_loss": "electrical_side",
    "bearing_defect": "other",
}

BOUNDARY_LABEL_ORDER = ["electrical_side", "other", "sensor_side"]
ELECTRO_THERMAL_SCENARIOS = [
    "bus_voltage_sag_fault",
    "inverter_voltage_loss",
    "partial_demagnetization",
    "thermal_saturation",
    "winding_resistance_rise",
]
DEFAULT_CLASS_ORDER = sorted(SEVERITY_BY_SCENARIO.keys())
ELECTRICAL_SCENARIO_INDICES = [DEFAULT_CLASS_ORDER.index(name) for name in ELECTRO_THERMAL_SCENARIOS]
ELECTRICAL_GLOBAL_TO_LOCAL = {global_idx: local_idx for local_idx, global_idx in enumerate(ELECTRICAL_SCENARIO_INDICES)}
ELECTRICAL_FAMILY_INDEX = sorted(set(FAMILY_BY_SCENARIO.values())).index("electrical_fault")
ELECTRICAL_BOUNDARY_INDEX = BOUNDARY_LABEL_ORDER.index("electrical_side")
LOAD_FAMILY_INDEX = sorted(set(FAMILY_BY_SCENARIO.values())).index("load_disturbance")
PARTIAL_DEMAGNETIZATION_SCENARIO_INDEX = DEFAULT_CLASS_ORDER.index("partial_demagnetization")
INVERTER_VOLTAGE_LOSS_SCENARIO_INDEX = DEFAULT_CLASS_ORDER.index("inverter_voltage_loss")
BEARING_SCENARIO_INDEX = DEFAULT_CLASS_ORDER.index("bearing_defect")
LOAD_DISTURBANCE_MILD_SCENARIO_INDEX = DEFAULT_CLASS_ORDER.index("load_disturbance_mild")
LOAD_DISTURBANCE_SEVERE_SCENARIO_INDEX = DEFAULT_CLASS_ORDER.index("load_disturbance_severe")
LOAD_SCENARIO_INDICES = [LOAD_DISTURBANCE_MILD_SCENARIO_INDEX, LOAD_DISTURBANCE_SEVERE_SCENARIO_INDEX]
LOAD_GLOBAL_TO_LOCAL = {
    LOAD_DISTURBANCE_MILD_SCENARIO_INDEX: 0,
    LOAD_DISTURBANCE_SEVERE_SCENARIO_INDEX: 1,
}
BEARING_COMPETITOR_SCENARIOS = [
    "backlash_growth",
    "friction_wear_mild",
    "friction_wear_severe",
    "load_disturbance_mild",
    "load_disturbance_severe",
]
BEARING_COMPETITOR_INDICES = [DEFAULT_CLASS_ORDER.index(name) for name in BEARING_COMPETITOR_SCENARIOS]
LOAD_COMPETITOR_SCENARIOS = [
    "normal",
    "speed_sensor_scale",
    "position_sensor_bias",
    "backlash_growth",
]
LOAD_COMPETITOR_INDICES = [DEFAULT_CLASS_ORDER.index(name) for name in LOAD_COMPETITOR_SCENARIOS]
FREQUENCY_STFT_OFFSET = len(FREQUENCY_SOURCE_COLUMNS)
BEARING_TARGET_FEATURE_DIM = 10
LOAD_TARGET_FEATURE_DIM = 10
DEFAULT_SAMPLE_RATE_HZ = 1000.0
DEFAULT_STFT_FRAME_SIZE = 64
BEARING_FAULT_FREQ_HZ = 38.0

DERIVED_RESIDUAL_FEATURE_NAMES = [
    "encoder_mismatch_deg",
    "electro_proxy_w",
    "power_gap_w",
    "thermal_gradient_c",
    "voltage_margin_v",
    "current_limit_margin_a",
    "dq_current_norm_a",
    "shaft_load_gap_nm",
]

LOCATION_BY_SCENARIO = {
    "normal": "normal",
    "load_disturbance_mild": "load_path",
    "load_disturbance_severe": "load_path",
    "friction_wear_mild": "mechanical_transmission",
    "friction_wear_severe": "mechanical_transmission",
    "jam_fault": "mechanical_transmission",
    "intermittent_jam_fault": "mechanical_transmission",
    "current_sensor_bias": "sensor_chain",
    "speed_sensor_scale": "sensor_chain",
    "position_sensor_bias": "sensor_chain",
    "winding_resistance_rise": "electrical_drive",
    "bus_voltage_sag_fault": "electrical_drive",
    "backlash_growth": "mechanical_transmission",
    "thermal_saturation": "thermal_management",
    "motor_encoder_freeze": "sensor_chain",
    "partial_demagnetization": "electrical_drive",
    "inverter_voltage_loss": "electrical_drive",
    "bearing_defect": "mechanical_transmission",
}

FAMILY_TO_KEYWORDS = {
    "normal": ["normal", "stable", "nominal"],
    "load_disturbance": ["load", "gust", "torque"],
    "friction_increase": ["friction", "drag", "stiction"],
    "jam": ["jam", "stuck", "blocked"],
    "sensor_fault": ["sensor", "encoder", "bias", "scale"],
    "electrical_fault": ["voltage", "current", "flux", "winding", "inverter"],
    "mechanical_fault": ["backlash", "bearing", "shaft", "mechanical"],
}


@dataclass
class MultimodalBatch:
    pos: Tensor
    elec: Tensor
    therm: Tensor
    vib: Tensor
    res: Tensor
    freq: Tensor
    ctx: Tensor
    y_det: Tensor
    y_cls: Tensor
    y_family: Tensor
    y_loc: Tensor
    y_boundary: Tensor
    y_sev: Tensor
    y_rul: Tensor


@dataclass
class DatasetArtifacts:
    output_path: Path
    metadata_path: Path
    num_windows: int
    window_size: int
    stride: int
    window_sizes: list[int]
    strides: list[int]
    class_names: list[str]
    family_names: list[str]
    location_names: list[str]


@dataclass
class ExplainabilityArtifacts:
    channel_weights: dict[str, Tensor]
    modality_importance: Tensor
    edge_attention: Tensor
    physics_loss: Tensor
    alignment_loss: Tensor


@dataclass
class ModelOutputs:
    det_logits: Tensor
    normality_logits: Tensor
    cls_logits: Tensor
    cls_logits_conditioned: Tensor
    cls_logits_fused: Tensor
    bearing_specialist_logit: Tensor
    load_presence_logit: Tensor
    load_severity_logits: Tensor
    family_logits: Tensor
    loc_logits: Tensor
    boundary_logits: Tensor
    electro_thermal_logits: Tensor
    severity: Tensor
    rul: Tensor
    explainability: ExplainabilityArtifacts


@dataclass
class TrainingMetrics:
    det_accuracy: float
    cls_accuracy: float
    family_accuracy: float
    loc_accuracy: float
    boundary_accuracy: float
    severity_mae: float
    rul_mae: float
    total_loss: float


@dataclass
class InferenceExplanation:
    diagnosis: str
    evidence_chain: list[str]
    mechanism: str
    maintenance: list[str]
    qa_hint: str
    events: list[dict[str, Any]]
    retrieved_items: list[dict[str, Any]]


class ServoMultimodalDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.indices = indices.astype(np.int64)
        self.arrays = arrays

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> MultimodalBatch:
        item = int(self.indices[idx])
        return MultimodalBatch(
            pos=torch.from_numpy(self.arrays["X_pos"][item]).float(),
            elec=torch.from_numpy(self.arrays["X_elec"][item]).float(),
            therm=torch.from_numpy(self.arrays["X_therm"][item]).float(),
            vib=torch.from_numpy(self.arrays["X_vib"][item]).float(),
            res=torch.from_numpy(self.arrays["X_res"][item]).float(),
            freq=torch.from_numpy(self.arrays["X_freq"][item]).float(),
            ctx=torch.from_numpy(self.arrays["X_ctx"][item]).float(),
            y_det=torch.tensor(self.arrays["y_det"][item], dtype=torch.float32),
            y_cls=torch.tensor(self.arrays["y_cls"][item], dtype=torch.long),
            y_family=torch.tensor(self.arrays["y_family"][item], dtype=torch.long),
            y_loc=torch.tensor(self.arrays["y_loc"][item], dtype=torch.long),
            y_boundary=torch.tensor(self.arrays["y_boundary"][item], dtype=torch.long),
            y_sev=torch.tensor(self.arrays["y_sev"][item], dtype=torch.float32),
            y_rul=torch.tensor(self.arrays["y_rul"][item], dtype=torch.float32),
        )


class ChannelGate(nn.Module):
    def __init__(self, channels: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels * 3, max(hidden_dim, channels)),
            nn.GELU(),
            nn.Linear(max(hidden_dim, channels), channels),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        max_abs = x.abs().amax(dim=1)
        stats = torch.cat([mean, std, max_abs], dim=1)
        gates = torch.sigmoid(self.mlp(stats))
        gated = x * gates.unsqueeze(1)
        return gated, gates


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()
        padding = dilation
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        value, gate = x.chunk(2, dim=1)
        x = value * torch.sigmoid(gate)
        x = self.dropout(x)
        x = self.norm(x[:, : residual.shape[1], :])
        return F.gelu(x + residual)


class AttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        pooled = torch.einsum("bs,bsd->bd", weights, tokens)
        return pooled, weights


class TemporalExplainableEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        token_count: int,
        hidden_dim: int,
        dilations: tuple[int, ...],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channel_gate = ChannelGate(input_dim, hidden_dim)
        self.input_proj = nn.Conv1d(input_dim, model_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [TemporalResidualBlock(model_dim, dilation=dilation, dropout=dropout) for dilation in dilations]
        )
        self.token_pool = nn.AdaptiveAvgPool1d(token_count)
        self.token_norm = nn.LayerNorm(model_dim)
        self.pool = AttentionPooling(model_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, channel_weights = self.channel_gate(x)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        tokens = self.token_pool(x).transpose(1, 2)
        tokens = self.token_norm(tokens)
        pooled, _ = self.pool(tokens)
        return tokens, pooled, channel_weights


class ThermalStateEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, token_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.channel_gate = ChannelGate(input_dim, hidden_dim)
        self.gru = nn.GRU(input_dim, model_dim // 2, num_layers=2, batch_first=True, bidirectional=True)
        self.token_pool = nn.AdaptiveAvgPool1d(token_count)
        self.token_norm = nn.LayerNorm(model_dim)
        self.pool = AttentionPooling(model_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, channel_weights = self.channel_gate(x)
        tokens, _ = self.gru(x)
        tokens = self.token_pool(tokens.transpose(1, 2)).transpose(1, 2)
        tokens = self.token_norm(tokens)
        pooled, _ = self.pool(tokens)
        return tokens, pooled, channel_weights


class VibrationStateEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, token_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.channel_gate = ChannelGate(input_dim, hidden_dim)
        self.time_branch = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.freq_proj = nn.Linear(input_dim, model_dim)
        self.token_count = token_count
        self.token_norm = nn.LayerNorm(model_dim)
        self.pool = AttentionPooling(model_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, channel_weights = self.channel_gate(x)
        time_tokens = self.time_branch(x.transpose(1, 2)).transpose(1, 2)
        time_tokens = F.adaptive_avg_pool1d(time_tokens.transpose(1, 2), self.token_count).transpose(1, 2)

        freq = torch.fft.rfft(x, dim=1)
        freq_mag = torch.log1p(freq.abs())
        freq_tokens = self.freq_proj(freq_mag)
        freq_tokens = F.adaptive_avg_pool1d(freq_tokens.transpose(1, 2), self.token_count).transpose(1, 2)

        tokens = self.token_norm(time_tokens + freq_tokens)
        pooled, _ = self.pool(tokens)
        return tokens, pooled, channel_weights


class SpectralStateEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, token_count: int, hidden_dim: int) -> None:
        super().__init__()
        self.channel_gate = ChannelGate(input_dim, hidden_dim)
        self.freq_branch = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.global_proj = nn.Linear(input_dim * 2, model_dim)
        self.token_count = token_count
        self.token_norm = nn.LayerNorm(model_dim)
        self.pool = AttentionPooling(model_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, channel_weights = self.channel_gate(x)
        local_tokens = self.freq_branch(x.transpose(1, 2)).transpose(1, 2)
        local_tokens = F.adaptive_avg_pool1d(local_tokens.transpose(1, 2), self.token_count).transpose(1, 2)

        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        global_token = self.global_proj(torch.cat([mean, std], dim=1)).unsqueeze(1)
        global_token = global_token.expand(-1, self.token_count, -1)

        tokens = self.token_norm(local_tokens + global_token)
        pooled, _ = self.pool(tokens)
        return tokens, pooled, channel_weights


class ContextEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int, dropout: float = 0.25, output_scale: float = 0.6) -> None:
        super().__init__()
        self.output_scale = output_scale
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 3, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        last = x[:, -1, :]
        stats = torch.cat([mean, std, last], dim=1)
        return self.mlp(stats) * self.output_scale


class PhysicsGraphFusion(nn.Module):
    def __init__(
        self,
        model_dim: int,
        num_nodes: int = len(GRAPH_NODE_NAMES),
        max_context_importance: float = 0.24,
        max_vib_importance: float = 0.26,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_nodes = num_nodes
        self.max_context_importance = max_context_importance
        self.max_vib_importance = max_vib_importance
        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(model_dim, model_dim)
        self.value_proj = nn.Linear(model_dim, model_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.update = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.importance = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 1),
        )
        self.pool = AttentionPooling(model_dim)

    def _cap_context_importance(self, importance: Tensor) -> Tensor:
        if self.max_context_importance <= 0.0 and self.max_vib_importance <= 0.0:
            return importance
        num_nodes = importance.shape[1]
        cap_vector = torch.ones(num_nodes, device=importance.device, dtype=importance.dtype)
        constrained_mask = torch.zeros(num_nodes, device=importance.device, dtype=torch.bool)
        if self.max_vib_importance > 0.0:
            cap_vector[3] = min(self.max_vib_importance, 1.0)
            constrained_mask[3] = True
        if self.max_context_importance > 0.0:
            cap_vector[-1] = min(self.max_context_importance, 1.0)
            constrained_mask[-1] = True

        capped = torch.minimum(importance, cap_vector.unsqueeze(0))
        free_mask = (~constrained_mask).to(dtype=importance.dtype).unsqueeze(0)
        free_mass = (importance * free_mask).sum(dim=1, keepdim=True)
        residual = (1.0 - capped.sum(dim=1, keepdim=True)).clamp_min(0.0)
        free_share = torch.where(
            free_mass > 1.0e-6,
            importance * free_mask / free_mass.clamp_min(1.0e-6),
            free_mask / free_mask.sum(dim=1, keepdim=True).clamp_min(1.0),
        )
        redistributed = capped + free_share * residual
        return redistributed / redistributed.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

    def forward(self, nodes: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        adjacency = GRAPH_ADJACENCY.to(nodes.device)
        context_node = nodes[:, -1:, :].expand(-1, nodes.shape[1], -1)
        q = self.query_proj(nodes)
        k = self.key_proj(nodes)
        v = self.value_proj(nodes)
        logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.model_dim)

        left = nodes.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        right = nodes.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)
        ctx = context_node.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        edge_inputs = torch.cat([left, right, ctx], dim=-1)
        edge_logits = self.edge_gate(edge_inputs).squeeze(-1)
        masked_logits = logits + edge_logits + (adjacency.unsqueeze(0) - 1.0) * 1.0e4
        attn = torch.softmax(masked_logits, dim=-1)
        messages = torch.matmul(attn, v)
        updated = self.update(torch.cat([nodes, messages], dim=-1)) + nodes

        global_query = updated[:, -1:, :].expand(-1, updated.shape[1], -1)
        importance_logits = self.importance(torch.cat([updated, global_query], dim=-1)).squeeze(-1)
        modality_importance = torch.softmax(importance_logits, dim=-1)
        modality_importance = self._cap_context_importance(modality_importance)
        fused = torch.einsum("bn,bnd->bd", modality_importance, updated)

        edge_mask = adjacency.bool()
        edge_cos = F.cosine_similarity(left[0], right[0], dim=-1) if nodes.shape[0] == 1 else None
        del edge_cos
        physics_loss = physics_consistency_loss(updated, adjacency)
        alignment_loss = alignment_consistency_loss(updated, fused)
        return fused, updated, modality_importance, attn, physics_loss + alignment_loss * 0.0, alignment_loss


class MultiTaskDiagnosticHead(nn.Module):
    def __init__(self, model_dim: int, num_classes: int, num_families: int, num_locations: int, num_boundary_classes: int) -> None:
        super().__init__()
        boundary_dim = model_dim * 4
        self.shared = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.det_head = nn.Linear(model_dim, 1)
        self.normality_head = nn.Linear(model_dim, 1)
        self.cls_head = nn.Linear(model_dim, num_classes)
        self.family_head = nn.Linear(model_dim, num_families)
        self.loc_head = nn.Linear(model_dim, num_locations)
        self.boundary_head = nn.Sequential(
            nn.LayerNorm(boundary_dim),
            nn.Linear(boundary_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(model_dim, num_boundary_classes),
        )
        self.electro_thermal_head = nn.Sequential(
            nn.LayerNorm(boundary_dim),
            nn.Linear(boundary_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(model_dim, len(ELECTRO_THERMAL_SCENARIOS)),
        )
        self.sev_head = nn.Sequential(nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Linear(model_dim // 2, 1))
        self.rul_head = nn.Sequential(nn.Linear(model_dim, model_dim // 2), nn.GELU(), nn.Linear(model_dim // 2, 1))

    def forward(self, fused: Tensor, boundary_features: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        shared = self.shared(fused)
        return (
            self.det_head(shared).squeeze(-1),
            self.normality_head(shared).squeeze(-1),
            self.cls_head(shared),
            self.family_head(shared),
            self.loc_head(shared),
            self.boundary_head(boundary_features),
            self.electro_thermal_head(boundary_features),
            self.sev_head(shared).squeeze(-1),
            self.rul_head(shared).squeeze(-1),
        )


class ServoReasoningNet(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        num_families: int,
        num_locations: int,
        scenario_to_family: list[int] | np.ndarray | Tensor,
        num_boundary_classes: int = len(BOUNDARY_LABEL_ORDER),
        model_dim: int = 128,
    ) -> None:
        super().__init__()
        scenario_to_family_tensor = torch.as_tensor(scenario_to_family, dtype=torch.long)
        family_to_scenario_mask = torch.zeros(num_families, num_classes, dtype=torch.bool)
        family_to_scenario_mask[scenario_to_family_tensor, torch.arange(num_classes, dtype=torch.long)] = True
        self.register_buffer("scenario_to_family", scenario_to_family_tensor, persistent=True)
        self.register_buffer("family_to_scenario_mask", family_to_scenario_mask, persistent=True)
        self.pos_encoder = TemporalExplainableEncoder(input_dims["pos"], model_dim, token_count=24, hidden_dim=48, dilations=(1, 2, 4, 8))
        self.elec_encoder = TemporalExplainableEncoder(input_dims["elec"], model_dim, token_count=24, hidden_dim=64, dilations=(1, 2, 4, 8))
        self.therm_encoder = ThermalStateEncoder(input_dims["therm"], model_dim, token_count=12, hidden_dim=32)
        self.vib_encoder = VibrationStateEncoder(input_dims["vib"], model_dim, token_count=16, hidden_dim=32)
        self.res_encoder = TemporalExplainableEncoder(input_dims["res"], model_dim, token_count=20, hidden_dim=64, dilations=(1, 2, 4, 8))
        self.freq_encoder = SpectralStateEncoder(input_dims["freq"], model_dim, token_count=12, hidden_dim=48)
        self.ctx_encoder = ContextEncoder(input_dims["ctx"], model_dim)
        self.fusion = PhysicsGraphFusion(model_dim)
        self.head = MultiTaskDiagnosticHead(model_dim, num_classes, num_families, num_locations, num_boundary_classes)
        self.bearing_feature_dim = BEARING_TARGET_FEATURE_DIM
        self.load_feature_dim = LOAD_TARGET_FEATURE_DIM
        self.bearing_specialist = nn.Sequential(
            nn.LayerNorm(model_dim * 2 + self.bearing_feature_dim),
            nn.Linear(model_dim * 2 + self.bearing_feature_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(model_dim, 1),
        )
        load_specialist_dim = model_dim * 4 + self.load_feature_dim
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

    def forward(self, batch: MultimodalBatch) -> ModelOutputs:
        _, pos_node, pos_gate = self.pos_encoder(batch.pos)
        _, elec_node, elec_gate = self.elec_encoder(batch.elec)
        _, therm_node, therm_gate = self.therm_encoder(batch.therm)
        _, vib_node, vib_gate = self.vib_encoder(batch.vib)
        _, res_node, res_gate = self.res_encoder(batch.res)
        _, freq_node, freq_gate = self.freq_encoder(batch.freq)
        ctx_node = self.ctx_encoder(batch.ctx)

        nodes = torch.stack([pos_node, elec_node, therm_node, vib_node, res_node, freq_node, ctx_node], dim=1)
        fused, _, modality_importance, edge_attention, physics_loss, alignment_loss = self.fusion(nodes)
        boundary_features = torch.cat([elec_node, therm_node, res_node, freq_node], dim=1)
        det_logits, normality_logits, cls_logits, family_logits, loc_logits, boundary_logits, electro_thermal_logits, severity, rul = self.head(fused, boundary_features)
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
                cls_logits_fused[:, global_index]
                + 0.85 * electrical_gate * centered_electrical_logits[:, local_index]
            )
        centered_load_logits = load_severity_logits - load_severity_logits.mean(dim=1, keepdim=True)
        for local_index, global_index in enumerate(LOAD_SCENARIO_INDICES):
            cls_logits_fused[:, global_index] = (
                cls_logits_fused[:, global_index]
                + 0.38 * load_family_gate
                + 0.55 * load_family_gate * centered_load_logits[:, local_index]
            )
        cls_logits_fused[:, BEARING_SCENARIO_INDEX] = (
            cls_logits_fused[:, BEARING_SCENARIO_INDEX]
            + 1.50 * bearing_specialist_logit
            - 0.20 * (1.0 - bearing_gate)
        )
        for competitor_index in BEARING_COMPETITOR_INDICES:
            cls_logits_fused[:, competitor_index] = cls_logits_fused[:, competitor_index] - 0.35 * bearing_gate
        for competitor_index in LOAD_COMPETITOR_INDICES:
            cls_logits_fused[:, competitor_index] = cls_logits_fused[:, competitor_index] - 0.12 * load_family_gate
        explainability = ExplainabilityArtifacts(
            channel_weights={
                "pos": pos_gate,
                "elec": elec_gate,
                "therm": therm_gate,
                "vib": vib_gate,
                "res": res_gate,
                "freq": freq_gate,
            },
            modality_importance=modality_importance,
            edge_attention=edge_attention,
            physics_loss=physics_loss,
            alignment_loss=alignment_loss,
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


def physics_consistency_loss(nodes: Tensor, adjacency: Tensor) -> Tensor:
    left = nodes.unsqueeze(2)
    right = nodes.unsqueeze(1)
    cosine = F.cosine_similarity(left, right, dim=-1)
    edge_mask = adjacency.to(nodes.device)
    edge_loss = ((1.0 - cosine) * edge_mask).sum() / edge_mask.sum().clamp_min(1.0)
    non_edge_mask = (1.0 - edge_mask) * (1.0 - torch.eye(adjacency.shape[0], device=nodes.device))
    non_edge_loss = F.relu(cosine - 0.35) * non_edge_mask
    non_edge_loss = non_edge_loss.sum() / non_edge_mask.sum().clamp_min(1.0)
    return edge_loss + 0.5 * non_edge_loss


def alignment_consistency_loss(nodes: Tensor, fused: Tensor) -> Tensor:
    fused = F.normalize(fused, dim=-1)
    nodes = F.normalize(nodes, dim=-1)
    cosine = torch.einsum("bnd,bd->bn", nodes, fused)
    return (1.0 - cosine).mean()


def build_scenario_family_mapping(class_names: list[str], family_names: list[str]) -> np.ndarray:
    family_index = {name: idx for idx, name in enumerate(family_names)}
    mapping = [family_index[FAMILY_BY_SCENARIO[str(class_name)]] for class_name in class_names]
    return np.array(mapping, dtype=np.int64)


def build_scenario_boundary_mapping(class_names: list[str], boundary_names: list[str]) -> np.ndarray:
    boundary_index = {name: idx for idx, name in enumerate(boundary_names)}
    mapping = [boundary_index[BOUNDARY_BY_SCENARIO[str(class_name)]] for class_name in class_names]
    return np.array(mapping, dtype=np.int64)


def build_scenario_location_mapping(class_names: list[str], location_names: list[str]) -> np.ndarray:
    location_index = {name: idx for idx, name in enumerate(location_names)}
    mapping = [location_index[LOCATION_BY_SCENARIO[str(class_name)]] for class_name in class_names]
    return np.array(mapping, dtype=np.int64)


def build_named_scenario_index(class_names: list[str], scenario_names: list[str]) -> np.ndarray:
    class_index = {name: idx for idx, name in enumerate(class_names)}
    return np.array([class_index[name] for name in scenario_names], dtype=np.int64)


def condition_scenario_logits(logits: Tensor, family_ids: Tensor, family_to_scenario_mask: Tensor) -> Tensor:
    scenario_mask = family_to_scenario_mask[family_ids]
    masked_logits = logits.masked_fill(~scenario_mask, -1.0e4)
    return masked_logits


def focal_cross_entropy(logits: Tensor, targets: Tensor, weight: Tensor | None = None, gamma: float = 1.5) -> Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    gathered_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    gathered_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    focal_factor = (1.0 - gathered_probs).pow(gamma)
    loss = -focal_factor * gathered_log_probs
    if weight is not None:
        loss = loss * weight[targets]
    return loss.mean()


def scenario_family_consistency_loss(cls_logits: Tensor, family_logits: Tensor, scenario_to_family: Tensor, num_families: int) -> Tensor:
    scenario_probs = torch.softmax(cls_logits, dim=1)
    aggregated = torch.zeros(cls_logits.shape[0], num_families, device=cls_logits.device, dtype=cls_logits.dtype)
    aggregated.scatter_add_(1, scenario_to_family.unsqueeze(0).expand(cls_logits.shape[0], -1), scenario_probs)
    family_probs = torch.softmax(family_logits, dim=1)
    return F.kl_div(torch.log(family_probs.clamp_min(1.0e-6)), aggregated.clamp_min(1.0e-6), reduction="batchmean")


def electrical_specialist_consistency_loss(cls_logits: Tensor, specialist_logits: Tensor) -> Tensor:
    global_logits = cls_logits[:, ELECTRICAL_SCENARIO_INDICES]
    global_probs = torch.softmax(global_logits, dim=1)
    specialist_probs = torch.softmax(specialist_logits, dim=1)
    return 0.5 * (
        F.kl_div(torch.log(global_probs.clamp_min(1.0e-6)), specialist_probs.detach().clamp_min(1.0e-6), reduction="batchmean")
        + F.kl_div(torch.log(specialist_probs.clamp_min(1.0e-6)), global_probs.detach().clamp_min(1.0e-6), reduction="batchmean")
    )


def load_specialist_consistency_loss(cls_logits: Tensor, specialist_logits: Tensor) -> Tensor:
    global_logits = cls_logits[:, LOAD_SCENARIO_INDICES]
    global_probs = torch.softmax(global_logits, dim=1)
    specialist_probs = torch.softmax(specialist_logits, dim=1)
    return 0.5 * (
        F.kl_div(torch.log(global_probs.clamp_min(1.0e-6)), specialist_probs.detach().clamp_min(1.0e-6), reduction="batchmean")
        + F.kl_div(torch.log(specialist_probs.clamp_min(1.0e-6)), global_probs.detach().clamp_min(1.0e-6), reduction="batchmean")
    )


def collate_multimodal_batch(items: list[MultimodalBatch]) -> MultimodalBatch:
    return MultimodalBatch(
        pos=torch.stack([item.pos for item in items], dim=0),
        elec=torch.stack([item.elec for item in items], dim=0),
        therm=torch.stack([item.therm for item in items], dim=0),
        vib=torch.stack([item.vib for item in items], dim=0),
        res=torch.stack([item.res for item in items], dim=0),
        freq=torch.stack([item.freq for item in items], dim=0),
        ctx=torch.stack([item.ctx for item in items], dim=0),
        y_det=torch.stack([item.y_det for item in items], dim=0),
        y_cls=torch.stack([item.y_cls for item in items], dim=0),
        y_family=torch.stack([item.y_family for item in items], dim=0),
        y_loc=torch.stack([item.y_loc for item in items], dim=0),
        y_boundary=torch.stack([item.y_boundary for item in items], dim=0),
        y_sev=torch.stack([item.y_sev for item in items], dim=0),
        y_rul=torch.stack([item.y_rul for item in items], dim=0),
    )


def move_batch_to_device(batch: MultimodalBatch, device: torch.device) -> MultimodalBatch:
    return MultimodalBatch(
        pos=batch.pos.to(device),
        elec=batch.elec.to(device),
        therm=batch.therm.to(device),
        vib=batch.vib.to(device),
        res=batch.res.to(device),
        freq=batch.freq.to(device),
        ctx=batch.ctx.to(device),
        y_det=batch.y_det.to(device),
        y_cls=batch.y_cls.to(device),
        y_family=batch.y_family.to(device),
        y_loc=batch.y_loc.to(device),
        y_boundary=batch.y_boundary.to(device),
        y_sev=batch.y_sev.to(device),
        y_rul=batch.y_rul.to(device),
    )


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _normalize_window_configs(config: WindowConfig | list[WindowConfig] | None) -> list[WindowConfig]:
    if config is None:
        return [WindowConfig(window_size=256, stride=64)]
    if isinstance(config, list):
        return config
    return [config]


def _reference_window_size(window_configs: list[WindowConfig]) -> int:
    sizes = sorted(config.window_size for config in window_configs)
    return sizes[len(sizes) // 2]


def _resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    if sequence.shape[0] == target_length:
        return sequence.astype(np.float32, copy=False)
    old_axis = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
    new_axis = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    channels = [
        np.interp(new_axis, old_axis, sequence[:, channel_index]).astype(np.float32)
        for channel_index in range(sequence.shape[1])
    ]
    return np.stack(channels, axis=1).astype(np.float32)


def _build_dct_basis(window_size: int, freq_bins: int) -> np.ndarray:
    freq_bins = min(freq_bins, window_size)
    n = np.arange(window_size, dtype=np.float32)
    k = np.arange(freq_bins, dtype=np.float32)[:, None]
    basis = np.cos((np.pi / float(window_size)) * (n + 0.5) * k)
    basis[0] *= math.sqrt(1.0 / float(window_size))
    if freq_bins > 1:
        basis[1:] *= math.sqrt(2.0 / float(window_size))
    return basis.astype(np.float32)


def build_frequency_modal_window(
    window_array: np.ndarray,
    dct_basis: np.ndarray,
    stft_frame_size: int = 64,
    stft_hop: int = 32,
) -> np.ndarray:
    centered = window_array - window_array.mean(axis=0, keepdims=True)
    dct_coeffs = np.abs(dct_basis @ centered).astype(np.float32)

    frame_size = max(16, min(int(stft_frame_size), centered.shape[0]))
    hop = max(8, min(int(stft_hop), max(1, frame_size // 2)))
    window_fn = np.hanning(frame_size).astype(np.float32)
    stft_frames: list[np.ndarray] = []
    for start in range(0, centered.shape[0] - frame_size + 1, hop):
        frame = centered[start : start + frame_size] * window_fn[:, None]
        spectrum = np.abs(np.fft.rfft(frame, axis=0)).astype(np.float32)
        stft_frames.append(spectrum)
    if not stft_frames:
        spectrum = np.abs(np.fft.rfft(centered, axis=0)).astype(np.float32)
        stft_frames.append(spectrum)
    stft_mean = np.mean(np.stack(stft_frames, axis=0), axis=0)
    target_bins = dct_basis.shape[0]
    if stft_mean.shape[0] < target_bins:
        pad = np.zeros((target_bins - stft_mean.shape[0], stft_mean.shape[1]), dtype=np.float32)
        stft_mean = np.concatenate([stft_mean, pad], axis=0)
    else:
        stft_mean = stft_mean[:target_bins]
    return np.concatenate([dct_coeffs, stft_mean], axis=1).astype(np.float32)


def _frequency_column_position(column_name: str, prefix: str = "stft") -> int:
    base_index = FREQUENCY_SOURCE_COLUMNS.index(column_name)
    if prefix == "dct":
        return base_index
    return FREQUENCY_STFT_OFFSET + base_index


def extract_bearing_targeted_features(freq: Tensor, vib: Tensor, frame_size: int = DEFAULT_STFT_FRAME_SIZE) -> Tensor:
    stft_band_idx = _frequency_column_position("vibration_band_mps2", "stft")
    stft_env_idx = _frequency_column_position("vibration_envelope_mps2", "stft")
    band_spectrum = freq[:, :, stft_band_idx]
    env_spectrum = freq[:, :, stft_env_idx]
    freq_axis = torch.arange(freq.shape[1], device=freq.device, dtype=freq.dtype) * (DEFAULT_SAMPLE_RATE_HZ / float(frame_size))
    fund_mask = (freq_axis >= 30.0) & (freq_axis <= 46.0)
    harmonic_mask = (freq_axis >= 70.0) & (freq_axis <= 86.0)

    def masked_mean(spectrum: Tensor, mask: Tensor) -> Tensor:
        masked = spectrum[:, mask]
        if masked.shape[1] == 0:
            return spectrum.new_zeros((spectrum.shape[0],))
        return masked.mean(dim=1)

    band_mean = band_spectrum.mean(dim=1)
    band_var = band_spectrum.var(dim=1, unbiased=False)
    band_centered = band_spectrum - band_mean.unsqueeze(1)
    band_kurtosis = (band_centered.pow(4).mean(dim=1) / band_var.clamp_min(1.0e-6).pow(2)).clamp_max(50.0)
    band_peak_ratio = band_spectrum.max(dim=1).values / band_mean.clamp_min(1.0e-6)

    features = torch.stack(
        [
            torch.sqrt(vib[:, :, 0].pow(2).mean(dim=1).clamp_min(1.0e-9)),
            vib[:, :, 1].abs().mean(dim=1),
            vib[:, :, 2].abs().mean(dim=1),
            torch.quantile(vib[:, :, 3], 0.95, dim=1),
            masked_mean(band_spectrum, fund_mask),
            masked_mean(band_spectrum, harmonic_mask),
            masked_mean(env_spectrum, fund_mask),
            masked_mean(env_spectrum, harmonic_mask),
            band_kurtosis,
            band_peak_ratio,
        ],
        dim=1,
    )
    return torch.log1p(features.clamp_min(0.0))


def extract_load_targeted_features(pos: Tensor, elec: Tensor, res: Tensor, ctx: Tensor) -> Tensor:
    torque_load = res[:, :, 5]
    speed_residual = res[:, :, 1]
    shaft_twist = res[:, :, 3]
    position_error = res[:, :, 0]
    current_q = elec[:, :, 2]
    load_accel = pos[:, :, 5]
    pwm_duty = ctx[:, :, 9]
    current_limit = ctx[:, :, 4]

    midpoint = max(torque_load.shape[1] // 2, 1)
    first_half_torque = torque_load[:, :midpoint].mean(dim=1)
    second_half_torque = torque_load[:, midpoint:].mean(dim=1)

    features = torch.stack(
        [
            position_error.abs().mean(dim=1),
            torch.sqrt(speed_residual.pow(2).mean(dim=1).clamp_min(1.0e-9)),
            torch.sqrt(shaft_twist.pow(2).mean(dim=1).clamp_min(1.0e-9)),
            torque_load.abs().mean(dim=1),
            torque_load.std(dim=1, unbiased=False),
            torch.sqrt(current_q.pow(2).mean(dim=1).clamp_min(1.0e-9)),
            load_accel.abs().mean(dim=1),
            pwm_duty.mean(dim=1),
            (current_limit - current_q.abs()).mean(dim=1),
            (second_half_torque - first_half_torque).abs(),
        ],
        dim=1,
    )
    return torch.log1p(features.clamp_min(0.0))


def build_derived_residual_array(run_rows: list[dict[str, str]], gear_ratio: float = 45.0) -> np.ndarray:
    values: list[list[float]] = []
    for row in run_rows:
        theta_motor = float(row["theta_motor_meas_deg"])
        theta_load = float(row["theta_meas_deg"])
        electrical_power = float(row["electrical_power_w"])
        mechanical_power = float(row["mechanical_power_w"])
        winding_temp = float(row["winding_temp_c"])
        housing_temp = float(row["housing_temp_c"])
        available_bus = float(row["available_bus_voltage_v"])
        voltage_meas = float(row["voltage_meas_v"])
        current_limit = float(row["current_limit_a"])
        current_meas = float(row["current_meas_a"])
        current_d = float(row["current_d_meas_a"])
        current_q = float(row["current_q_meas_a"])
        torque_shaft = float(row["torque_shaft_nm"])
        torque_load = float(row["torque_load_nm"])
        values.append(
            [
                theta_motor / gear_ratio - theta_load,
                current_q * float(row["back_emf_v"]),
                electrical_power - mechanical_power,
                winding_temp - housing_temp,
                available_bus - voltage_meas,
                current_limit - abs(current_meas),
                math.hypot(current_d, current_q),
                torque_shaft - torque_load,
            ]
        )
    return np.array(values, dtype=np.float32)


def _encode_labels(labels: list[str]) -> tuple[np.ndarray, list[str], dict[str, int]]:
    names = sorted(set(labels))
    mapping = {name: idx for idx, name in enumerate(names)}
    encoded = np.array([mapping[label] for label in labels], dtype=np.int64)
    return encoded, names, mapping


def build_multimodal_window_dataset(
    rows: list[dict[str, str]],
    output_path: Path,
    metadata_path: Path,
    config: WindowConfig | list[WindowConfig] | None = None,
    active_ratio_threshold: float = FAULT_ACTIVE_RATIO_THRESHOLD,
    drop_ambiguous: bool = True,
) -> DatasetArtifacts:
    window_configs = _normalize_window_configs(config)
    target_window_size = _reference_window_size(window_configs)
    grouped = group_rows_by_run(rows)

    modality_buffers: dict[str, list[np.ndarray]] = {name: [] for name in MODALITY_NAMES}
    context_buffer: list[np.ndarray] = []
    metadata_rows: list[dict[str, str | int | float]] = []
    scenario_names: list[str] = []
    family_names: list[str] = []
    location_names: list[str] = []
    boundary_names: list[str] = []
    severity_values: list[float] = []
    rul_values: list[float] = []
    det_values: list[float] = []

    for (condition_name, _scenario_name), run_rows in grouped.items():
        total = len(run_rows)
        arrays = {
            name: np.array(
                [[float(row[column]) for column in columns] for row in run_rows],
                dtype=np.float32,
            )
            for name, columns in MODALITY_COLUMN_GROUPS.items()
        }
        derived_residual_array = build_derived_residual_array(run_rows)
        arrays["res"] = np.concatenate([arrays["res"], derived_residual_array], axis=1)
        freq_source_array = np.array(
            [[float(row[column]) for column in FREQUENCY_SOURCE_COLUMNS] for row in run_rows],
            dtype=np.float32,
        )
        ctx_array = np.array(
            [[float(row[column]) for column in CONTEXT_COLUMNS] for row in run_rows],
            dtype=np.float32,
        )
        active_mask = build_fault_active_mask(run_rows)

        for window_cfg in window_configs:
            if total < window_cfg.window_size:
                continue
            dct_basis = _build_dct_basis(window_cfg.window_size, freq_bins=min(32, window_cfg.window_size))
            stft_frame = max(32, min(128, window_cfg.window_size // 2))
            stft_hop = max(16, stft_frame // 2)
            for start in range(0, total - window_cfg.window_size + 1, window_cfg.stride):
                end = start + window_cfg.window_size
                label = assign_window_label(
                    run_rows,
                    active_mask,
                    start,
                    end,
                    active_ratio_threshold=active_ratio_threshold,
                    drop_ambiguous=drop_ambiguous,
                )
                if label is None:
                    continue
                for name in MODALITY_NAMES:
                    if name == "freq":
                        freq_window = build_frequency_modal_window(
                            freq_source_array[start:end],
                            dct_basis=dct_basis,
                            stft_frame_size=stft_frame,
                            stft_hop=stft_hop,
                        )
                        modality_buffers["freq"].append(freq_window)
                    else:
                        modality_buffers[name].append(_resample_sequence(arrays[name][start:end], target_window_size))
                context_buffer.append(_resample_sequence(ctx_array[start:end], target_window_size))

                scenario = label.scenario
                family = label.fault_label
                location = LOCATION_BY_SCENARIO[scenario]
                boundary = BOUNDARY_BY_SCENARIO[scenario]
                severity_source = label.source_scenario if label.source_scenario in SEVERITY_BY_SCENARIO else scenario
                severity = float(SEVERITY_BY_SCENARIO[severity_source])
                rul = float(max(0.0, 1.0 - severity))
                det = 0.0 if family == "normal" else 1.0

                scenario_names.append(scenario)
                family_names.append(family)
                location_names.append(location)
                boundary_names.append(boundary)
                severity_values.append(severity)
                rul_values.append(rul)
                det_values.append(det)
                metadata_rows.append(
                    {
                        "condition_name": condition_name,
                        "scenario": scenario,
                        "fault_label": family,
                        "location_label": location,
                        "source_scenario": label.source_scenario,
                        "source_fault_label": label.source_fault_label,
                        "source_fault_id": label.source_fault_id,
                        "fault_active_center": int(label.fault_active_center),
                        "fault_active_ratio": label.fault_active_ratio,
                        "window_label_state": label.window_label_state,
                        "window_start": start,
                        "window_end": end,
                        "window_size": window_cfg.window_size,
                        "window_stride": window_cfg.stride,
                        "time_start_s": float(run_rows[start]["time_s"]),
                        "time_end_s": float(run_rows[end - 1]["time_s"]),
                    }
                )

    y_cls, class_names, _ = _encode_labels(scenario_names)
    y_family, family_name_list, _ = _encode_labels(family_names)
    y_loc, location_name_list, _ = _encode_labels(location_names)
    y_boundary, boundary_name_list, _ = _encode_labels(boundary_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        X_pos=np.stack(modality_buffers["pos"]).astype(np.float32),
        X_elec=np.stack(modality_buffers["elec"]).astype(np.float32),
        X_therm=np.stack(modality_buffers["therm"]).astype(np.float32),
        X_vib=np.stack(modality_buffers["vib"]).astype(np.float32),
        X_res=np.stack(modality_buffers["res"]).astype(np.float32),
        X_freq=np.stack(modality_buffers["freq"]).astype(np.float32),
        X_ctx=np.stack(context_buffer).astype(np.float32),
        y_det=np.array(det_values, dtype=np.float32),
        y_cls=y_cls,
        y_family=y_family,
        y_loc=y_loc,
        y_boundary=y_boundary,
        y_sev=np.array(severity_values, dtype=np.float32),
        y_rul=np.array(rul_values, dtype=np.float32),
        class_names=np.array(class_names, dtype=object),
        family_names=np.array(family_name_list, dtype=object),
        scenario_to_family_idx=build_scenario_family_mapping(class_names, family_name_list),
        location_names=np.array(location_name_list, dtype=object),
        boundary_names=np.array(boundary_name_list, dtype=object),
        condition_names=np.array([row["condition_name"] for row in metadata_rows], dtype=object),
        scenario_names=np.array(scenario_names, dtype=object),
        source_scenario_names=np.array([row["source_scenario"] for row in metadata_rows], dtype=object),
        fault_active_center=np.array([int(row["fault_active_center"]) for row in metadata_rows], dtype=np.int64),
        fault_active_ratio=np.array([float(row["fault_active_ratio"]) for row in metadata_rows], dtype=np.float32),
        window_label_state=np.array([row["window_label_state"] for row in metadata_rows], dtype=object),
        pos_columns=np.array(MODALITY_COLUMN_GROUPS["pos"], dtype=object),
        elec_columns=np.array(MODALITY_COLUMN_GROUPS["elec"], dtype=object),
        therm_columns=np.array(MODALITY_COLUMN_GROUPS["therm"], dtype=object),
        vib_columns=np.array(MODALITY_COLUMN_GROUPS["vib"], dtype=object),
        res_columns=np.array(MODALITY_COLUMN_GROUPS["res"] + DERIVED_RESIDUAL_FEATURE_NAMES, dtype=object),
        freq_columns=np.array(FREQUENCY_FEATURE_NAMES, dtype=object),
        ctx_columns=np.array(CONTEXT_COLUMNS, dtype=object),
        window_sizes=np.array([cfg.window_size for cfg in window_configs], dtype=np.int64),
        window_strides=np.array([cfg.stride for cfg in window_configs], dtype=np.int64),
        active_ratio_threshold=np.array([active_ratio_threshold], dtype=np.float32),
        dataset_version=np.array(["servo_multimodal_v4"], dtype=object),
    )

    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    return DatasetArtifacts(
        output_path=output_path,
        metadata_path=metadata_path,
        num_windows=len(metadata_rows),
        window_size=target_window_size,
        stride=window_configs[0].stride,
        window_sizes=[cfg.window_size for cfg in window_configs],
        strides=[cfg.stride for cfg in window_configs],
        class_names=class_names,
        family_names=family_name_list,
        location_names=location_name_list,
    )


def load_multimodal_arrays(path: Path) -> dict[str, np.ndarray]:
    npz = np.load(path, allow_pickle=True)
    arrays = {key: npz[key] for key in npz.files}
    if "scenario_to_family_idx" not in arrays and "class_names" in arrays and "family_names" in arrays:
        class_names = [str(item) for item in arrays["class_names"].tolist()]
        family_names = [str(item) for item in arrays["family_names"].tolist()]
        arrays["scenario_to_family_idx"] = build_scenario_family_mapping(class_names, family_names)
    if "boundary_names" not in arrays:
        arrays["boundary_names"] = np.array(BOUNDARY_LABEL_ORDER, dtype=object)
    if "y_boundary" not in arrays and "scenario_names" in arrays:
        boundary_index = {name: idx for idx, name in enumerate(BOUNDARY_LABEL_ORDER)}
        arrays["y_boundary"] = np.array(
            [boundary_index[BOUNDARY_BY_SCENARIO[str(name)]] for name in arrays["scenario_names"].astype(str)],
            dtype=np.int64,
        )
    return arrays


def split_indices_by_condition(arrays: dict[str, np.ndarray], holdout_condition: str) -> tuple[np.ndarray, np.ndarray]:
    condition_names = arrays["condition_names"].astype(str)
    train_idx = np.where(condition_names != holdout_condition)[0]
    test_idx = np.where(condition_names == holdout_condition)[0]
    if train_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"Invalid holdout condition: {holdout_condition}")
    return train_idx, test_idx


def compute_normalization_stats(arrays: dict[str, np.ndarray], train_idx: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key in ["X_pos", "X_elec", "X_therm", "X_vib", "X_res", "X_freq", "X_ctx"]:
        train_values = arrays[key][train_idx]
        mean = train_values.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        std = train_values.std(axis=(0, 1), keepdims=True).astype(np.float32)
        std[std < 1.0e-6] = 1.0
        stats[key] = {"mean": mean, "std": std}
    return stats


def apply_normalization(arrays: dict[str, np.ndarray], stats: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if key in stats:
            normalized[key] = ((value - stats[key]["mean"]) / stats[key]["std"]).astype(np.float32)
        else:
            normalized[key] = value
    return normalized


def build_dataloaders(
    arrays: dict[str, np.ndarray],
    holdout_condition: str,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[dict[str, dict[str, np.ndarray]], DataLoader, DataLoader, np.ndarray, np.ndarray]:
    train_idx, test_idx = split_indices_by_condition(arrays, holdout_condition)
    stats = compute_normalization_stats(arrays, train_idx)
    normalized = apply_normalization(arrays, stats)
    train_dataset = ServoMultimodalDataset(normalized, train_idx)
    test_dataset = ServoMultimodalDataset(normalized, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_multimodal_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_multimodal_batch)
    return stats, train_loader, test_loader, train_idx, test_idx


def batch_input_dims(batch: MultimodalBatch) -> dict[str, int]:
    return {
        "pos": int(batch.pos.shape[-1]),
        "elec": int(batch.elec.shape[-1]),
        "therm": int(batch.therm.shape[-1]),
        "vib": int(batch.vib.shape[-1]),
        "res": int(batch.res.shape[-1]),
        "freq": int(batch.freq.shape[-1]),
        "ctx": int(batch.ctx.shape[-1]),
    }


def multitask_loss(
    outputs: ModelOutputs,
    batch: MultimodalBatch,
    scenario_to_family: Tensor,
    family_to_scenario_mask: Tensor,
    class_weights: dict[str, Tensor] | None = None,
    lambda_phy: float = 0.10,
    lambda_align: float = 0.05,
    lambda_hier: float = 0.35,
    lambda_ctx: float = 0.10,
) -> tuple[Tensor, dict[str, float]]:
    class_weights = class_weights or {}
    det_loss = F.binary_cross_entropy_with_logits(outputs.det_logits, batch.y_det)
    true_family_conditioned_logits = condition_scenario_logits(outputs.cls_logits, batch.y_family, family_to_scenario_mask)
    cls_loss = focal_cross_entropy(true_family_conditioned_logits, batch.y_cls, weight=class_weights.get("cls"))
    family_loss = F.cross_entropy(outputs.family_logits, batch.y_family, weight=class_weights.get("family"))
    loc_loss = F.cross_entropy(outputs.loc_logits, batch.y_loc, weight=class_weights.get("loc"))
    boundary_loss = F.cross_entropy(outputs.boundary_logits, batch.y_boundary, weight=class_weights.get("boundary"))
    sev_loss = F.smooth_l1_loss(outputs.severity, batch.y_sev)
    rul_loss = F.smooth_l1_loss(outputs.rul, batch.y_rul)
    phy_loss = outputs.explainability.physics_loss
    align_loss = outputs.explainability.alignment_loss
    hier_loss = scenario_family_consistency_loss(outputs.cls_logits, outputs.family_logits, scenario_to_family, family_to_scenario_mask.shape[0])
    ctx_penalty = torch.relu(outputs.explainability.modality_importance[:, -1] - 0.22).mean()
    elec_res_importance = outputs.explainability.modality_importance[:, 1] + outputs.explainability.modality_importance[:, 4]
    therm_ctx_importance = outputs.explainability.modality_importance[:, 2] + outputs.explainability.modality_importance[:, 6]
    bearing_importance = outputs.explainability.modality_importance[:, 3] + outputs.explainability.modality_importance[:, 5]
    bearing_mask = batch.y_cls == BEARING_SCENARIO_INDEX
    if torch.any(bearing_mask):
        bearing_focus_penalty = torch.relu(0.24 - bearing_importance[bearing_mask]).mean()
    else:
        bearing_focus_penalty = outputs.cls_logits.new_zeros(())
    non_bearing_mask = ~bearing_mask
    bearing_prob = torch.softmax(outputs.cls_logits, dim=1)[:, BEARING_SCENARIO_INDEX]
    if torch.any(non_bearing_mask):
        bearing_false_attractor_penalty = (
            bearing_prob[non_bearing_mask] * torch.relu(0.14 - bearing_importance[non_bearing_mask])
        ).mean()
    else:
        bearing_false_attractor_penalty = outputs.cls_logits.new_zeros(())
    electrical_mask = batch.y_family == ELECTRICAL_FAMILY_INDEX
    if torch.any(electrical_mask):
        electrical_targets = batch.y_cls[electrical_mask].clone()
        electrical_local_targets = torch.empty_like(electrical_targets)
        for global_index, local_index in ELECTRICAL_GLOBAL_TO_LOCAL.items():
            electrical_local_targets[electrical_targets == global_index] = local_index
        electrical_specialist_loss = F.cross_entropy(outputs.electro_thermal_logits[electrical_mask], electrical_local_targets)
        electrical_consistency_loss = electrical_specialist_consistency_loss(
            outputs.cls_logits[electrical_mask],
            outputs.electro_thermal_logits[electrical_mask],
        )
        electrical_target_logits = outputs.electro_thermal_logits[electrical_mask].gather(1, electrical_local_targets.unsqueeze(1)).squeeze(1)
        exclusion_mask = F.one_hot(electrical_local_targets, num_classes=len(ELECTRO_THERMAL_SCENARIOS)).bool()
        electrical_other_logits = outputs.electro_thermal_logits[electrical_mask].masked_fill(exclusion_mask, -1.0e9)
        electrical_exclusion_loss = torch.relu(0.45 - (electrical_target_logits - electrical_other_logits.max(dim=1).values)).mean()
    else:
        electrical_specialist_loss = outputs.cls_logits.new_zeros(())
        electrical_consistency_loss = outputs.cls_logits.new_zeros(())
        electrical_exclusion_loss = outputs.cls_logits.new_zeros(())
    load_mask = batch.y_family == LOAD_FAMILY_INDEX
    load_presence_target = load_mask.float()
    load_presence_loss = F.binary_cross_entropy_with_logits(outputs.load_presence_logit, load_presence_target)
    if torch.any(load_mask):
        load_targets = batch.y_cls[load_mask].clone()
        load_local_targets = torch.empty_like(load_targets)
        for global_index, local_index in LOAD_GLOBAL_TO_LOCAL.items():
            load_local_targets[load_targets == global_index] = local_index
        load_severity_loss = F.cross_entropy(outputs.load_severity_logits[load_mask], load_local_targets)
        load_consistency_loss = load_specialist_consistency_loss(
            outputs.cls_logits[load_mask],
            outputs.load_severity_logits[load_mask],
        )
        load_target_logits = outputs.load_severity_logits[load_mask].gather(1, load_local_targets.unsqueeze(1)).squeeze(1)
        load_exclusion_mask = F.one_hot(load_local_targets, num_classes=2).bool()
        load_other_logits = outputs.load_severity_logits[load_mask].masked_fill(load_exclusion_mask, -1.0e9)
        load_exclusion_loss = torch.relu(0.35 - (load_target_logits - load_other_logits.max(dim=1).values)).mean()
        load_primary = (
            outputs.explainability.modality_importance[load_mask, 0]
            + outputs.explainability.modality_importance[load_mask, 1]
            + outputs.explainability.modality_importance[load_mask, 4]
        )
        load_secondary = outputs.explainability.modality_importance[load_mask, 2] + outputs.explainability.modality_importance[load_mask, 6]
        load_modality_penalty = (
            torch.relu(0.44 - load_primary).mean()
            + 0.35 * torch.relu(load_secondary - (load_primary - 0.08)).mean()
        )
    else:
        load_severity_loss = outputs.cls_logits.new_zeros(())
        load_consistency_loss = outputs.cls_logits.new_zeros(())
        load_exclusion_loss = outputs.cls_logits.new_zeros(())
        load_modality_penalty = outputs.cls_logits.new_zeros(())
    partial_demag_mask = batch.y_cls == PARTIAL_DEMAGNETIZATION_SCENARIO_INDEX
    if torch.any(partial_demag_mask):
        partial_primary = elec_res_importance[partial_demag_mask] + 0.5 * outputs.explainability.modality_importance[partial_demag_mask, 5]
        partial_secondary = therm_ctx_importance[partial_demag_mask]
        partial_demag_modality_penalty = (
            torch.relu(0.40 - partial_primary).mean()
            + 0.50 * torch.relu(partial_secondary - (partial_primary - 0.06)).mean()
        )
    else:
        partial_demag_modality_penalty = outputs.cls_logits.new_zeros(())
    inverter_mask = batch.y_cls == INVERTER_VOLTAGE_LOSS_SCENARIO_INDEX
    if torch.any(inverter_mask):
        inverter_primary = elec_res_importance[inverter_mask] + 0.35 * outputs.explainability.modality_importance[inverter_mask, 0]
        inverter_secondary = outputs.explainability.modality_importance[inverter_mask, 3] + outputs.explainability.modality_importance[inverter_mask, 6]
        inverter_modality_penalty = (
            torch.relu(0.38 - inverter_primary).mean()
            + 0.40 * torch.relu(inverter_secondary - (inverter_primary - 0.04)).mean()
        )
    else:
        inverter_modality_penalty = outputs.cls_logits.new_zeros(())
    bearing_specialist_target = bearing_mask.float()
    bearing_specialist_loss = F.binary_cross_entropy_with_logits(
        outputs.bearing_specialist_logit,
        bearing_specialist_target,
    )
    total = (
        0.2 * det_loss
        + 1.2 * cls_loss
        + 1.0 * family_loss
        + 0.6 * loc_loss
        + 0.9 * boundary_loss
        + 0.35 * sev_loss
        + 0.15 * rul_loss
        + lambda_phy * phy_loss
        + lambda_align * align_loss
        + lambda_hier * hier_loss
        + lambda_ctx * ctx_penalty
        + 0.55 * electrical_specialist_loss
        + 0.25 * electrical_consistency_loss
        + 0.20 * electrical_exclusion_loss
        + 0.15 * load_presence_loss
        + 0.20 * load_severity_loss
        + 0.08 * load_consistency_loss
        + 0.05 * load_exclusion_loss
        + 0.05 * load_modality_penalty
        + 0.18 * partial_demag_modality_penalty
        + 0.16 * inverter_modality_penalty
        + 0.60 * bearing_specialist_loss
        + 0.30 * bearing_focus_penalty
        + 0.15 * bearing_false_attractor_penalty
    )
    parts = {
        "det_loss": float(det_loss.detach().cpu().item()),
        "cls_loss": float(cls_loss.detach().cpu().item()),
        "family_loss": float(family_loss.detach().cpu().item()),
        "loc_loss": float(loc_loss.detach().cpu().item()),
        "boundary_loss": float(boundary_loss.detach().cpu().item()),
        "sev_loss": float(sev_loss.detach().cpu().item()),
        "rul_loss": float(rul_loss.detach().cpu().item()),
        "phy_loss": float(phy_loss.detach().cpu().item()),
        "align_loss": float(align_loss.detach().cpu().item()),
        "hier_loss": float(hier_loss.detach().cpu().item()),
        "ctx_penalty": float(ctx_penalty.detach().cpu().item()),
        "electrical_specialist_loss": float(electrical_specialist_loss.detach().cpu().item()),
        "electrical_consistency_loss": float(electrical_consistency_loss.detach().cpu().item()),
        "electrical_exclusion_loss": float(electrical_exclusion_loss.detach().cpu().item()),
        "load_presence_loss": float(load_presence_loss.detach().cpu().item()),
        "load_severity_loss": float(load_severity_loss.detach().cpu().item()),
        "load_consistency_loss": float(load_consistency_loss.detach().cpu().item()),
        "load_exclusion_loss": float(load_exclusion_loss.detach().cpu().item()),
        "load_modality_penalty": float(load_modality_penalty.detach().cpu().item()),
        "partial_demag_modality_penalty": float(partial_demag_modality_penalty.detach().cpu().item()),
        "inverter_modality_penalty": float(inverter_modality_penalty.detach().cpu().item()),
        "bearing_specialist_loss": float(bearing_specialist_loss.detach().cpu().item()),
        "bearing_focus_penalty": float(bearing_focus_penalty.detach().cpu().item()),
        "bearing_false_attractor_penalty": float(bearing_false_attractor_penalty.detach().cpu().item()),
    }
    return total, parts


def make_class_weights(labels: np.ndarray, num_classes: int) -> Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float32)
    counts[counts < 1.0] = 1.0
    weights = counts.sum() / (counts * float(num_classes))
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: ServoReasoningNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    class_weights: dict[str, Tensor],
) -> TrainingMetrics:
    model.train()
    totals = {"loss": 0.0, "det": 0.0, "cls": 0.0, "family": 0.0, "loc": 0.0, "boundary": 0.0, "sev": 0.0, "rul": 0.0, "samples": 0.0}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        outputs = model(batch)
        loss, _ = multitask_loss(outputs, batch, model.scenario_to_family, model.family_to_scenario_mask, class_weights)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
        optimizer.step()
        update_metric_totals(totals, outputs, batch, float(loss.detach().cpu().item()))
    return finalize_metrics(totals)


def evaluate_model(
    model: ServoReasoningNet,
    loader: DataLoader,
    device: torch.device,
    class_weights: dict[str, Tensor],
) -> TrainingMetrics:
    model.eval()
    totals = {"loss": 0.0, "det": 0.0, "cls": 0.0, "family": 0.0, "loc": 0.0, "boundary": 0.0, "sev": 0.0, "rul": 0.0, "samples": 0.0}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss, _ = multitask_loss(outputs, batch, model.scenario_to_family, model.family_to_scenario_mask, class_weights)
            update_metric_totals(totals, outputs, batch, float(loss.detach().cpu().item()))
    return finalize_metrics(totals)


def update_metric_totals(totals: dict[str, float], outputs: ModelOutputs, batch: MultimodalBatch, loss_value: float) -> None:
    batch_size = float(batch.y_cls.shape[0])
    det_pred = (torch.sigmoid(outputs.det_logits) >= 0.5).float()
    cls_pred = outputs.cls_logits_fused.argmax(dim=1)
    family_pred = outputs.family_logits.argmax(dim=1)
    loc_pred = outputs.loc_logits.argmax(dim=1)
    boundary_pred = outputs.boundary_logits.argmax(dim=1)
    totals["loss"] += loss_value * batch_size
    totals["det"] += float((det_pred == batch.y_det).sum().detach().cpu().item())
    totals["cls"] += float((cls_pred == batch.y_cls).sum().detach().cpu().item())
    totals["family"] += float((family_pred == batch.y_family).sum().detach().cpu().item())
    totals["loc"] += float((loc_pred == batch.y_loc).sum().detach().cpu().item())
    totals["boundary"] += float((boundary_pred == batch.y_boundary).sum().detach().cpu().item())
    totals["sev"] += float(torch.abs(outputs.severity - batch.y_sev).sum().detach().cpu().item())
    totals["rul"] += float(torch.abs(outputs.rul - batch.y_rul).sum().detach().cpu().item())
    totals["samples"] += batch_size


def finalize_metrics(totals: dict[str, float]) -> TrainingMetrics:
    samples = max(totals["samples"], 1.0)
    return TrainingMetrics(
        det_accuracy=totals["det"] / samples,
        cls_accuracy=totals["cls"] / samples,
        family_accuracy=totals["family"] / samples,
        loc_accuracy=totals["loc"] / samples,
        boundary_accuracy=totals["boundary"] / samples,
        severity_mae=totals["sev"] / samples,
        rul_mae=totals["rul"] / samples,
        total_loss=totals["loss"] / samples,
    )


def save_checkpoint(
    path: Path,
    model: ServoReasoningNet,
    optimizer: torch.optim.Optimizer,
    stats: dict[str, dict[str, np.ndarray]],
    arrays: dict[str, np.ndarray],
    holdout_condition: str,
    epoch: int,
    metrics: TrainingMetrics,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "stats": stats,
        "holdout_condition": holdout_condition,
        "epoch": epoch,
        "metrics": asdict(metrics),
        "class_names": arrays["class_names"].tolist(),
        "family_names": arrays["family_names"].tolist(),
        "scenario_to_family_idx": arrays["scenario_to_family_idx"].tolist(),
        "location_names": arrays["location_names"].tolist(),
        "boundary_names": arrays["boundary_names"].tolist(),
        "input_dims": {
            "pos": int(arrays["X_pos"].shape[-1]),
            "elec": int(arrays["X_elec"].shape[-1]),
            "therm": int(arrays["X_therm"].shape[-1]),
            "vib": int(arrays["X_vib"].shape[-1]),
            "res": int(arrays["X_res"].shape[-1]),
            "freq": int(arrays["X_freq"].shape[-1]),
            "ctx": int(arrays["X_ctx"].shape[-1]),
        },
        "columns": {
            "pos": arrays["pos_columns"].tolist(),
            "elec": arrays["elec_columns"].tolist(),
            "therm": arrays["therm_columns"].tolist(),
            "vib": arrays["vib_columns"].tolist(),
            "res": arrays["res_columns"].tolist(),
            "freq": arrays["freq_columns"].tolist(),
            "ctx": arrays["ctx_columns"].tolist(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[ServoReasoningNet, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "scenario_to_family_idx" not in payload:
        payload["scenario_to_family_idx"] = build_scenario_family_mapping(payload["class_names"], payload["family_names"]).tolist()
    input_dims = {key: int(value) for key, value in payload["input_dims"].items()}
    model = ServoReasoningNet(
        input_dims=input_dims,
        num_classes=len(payload["class_names"]),
        num_families=len(payload["family_names"]),
        num_locations=len(payload["location_names"]),
        scenario_to_family=payload["scenario_to_family_idx"],
        num_boundary_classes=len(payload.get("boundary_names", BOUNDARY_LABEL_ORDER)),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload


def load_knowledge_base(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def topk_channel_events(
    window_map: dict[str, np.ndarray],
    channel_weights: dict[str, np.ndarray],
    columns: dict[str, list[str]],
    time_range: tuple[float, float],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for modality in MODALITY_NAMES:
        weights = channel_weights[modality]
        window = window_map[modality]
        top_indices = np.argsort(weights)[::-1][: min(top_k, len(weights))]
        for index in top_indices:
            series = window[:, index]
            mean = float(np.mean(series))
            std = float(np.std(series))
            energy = float(np.sqrt(np.mean(np.square(series))) + 1.0e-6)
            peak = float(np.max(np.abs(series)))
            if modality == "freq":
                peak_bin = int(np.argmax(np.abs(series)))
                score = float(abs(weights[index]) * ((peak / energy) + (std / energy)))
                events.append(
                    {
                        "signal": columns[modality][int(index)],
                        "trend": "spectral_peak",
                        "interval": [float(time_range[0]), float(time_range[1])],
                        "score": score,
                        "description": f"{columns[modality][int(index)]} shows spectral concentration near bin={peak_bin} with peak={peak:.3f}, mean={mean:.3f}, std={std:.3f}",
                    }
                )
            else:
                slope = float(series[-1] - series[0])
                level_score = abs(mean) / energy
                var_score = std / energy
                slope_score = abs(slope) / (energy + 1.0)
                trend = "increase" if slope >= 0.0 else "decrease"
                score = float(abs(weights[index]) * (level_score + var_score + slope_score))
                events.append(
                    {
                        "signal": columns[modality][int(index)],
                        "trend": trend,
                        "interval": [float(time_range[0]), float(time_range[1])],
                        "score": score,
                        "description": f"{columns[modality][int(index)]} shows {trend} with mean={mean:.3f}, std={std:.3f}, slope={slope:.3f}",
                    }
                )
    events.sort(key=lambda item: item["score"], reverse=True)
    return events[:top_k]


def retrieve_knowledge(
    knowledge_base: dict[str, Any],
    family_name: str,
    scenario_name: str,
    location_name: str,
    events: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    items = knowledge_base.get("fault_entries", [])
    event_signals = {str(item["signal"]) for item in events}
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in items:
        score = 0.0
        if item.get("family") == family_name:
            score += 2.0
        if item.get("scenario") == scenario_name:
            score += 3.0
        if item.get("location") == location_name:
            score += 1.5
        score += sum(0.3 for signal in item.get("signals", []) if signal in event_signals)
        if score > 0.0:
            scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [item for _, item in scored[:top_k]]


def render_reasoning_output(
    scenario_name: str,
    family_name: str,
    location_name: str,
    severity_value: float,
    events: list[dict[str, Any]],
    retrieved_items: list[dict[str, Any]],
) -> InferenceExplanation:
    if retrieved_items:
        top = retrieved_items[0]
        mechanism = str(top.get("mechanism", "No mechanism summary available."))
        maintenance = [str(item) for item in top.get("maintenance", [])]
    else:
        mechanism = f"Predicted {family_name} centered on {location_name}; evidence remains limited and should be verified against waveform trends."
        maintenance = ["Inspect the top-ranked abnormal channels.", "Cross-check the prediction with holdout-condition statistics."]

    evidence_chain = [f"Predicted scenario: {scenario_name}", f"Predicted family: {family_name}", f"Predicted location: {location_name}"]
    evidence_chain.extend(event["description"] for event in events[:3])
    qa_hint = f"Ask the assistant to explain why {events[0]['signal']} was ranked highest." if events else "Ask for an evidence review by modality."
    diagnosis = f"{scenario_name} ({family_name}) with estimated severity {severity_value:.2f} on {location_name}."
    return InferenceExplanation(
        diagnosis=diagnosis,
        evidence_chain=evidence_chain,
        mechanism=mechanism,
        maintenance=maintenance,
        qa_hint=qa_hint,
        events=events,
        retrieved_items=retrieved_items,
    )
