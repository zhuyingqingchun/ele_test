from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from experiments_smoke_20260316.modality_evidence_topk_v5 import (
    BRANCH_NAMES,
    PRIMARY_MODALITY_BY_SCENARIO,
    SUPPORT_MODALITIES_BY_SCENARIO,
    canonical_scenario_name,
)

TOKEN_GROUP_LENGTHS = [20, 24, 8, 16]


def build_token_group_slices(lengths: list[int] | None = None) -> list[slice]:
    lengths = lengths or TOKEN_GROUP_LENGTHS
    slices: list[slice] = []
    start = 0
    for ln in lengths:
        slices.append(slice(start, start + int(ln)))
        start += int(ln)
    return slices


def expand_modality_mask_to_tokens(mask: Tensor, lengths: list[int] | None = None) -> Tensor:
    lengths = lengths or TOKEN_GROUP_LENGTHS
    chunks = [mask[:, i:i + 1].expand(-1, int(lengths[i])) for i in range(mask.shape[1])]
    return torch.cat(chunks, dim=1)


def token_transport_plan(signal_tokens: Tensor, evidence_emb: Tensor, mechanism_emb: Tensor, temperature: float = 0.07) -> Tensor:
    views = torch.stack([evidence_emb, mechanism_emb], dim=1)  # [B, 2, D]
    signal_tokens = F.normalize(signal_tokens, dim=-1)
    views = F.normalize(views, dim=-1)
    logits = torch.einsum("btd,bvd->btv", signal_tokens, views) / max(float(temperature), 1.0e-6)
    return torch.softmax(logits, dim=1)


def aggregate_token_mass_by_modality(token_plan: Tensor, lengths: list[int] | None = None) -> Tensor:
    lengths = lengths or TOKEN_GROUP_LENGTHS
    masses = []
    start = 0
    for ln in lengths:
        masses.append(token_plan[:, start:start + int(ln), :].sum(dim=1))
        start += int(ln)
    return torch.stack(masses, dim=1)  # [B, M, 2]


def token_primary_support_loss(
    token_plan: Tensor,
    primary_mask: Tensor,
    support_mask: Tensor,
    *,
    lambda_support: float = 0.5,
    lambda_sparse: float = 0.05,
) -> tuple[Tensor, Tensor]:
    token_primary = expand_modality_mask_to_tokens(primary_mask)
    token_support = expand_modality_mask_to_tokens(support_mask)
    token_other = 1.0 - torch.clamp(token_primary + token_support, max=1.0)

    # evidence view: primary should dominate
    evidence_primary_mass = (token_plan[:, :, 0] * token_primary).sum(dim=1)
    evidence_other_mass = (token_plan[:, :, 0] * token_other).sum(dim=1)

    # mechanism view: primary + support should dominate
    mechanism_primary_support_mass = (token_plan[:, :, 1] * torch.clamp(token_primary + token_support, max=1.0)).sum(dim=1)
    mechanism_other_mass = (token_plan[:, :, 1] * token_other).sum(dim=1)

    main_loss = (
        -torch.log(evidence_primary_mass.clamp_min(1.0e-6))
        -lambda_support * torch.log(mechanism_primary_support_mass.clamp_min(1.0e-6))
        + 0.2 * F.relu(0.10 - (evidence_primary_mass - evidence_other_mass))
        + 0.2 * F.relu(0.10 - (mechanism_primary_support_mass - mechanism_other_mass))
    ).mean()

    entropy = -(token_plan.clamp_min(1.0e-8) * torch.log(token_plan.clamp_min(1.0e-8))).sum(dim=1).mean()
    total = main_loss + lambda_sparse * entropy
    return total, entropy


def summarize_token_alignment_rows(rows: list[dict]) -> dict:
    fault_rows = [r for r in rows if canonical_scenario_name(r["scenario"]) != "normal"]
    if not fault_rows:
        return {}

    def mean_bool(key: str) -> float:
        return sum(1.0 if r.get(key) else 0.0 for r in fault_rows) / len(fault_rows)

    def mean_float(key: str) -> float:
        return sum(float(r.get(key, 0.0)) for r in fault_rows) / len(fault_rows)

    return {
        "fault_only_count": len(fault_rows),
        "evidence_primary_in_top2": mean_bool("evidence_primary_in_top2"),
        "evidence_primary_in_top3": mean_bool("evidence_primary_in_top3"),
        "evidence_primary_or_support_at3": mean_bool("evidence_primary_or_support_at3"),
        "evidence_weighted_consistency_at3": mean_float("evidence_weighted_consistency_at3"),
        "mechanism_primary_in_top2": mean_bool("mechanism_primary_in_top2"),
        "mechanism_primary_in_top3": mean_bool("mechanism_primary_in_top3"),
        "mechanism_primary_or_support_at3": mean_bool("mechanism_primary_or_support_at3"),
        "mechanism_weighted_consistency_at3": mean_float("mechanism_weighted_consistency_at3"),
        "mean_evidence_mass_position": mean_float("evidence_mass_position"),
        "mean_evidence_mass_electrical": mean_float("evidence_mass_electrical"),
        "mean_evidence_mass_thermal": mean_float("evidence_mass_thermal"),
        "mean_evidence_mass_vibration": mean_float("evidence_mass_vibration"),
        "mean_mechanism_mass_position": mean_float("mechanism_mass_position"),
        "mean_mechanism_mass_electrical": mean_float("mechanism_mass_electrical"),
        "mean_mechanism_mass_thermal": mean_float("mechanism_mass_thermal"),
        "mean_mechanism_mass_vibration": mean_float("mechanism_mass_vibration"),
    }


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
