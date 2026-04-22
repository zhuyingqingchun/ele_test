from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor


BRANCH_NAMES = ["position", "electrical", "thermal", "vibration"]

PRIMARY_MODALITY_BY_SCENARIO: dict[str, str] = {
    "normal": "balanced",
    "load_disturbance_severe": "position",
    "load_disturbance_mild": "position",
    "position_sensor_bias": "position",
    "speed_sensor_scale": "position",
    "motor_encoder_freeze": "position",
    "backlash_growth": "position",
    "jam_fault": "position",
    "intermittent_jam_fault": "position",
    "bearing_defect": "vibration",
    "winding_resistance_rise": "electrical",
    "bus_voltage_sag_fault": "electrical",
    "inverter_voltage_loss": "electrical",
    "partial_demagnetization": "electrical",
    "current_sensor_bias": "electrical",
    "thermal_saturation": "thermal",
    "friction_wear_mild": "position",
    "friction_wear_severe": "position",
}

SUPPORT_MODALITIES_BY_SCENARIO: dict[str, list[str]] = {
    "normal": BRANCH_NAMES,
    "load_disturbance_severe": ["electrical"],
    "load_disturbance_mild": ["electrical"],
    "position_sensor_bias": ["electrical"],
    "speed_sensor_scale": ["electrical"],
    "motor_encoder_freeze": ["electrical"],
    "backlash_growth": ["vibration", "electrical"],
    "jam_fault": ["electrical"],
    "intermittent_jam_fault": ["electrical", "vibration"],
    "bearing_defect": ["electrical"],
    "winding_resistance_rise": ["thermal"],
    "bus_voltage_sag_fault": ["position"],
    "inverter_voltage_loss": ["position"],
    "partial_demagnetization": ["thermal", "position"],
    "current_sensor_bias": ["position"],
    "thermal_saturation": ["electrical"],
    "friction_wear_mild": ["electrical"],
    "friction_wear_severe": ["electrical", "vibration"],
}

CONFUSION_HARD_NEGATIVES: dict[str, list[str]] = {
    "friction_wear_mild": ["friction_wear_severe", "backlash_growth", "load_disturbance_mild"],
    "friction_wear_severe": ["friction_wear_mild", "jam_fault", "bearing_defect"],
    "current_sensor_bias": ["winding_resistance_rise", "inverter_voltage_loss", "partial_demagnetization"],
    "partial_demagnetization": ["inverter_voltage_loss", "winding_resistance_rise", "thermal_saturation"],
    "bearing_defect": ["backlash_growth", "friction_wear_severe", "intermittent_jam_fault"],
    "backlash_growth": ["bearing_defect", "friction_wear_mild", "jam_fault"],
}


def canonical_scenario_name(text: str) -> str:
    return str(text).strip().lower().replace(" ", "_")


def _mask_from_names(names: list[str]) -> Tensor:
    mask = torch.zeros(len(BRANCH_NAMES), dtype=torch.float32)
    for name in names:
        if name in BRANCH_NAMES:
            mask[BRANCH_NAMES.index(name)] = 1.0
    return mask


def primary_support_masks(scenario: str) -> tuple[Tensor, Tensor]:
    scenario = canonical_scenario_name(scenario)
    primary = PRIMARY_MODALITY_BY_SCENARIO.get(scenario, "balanced")
    if primary == "balanced":
        primary_mask = torch.ones(len(BRANCH_NAMES), dtype=torch.float32)
    else:
        primary_mask = _mask_from_names([primary])
    support_mask = _mask_from_names(SUPPORT_MODALITIES_BY_SCENARIO.get(scenario, []))
    return primary_mask, support_mask


def batch_primary_support_masks(scenarios: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
    primary, support = [], []
    for scenario in scenarios:
        p, s = primary_support_masks(scenario)
        primary.append(p)
        support.append(s)
    return torch.stack(primary, dim=0).to(device), torch.stack(support, dim=0).to(device)


def listwise_topk_loss(modality_scores: Tensor, primary_mask: Tensor, support_mask: Tensor, *, margin_primary: float = 0.20, margin_support: float = 0.05) -> Tensor:
    primary_scores = (modality_scores * primary_mask).sum(dim=1) / primary_mask.sum(dim=1).clamp_min(1.0)
    support_scores = (modality_scores * support_mask).sum(dim=1) / support_mask.sum(dim=1).clamp_min(1.0)
    other_mask = (1.0 - torch.clamp(primary_mask + support_mask, max=1.0))
    other_scores = (modality_scores * other_mask).sum(dim=1) / other_mask.sum(dim=1).clamp_min(1.0)
    return (F.relu(margin_primary - (primary_scores - other_scores)) + 0.5 * F.relu(margin_support - (support_scores - other_scores))).mean()


def contrast_separation_loss(signal_emb: Tensor, positive_text_emb: Tensor, negative_text_emb: Tensor, margin: float = 0.10) -> Tensor:
    pos = F.cosine_similarity(signal_emb, positive_text_emb, dim=-1)
    neg = F.cosine_similarity(signal_emb, negative_text_emb, dim=-1)
    return F.relu(margin - (pos - neg)).mean()


class SoftPrototypeBank:
    def __init__(self, num_classes: int, embed_dim: int, device: torch.device, momentum: float = 0.97) -> None:
        self.device = device
        self.momentum = float(momentum)
        self.bank = {
            "signal": torch.zeros(num_classes, embed_dim, device=device),
            "evidence": torch.zeros(num_classes, embed_dim, device=device),
            "mechanism": torch.zeros(num_classes, embed_dim, device=device),
        }
        self.counts = torch.zeros(num_classes, device=device)

    @torch.no_grad()
    def update(self, labels: Tensor, signal_emb: Tensor, evidence_emb: Tensor, mechanism_emb: Tensor) -> None:
        labels = labels.long()
        for cls in labels.unique():
            idx = labels == cls
            cls_i = int(cls.item())
            sig = F.normalize(signal_emb[idx].mean(dim=0), dim=-1)
            evd = F.normalize(evidence_emb[idx].mean(dim=0), dim=-1)
            mech = F.normalize(mechanism_emb[idx].mean(dim=0), dim=-1)
            if self.counts[cls_i] <= 0:
                self.bank["signal"][cls_i] = sig
                self.bank["evidence"][cls_i] = evd
                self.bank["mechanism"][cls_i] = mech
            else:
                m = self.momentum
                self.bank["signal"][cls_i] = F.normalize(m * self.bank["signal"][cls_i] + (1.0 - m) * sig, dim=-1)
                self.bank["evidence"][cls_i] = F.normalize(m * self.bank["evidence"][cls_i] + (1.0 - m) * evd, dim=-1)
                self.bank["mechanism"][cls_i] = F.normalize(m * self.bank["mechanism"][cls_i] + (1.0 - m) * mech, dim=-1)
            self.counts[cls_i] += idx.sum()

    def prototype_alignment_loss(self, labels: Tensor, signal_emb: Tensor, evidence_emb: Tensor, mechanism_emb: Tensor) -> Tensor:
        labels = labels.long()
        if not (self.counts[labels] > 0).any():
            return signal_emb.new_zeros(())
        sig_proto = self.bank["signal"][labels]
        evd_proto = self.bank["evidence"][labels]
        mech_proto = self.bank["mechanism"][labels]
        losses = [
            1.0 - F.cosine_similarity(signal_emb, sig_proto, dim=-1).mean(),
            1.0 - F.cosine_similarity(evidence_emb, evd_proto, dim=-1).mean(),
            1.0 - F.cosine_similarity(mechanism_emb, mech_proto, dim=-1).mean(),
        ]
        return torch.stack(losses).mean()


def confusion_aware_negative_mask(labels: Tensor, class_names: list[str]) -> Tensor:
    names = [canonical_scenario_name(class_names[int(x)]) for x in labels.detach().cpu().tolist()]
    mask = torch.zeros((len(names), len(names)), dtype=torch.bool)
    for i, scenario in enumerate(names):
        hard_set = set(CONFUSION_HARD_NEGATIVES.get(scenario, []))
        for j, other in enumerate(names):
            if i != j and other in hard_set:
                mask[i, j] = True
    return mask


def hard_negative_text_loss(signal_emb: Tensor, text_emb: Tensor, labels: Tensor, class_names: list[str], margin: float = 0.05) -> Tensor:
    if signal_emb.shape[0] <= 1:
        return signal_emb.new_zeros(())
    sim = signal_emb @ text_emb.t()
    pos = sim.diag()
    mask = confusion_aware_negative_mask(labels, class_names).to(signal_emb.device)
    losses = []
    for i in range(sim.shape[0]):
        if mask[i].any():
            hard_neg = sim[i][mask[i]].max()
            losses.append(F.relu(margin - (pos[i] - hard_neg)))
    return torch.stack(losses).mean() if losses else signal_emb.new_zeros(())


def topk_modalities(scores: Tensor, k: int = 3) -> list[list[str]]:
    k = min(k, scores.shape[1])
    idx = scores.topk(k=k, dim=1).indices.detach().cpu().tolist()
    return [[BRANCH_NAMES[j] for j in row] for row in idx]


def weighted_consistency_at3(top3: list[str], primary: str, support: list[str]) -> float:
    weights = [1.0, 0.6, 0.3]
    total = 0.0
    for rank, name in enumerate(top3[:3]):
        if name == primary:
            total += weights[rank]
        elif name in support:
            total += 0.5 * weights[rank]
    return total


def summarize_topk_rows(rows: list[dict]) -> dict:
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
    }


def save_summary_json(path: Path, summary: dict) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
