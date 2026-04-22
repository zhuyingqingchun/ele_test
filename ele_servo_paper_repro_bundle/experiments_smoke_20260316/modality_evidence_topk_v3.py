
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BRANCH_NAMES = ["position", "electrical", "thermal", "vibration"]
BRANCH_TO_INDEX = {name: idx for idx, name in enumerate(BRANCH_NAMES)}

PRIMARY_MODALITY_BY_SCENARIO: dict[str, str] = {
    "normal": "balanced",
    "load_disturbance_mild": "position",
    "load_disturbance_severe": "position",
    "friction_wear_mild": "position",
    "friction_wear_severe": "position",
    "jam_fault": "position",
    "intermittent_jam_fault": "position",
    "current_sensor_bias": "electrical",
    "speed_sensor_scale": "position",
    "position_sensor_bias": "position",
    "winding_resistance_rise": "electrical",
    "bus_voltage_sag_fault": "electrical",
    "backlash_growth": "position",
    "thermal_saturation": "thermal",
    "motor_encoder_freeze": "position",
    "partial_demagnetization": "electrical",
    "inverter_voltage_loss": "electrical",
    "bearing_defect": "vibration",
}

SUPPORT_MODALITIES_BY_SCENARIO: dict[str, list[str]] = {
    "normal": BRANCH_NAMES,
    "load_disturbance_mild": ["electrical"],
    "load_disturbance_severe": ["electrical"],
    "friction_wear_mild": ["electrical"],
    "friction_wear_severe": ["electrical", "vibration"],
    "jam_fault": ["electrical"],
    "intermittent_jam_fault": ["electrical", "vibration"],
    "current_sensor_bias": ["position"],
    "speed_sensor_scale": ["electrical"],
    "position_sensor_bias": ["electrical"],
    "winding_resistance_rise": ["thermal"],
    "bus_voltage_sag_fault": ["position"],
    "backlash_growth": ["vibration", "electrical"],
    "thermal_saturation": ["electrical"],
    "motor_encoder_freeze": ["electrical"],
    "partial_demagnetization": ["thermal", "position"],
    "inverter_voltage_loss": ["position"],
    "bearing_defect": ["electrical"],
}

CONFUSION_AWARE_HARD_NEGATIVES: dict[str, list[str]] = {
    "friction_wear_mild": ["friction_wear_severe", "backlash_growth", "load_disturbance_mild"],
    "friction_wear_severe": ["friction_wear_mild", "jam_fault", "backlash_growth"],
    "current_sensor_bias": ["winding_resistance_rise", "bus_voltage_sag_fault", "position_sensor_bias"],
    "partial_demagnetization": ["inverter_voltage_loss", "winding_resistance_rise"],
    "bearing_defect": ["backlash_growth", "friction_wear_severe"],
    "bus_voltage_sag_fault": ["inverter_voltage_loss", "partial_demagnetization"],
}

def normalize_scenario_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")

def primary_index_for_scenario(name: str) -> int | None:
    key = normalize_scenario_name(name)
    primary = PRIMARY_MODALITY_BY_SCENARIO.get(key)
    if primary is None or primary == "balanced":
        return None
    return BRANCH_TO_INDEX[primary]

def support_indices_for_scenario(name: str) -> list[int]:
    key = normalize_scenario_name(name)
    return [BRANCH_TO_INDEX[m] for m in SUPPORT_MODALITIES_BY_SCENARIO.get(key, []) if m in BRANCH_TO_INDEX]

def target_distribution_for_scenario(
    name: str,
    *,
    evidence_primary_weight: float = 0.75,
    mechanism_primary_weight: float = 0.55,
    view: str = "evidence",
) -> list[float]:
    key = normalize_scenario_name(name)
    if key == "normal":
        return [1.0 / len(BRANCH_NAMES)] * len(BRANCH_NAMES)
    primary = primary_index_for_scenario(key)
    supports = [idx for idx in support_indices_for_scenario(key) if idx != primary]
    if primary is None:
        return [1.0 / len(BRANCH_NAMES)] * len(BRANCH_NAMES)
    primary_weight = evidence_primary_weight if view == "evidence" else mechanism_primary_weight
    primary_weight = float(max(0.05, min(primary_weight, 0.95)))
    remaining = 1.0 - primary_weight
    q = [0.0] * len(BRANCH_NAMES)
    q[primary] = primary_weight
    if supports:
        w = remaining / len(supports)
        for idx in supports:
            q[idx] = w
    else:
        q[primary] = 1.0
    return q

def primary_support_targets_for_scenario(name: str) -> tuple[int | None, list[int]]:
    return primary_index_for_scenario(name), support_indices_for_scenario(name)

@dataclass
class TopKDecision:
    top1: str
    top2: list[str]
    top3: list[str]
    primary_in_top2: bool
    primary_in_top3: bool
    primary_or_support_at2: bool
    primary_or_support_at3: bool
    weighted_consistency_at3: float

def evaluate_topk_consistency(
    scenario: str,
    sorted_modalities: list[str],
    *,
    rank_weights: tuple[float, float, float] = (1.0, 0.65, 0.40),
) -> TopKDecision:
    key = normalize_scenario_name(scenario)
    primary = PRIMARY_MODALITY_BY_SCENARIO.get(key, "balanced")
    supports = set(SUPPORT_MODALITIES_BY_SCENARIO.get(key, []))
    top2 = list(sorted_modalities[:2])
    top3 = list(sorted_modalities[:3])
    if key == "normal":
        primary_in_top2 = False
        primary_in_top3 = False
        ps2 = False
        ps3 = False
        weighted = 0.0
    else:
        primary_in_top2 = primary in top2
        primary_in_top3 = primary in top3
        ps2 = any(item == primary or item in supports for item in top2)
        ps3 = any(item == primary or item in supports for item in top3)
        weighted = 0.0
        for rank, name in enumerate(top3):
            if name == primary:
                weighted += rank_weights[rank]
            elif name in supports:
                weighted += 0.6 * rank_weights[rank]
        weighted = min(weighted, 1.0)
    return TopKDecision(
        top1=sorted_modalities[0],
        top2=top2,
        top3=top3,
        primary_in_top2=primary_in_top2,
        primary_in_top3=primary_in_top3,
        primary_or_support_at2=ps2,
        primary_or_support_at3=ps3,
        weighted_consistency_at3=weighted,
    )

def entropy_from_probs(probs: Iterable[float]) -> float:
    values = [max(float(x), 1.0e-12) for x in probs]
    total = sum(values)
    if total <= 0.0:
        return 0.0
    values = [x / total for x in values]
    return -sum(v * math.log(v + 1.0e-12) for v in values)

def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
