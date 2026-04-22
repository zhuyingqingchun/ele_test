from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

MODALITY_ORDER = ["position", "electrical", "thermal", "vibration"]


@dataclass(frozen=True)
class ScenarioModalityPrior:
    primary: tuple[str, ...]
    support: tuple[str, ...]
    note: str = ""


SCENARIO_MODALITY_PRIORS_V2: dict[str, ScenarioModalityPrior] = {
    "normal": ScenarioModalityPrior(
        primary=("position", "electrical", "thermal", "vibration"),
        support=(),
        note="Normal samples should remain approximately balanced across modalities.",
    ),
    "load_disturbance_severe": ScenarioModalityPrior(
        primary=("position",),
        support=("electrical",),
        note="Load disturbance should first appear in tracking/position, with electrical effort as support.",
    ),
    "load_disturbance_mild": ScenarioModalityPrior(
        primary=("position",),
        support=("electrical",),
        note="Mild load disturbance still mainly manifests in tracking deviation.",
    ),
    "position_sensor_bias": ScenarioModalityPrior(
        primary=("position",),
        support=("electrical",),
        note="Position bias is directly visible in position-related feedback.",
    ),
    "speed_sensor_scale": ScenarioModalityPrior(
        primary=("position",),
        support=("electrical",),
        note="Speed-scale mismatch propagates into motion/position consistency.",
    ),
    "motor_encoder_freeze": ScenarioModalityPrior(
        primary=("position",),
        support=("electrical",),
        note="Encoder freeze is primarily a motion/feedback phenomenon.",
    ),
    "backlash_growth": ScenarioModalityPrior(
        primary=("position",),
        support=("vibration", "electrical"),
        note="Backlash first shows through reversal dead-zone and position inconsistency.",
    ),
    "jam_fault": ScenarioModalityPrior(
        primary=("position",),
        support=("electrical",),
        note="Hard jams collapse motion first; electrical effort is support evidence.",
    ),
    "intermittent_jam_fault": ScenarioModalityPrior(
        primary=("position",),
        support=("electrical", "vibration"),
        note="Intermittent jams appear in motion collapse bursts with support from actuation and vibration.",
    ),
    "bearing_defect": ScenarioModalityPrior(
        primary=("vibration",),
        support=("electrical",),
        note="Bearing defects should be vibration-led, with electrical disturbance as support.",
    ),
    "winding_resistance_rise": ScenarioModalityPrior(
        primary=("electrical",),
        support=("thermal",),
        note="Electrical variables dominate, while thermal accumulation provides support.",
    ),
    "bus_voltage_sag_fault": ScenarioModalityPrior(
        primary=("electrical",),
        support=("position",),
        note="Bus sag is first an electrical shortage, then a tracking limitation.",
    ),
    "inverter_voltage_loss": ScenarioModalityPrior(
        primary=("electrical",),
        support=("position",),
        note="Inverter voltage loss is primarily electrical with downstream motion effects.",
    ),
    "partial_demagnetization": ScenarioModalityPrior(
        primary=("electrical",),
        support=("thermal", "position"),
        note="Demagnetization first affects flux/electrical conversion, with thermal and tracking support.",
    ),
    "current_sensor_bias": ScenarioModalityPrior(
        primary=("electrical",),
        support=("position",),
        note="Sensor bias is electrical-primary but may propagate to motion inconsistency.",
    ),
    "thermal_saturation": ScenarioModalityPrior(
        primary=("thermal",),
        support=("electrical",),
        note="Thermal saturation should be temperature-led with electrical derating support.",
    ),
    "friction_wear_mild": ScenarioModalityPrior(
        primary=("position",),
        support=("electrical",),
        note="Mild friction is better represented as motion/tracking degradation with electrical support.",
    ),
    "friction_wear_severe": ScenarioModalityPrior(
        primary=("position",),
        support=("electrical", "vibration"),
        note="Severe friction remains motion-primary with stronger electrical and vibration support.",
    ),
}


def normalize_scenario_name(name: str) -> str:
    text = str(name).strip().lower().replace(" ", "_")
    return text


def normalize_modality_name(name: str) -> str:
    text = str(name).strip().lower().replace("-", "_")
    mapping = {
        "pos": "position",
        "position": "position",
        "electrical": "electrical",
        "elec": "electrical",
        "thermal": "thermal",
        "therm": "thermal",
        "vibration": "vibration",
        "vib": "vibration",
    }
    return mapping.get(text, text)


def get_prior_for_scenario(name: str) -> ScenarioModalityPrior:
    key = normalize_scenario_name(name)
    if key in SCENARIO_MODALITY_PRIORS_V2:
        return SCENARIO_MODALITY_PRIORS_V2[key]
    return ScenarioModalityPrior(primary=("position", "electrical"), support=("thermal", "vibration"), note="Fallback prior.")


def topk_contains(topk_modalities: Iterable[str], wanted: Iterable[str]) -> bool:
    topk_norm = {normalize_modality_name(x) for x in topk_modalities}
    wanted_norm = {normalize_modality_name(x) for x in wanted}
    return bool(topk_norm & wanted_norm)


def weighted_consistency_at_3(rank_modalities: list[str], prior: ScenarioModalityPrior) -> float:
    rank_weights = [1.0, 0.6, 0.3]
    total = 0.0
    for i, modality in enumerate(rank_modalities[:3]):
        norm = normalize_modality_name(modality)
        if norm in {normalize_modality_name(x) for x in prior.primary}:
            total += 1.0 * rank_weights[i]
        elif norm in {normalize_modality_name(x) for x in prior.support}:
            total += 0.6 * rank_weights[i]
    return float(total)


def attention_entropy(probabilities: list[float]) -> float:
    import math

    vals = [max(float(x), 1.0e-12) for x in probabilities]
    s = sum(vals)
    if s <= 0.0:
        return 0.0
    probs = [v / s for v in vals]
    return float(-sum(p * math.log(p) for p in probs))
