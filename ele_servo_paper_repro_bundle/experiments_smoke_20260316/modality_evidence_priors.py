from __future__ import annotations

from dataclasses import dataclass

BRANCH_NAMES = ["position", "electrical", "thermal", "vibration"]
BRANCH_INDEX = {name: idx for idx, name in enumerate(BRANCH_NAMES)}


@dataclass(frozen=True)
class ModalityPrior:
    primary: str
    support: tuple[str, ...]


SCENARIO_MODALITY_PRIORS: dict[str, ModalityPrior] = {
    "normal": ModalityPrior("position", ("electrical", "thermal", "vibration")),
    "position_sensor_bias": ModalityPrior("position", ("electrical",)),
    "speed_sensor_scale": ModalityPrior("position", ("electrical",)),
    "motor_encoder_freeze": ModalityPrior("position", ("electrical",)),
    "load_disturbance_mild": ModalityPrior("position", ("electrical", "vibration")),
    "load_disturbance_severe": ModalityPrior("position", ("electrical", "vibration")),
    "backlash_growth": ModalityPrior("position", ("vibration",)),
    "jam_fault": ModalityPrior("position", ("electrical", "vibration")),
    "intermittent_jam_fault": ModalityPrior("position", ("electrical", "vibration")),
    "winding_resistance_rise": ModalityPrior("electrical", ("thermal",)),
    "bus_voltage_sag_fault": ModalityPrior("electrical", ("position",)),
    "inverter_voltage_loss": ModalityPrior("electrical", ("position",)),
    "current_sensor_bias": ModalityPrior("electrical", ("position",)),
    "partial_demagnetization": ModalityPrior("electrical", ("position",)),
    "thermal_saturation": ModalityPrior("thermal", ("electrical",)),
    "friction_wear_mild": ModalityPrior("electrical", ("position", "thermal")),
    "friction_wear_severe": ModalityPrior("electrical", ("position", "thermal")),
    "bearing_defect": ModalityPrior("vibration", ("position",)),
}


MECHANICAL_RELATED = {
    "backlash_growth",
    "jam_fault",
    "intermittent_jam_fault",
    "bearing_defect",
    "friction_wear_mild",
    "friction_wear_severe",
    "load_disturbance_mild",
    "load_disturbance_severe",
}


def get_modality_prior(scenario_name: str) -> ModalityPrior:
    key = str(scenario_name).strip()
    return SCENARIO_MODALITY_PRIORS.get(key, ModalityPrior("position", ("electrical",)))


def primary_index(scenario_name: str) -> int:
    return BRANCH_INDEX[get_modality_prior(scenario_name).primary]


def support_mask(scenario_name: str) -> list[float]:
    prior = get_modality_prior(scenario_name)
    active = {prior.primary, *prior.support}
    return [1.0 if name in active else 0.0 for name in BRANCH_NAMES]
