from __future__ import annotations

import random
from dataclasses import dataclass

from servo_diagnostic.multimodal_method import (
    BOUNDARY_BY_SCENARIO,
    FAMILY_BY_SCENARIO,
    LOCATION_BY_SCENARIO,
)
from servo_llm_alignment.template_config import DEFAULT_TEMPLATE_NAME, TEMPLATE_CONFIGS, TemplateConfig


SCENARIO_MECHANISM_CUES: dict[str, list[str]] = {
    "normal": [
        "tracking remains stable across modalities",
        "no persistent fault signature emerges",
        "electrical, thermal and mechanical responses remain mutually consistent",
    ],
    "load_disturbance_severe": [
        "external load-path stress becomes dominant",
        "load torque demand rises persistently",
        "tracking is stressed by strong disturbance from the load side",
    ],
    "friction_wear_mild": [
        "drag torque rises mildly",
        "low-speed actuation requires extra effort",
        "controllability is preserved but frictional burden increases",
    ],
    "friction_wear_severe": [
        "drag torque grows substantially",
        "mechanical dissipation becomes pronounced",
        "actuation effort rises under strong frictional loading",
    ],
    "jam_fault": [
        "load-side motion is blocked",
        "shaft torque accumulates against stalled motion",
        "command tracking collapses under a hard mechanical obstruction",
    ],
    "intermittent_jam_fault": [
        "motion is periodically blocked",
        "repeated obstruction produces transient collapse events",
        "torque bursts recur during intermittent mechanical sticking",
    ],
    "current_sensor_bias": [
        "measured current deviates from actual drive current",
        "current-loop feedback is corrupted by sensing bias",
        "electrical feedback becomes internally inconsistent",
    ],
    "speed_sensor_scale": [
        "speed feedback is systematically mis-scaled",
        "control action is driven by distorted speed measurement",
        "feedback dynamics drift because sensed speed is not physically consistent",
    ],
    "position_sensor_bias": [
        "outer-loop position feedback is shifted",
        "tracking error contains a persistent sensing offset",
        "motion remains feasible while the sensed position is biased",
    ],
    "winding_resistance_rise": [
        "phase resistance growth increases electrical stress",
        "copper dissipation and imbalance become more pronounced",
        "drive effort rises under resistance-induced loss",
    ],
    "bus_voltage_sag_fault": [
        "available voltage headroom collapses",
        "dynamic tracking becomes supply-limited",
        "drive capability is restricted by bus-side voltage shortage",
    ],
    "backlash_growth": [
        "transmission slack grows",
        "mechanical free-play becomes persistent",
        "dead-zone behavior emerges in the transmission chain",
    ],
    "thermal_saturation": [
        "thermal buildup constrains deliverable drive effort",
        "temperature-induced derating reduces dynamic capability",
        "performance reduction is driven by thermal accumulation",
    ],
    "motor_encoder_freeze": [
        "motor-side feedback intermittently stops updating",
        "electrical-angle-related feedback becomes unreliable",
        "drive-side motion estimation is corrupted by frozen encoder feedback",
    ],
    "partial_demagnetization": [
        "flux linkage weakens",
        "torque production efficiency degrades",
        "electromagnetic conversion becomes less effective",
    ],
    "inverter_voltage_loss": [
        "converter output capability is degraded",
        "commanded and applied voltage no longer remain consistent",
        "drive-side actuation suffers from inverter-side voltage loss",
    ],
    "bearing_defect": [
        "bearing-side vibration excitation becomes persistent",
        "impulsive mechanical contact signatures intensify",
        "repetitive rolling-element impact behavior appears in vibration channels",
    ],
}


SCENARIO_SIGNAL_CUES: dict[str, list[str]] = {
    "normal": [
        "low residual activity",
        "stable thermal channels",
        "electrical and mechanical responses stay consistent",
        "no persistent anomaly across major modalities",
    ],
    "load_disturbance_severe": [
        "elevated load torque",
        "shaft twist rises under disturbance",
        "speed residual remains elevated",
        "load-side stress dominates the dynamic response",
    ],
    "friction_wear_mild": [
        "friction torque is mildly elevated",
        "q-axis current demand increases",
        "position error grows at low speed",
        "extra drive effort appears without severe collapse",
    ],
    "friction_wear_severe": [
        "friction torque is strongly elevated",
        "current demand rises persistently",
        "mechanical loss increases",
        "motion is maintained with noticeably higher drag burden",
    ],
    "jam_fault": [
        "shaft torque builds up sharply",
        "position error becomes large",
        "load-side speed collapses",
        "motion command and realized motion diverge severely",
    ],
    "intermittent_jam_fault": [
        "repeated torque spikes appear",
        "transient speed collapse recurs",
        "shaft twist grows during blocked intervals",
        "tracking failure appears in bursts rather than continuously",
    ],
    "current_sensor_bias": [
        "current residual remains elevated",
        "phase-current-related measurements are inconsistent",
        "dq-current estimates do not align with physical drive behavior",
    ],
    "speed_sensor_scale": [
        "speed residual is elevated",
        "load acceleration estimate becomes inconsistent",
        "tracking error drifts with mis-scaled speed feedback",
    ],
    "position_sensor_bias": [
        "position error shows persistent offset",
        "encoder mismatch stays elevated",
        "outer-loop deviation remains biased rather than transient",
    ],
    "winding_resistance_rise": [
        "phase current imbalance increases",
        "copper loss rises",
        "effective bus availability is reduced under load",
        "electrical stress grows together with thermal burden",
    ],
    "bus_voltage_sag_fault": [
        "available bus voltage is reduced",
        "pwm duty approaches saturation",
        "voltage margin shrinks during command tracking",
        "dynamic capability degrades because voltage reserve is insufficient",
    ],
    "backlash_growth": [
        "encoder mismatch increases",
        "shaft twist rises during reversal",
        "reversal dead-zone becomes visible",
        "load-side motion lags behind motor-side motion",
    ],
    "thermal_saturation": [
        "winding temperature is elevated",
        "current limit is reduced",
        "temperature rise remains active",
        "thermal channels indicate ongoing derating pressure",
    ],
    "motor_encoder_freeze": [
        "motor encoder count becomes stuck",
        "electrical angle observation is inconsistent",
        "speed feedback shows discontinuity",
        "drive-side motion sensing intermittently stops evolving",
    ],
    "partial_demagnetization": [
        "back-EMF per speed is reduced",
        "q-axis current demand rises",
        "electrical-mechanical power gap enlarges",
        "torque support requires extra electrical effort",
    ],
    "inverter_voltage_loss": [
        "commanded and measured voltage diverge",
        "phase current imbalance increases",
        "pwm duty remains near saturation",
        "actuation asymmetry grows in the electrical drive chain",
    ],
    "bearing_defect": [
        "band-limited vibration energy rises",
        "vibration envelope grows",
        "shock index is elevated",
        "repetitive impulsive vibration becomes prominent",
    ],
}

PRIMARY_MODALITY_BY_SCENARIO: dict[str, str] = {
    "normal": "balanced",
    "load_disturbance_severe": "position",
    "friction_wear_mild": "electrical",
    "friction_wear_severe": "electrical",
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
    "normal": ["position", "electrical", "thermal", "vibration"],
    "load_disturbance_severe": ["electrical", "thermal"],
    "friction_wear_mild": ["position", "thermal"],
    "friction_wear_severe": ["position", "thermal"],
    "jam_fault": ["electrical", "vibration"],
    "intermittent_jam_fault": ["electrical", "vibration"],
    "current_sensor_bias": ["position"],
    "speed_sensor_scale": ["electrical"],
    "position_sensor_bias": ["electrical"],
    "winding_resistance_rise": ["thermal", "position"],
    "bus_voltage_sag_fault": ["position"],
    "backlash_growth": ["vibration", "electrical"],
    "thermal_saturation": ["electrical", "position"],
    "motor_encoder_freeze": ["electrical"],
    "partial_demagnetization": ["position", "thermal"],
    "inverter_voltage_loss": ["position", "vibration"],
    "bearing_defect": ["position", "electrical"],
}


SCENARIO_CONTRAST_CUES: dict[str, list[str]] = {
    "normal": [
        "not characterized by persistent stress concentration in any fault-specific channel",
    ],
    "load_disturbance_severe": [
        "not mainly a sensor-side bias pattern",
        "not dominated by thermal derating",
    ],
    "friction_wear_mild": [
        "not a hard blockage pattern",
        "not primarily a supply-voltage limitation",
    ],
    "friction_wear_severe": [
        "not an intermittent encoder-freeze pattern",
        "not mainly a flux-weakening signature",
    ],
    "jam_fault": [
        "not a mild drag increase pattern",
        "not mainly a sensing offset pattern",
    ],
    "intermittent_jam_fault": [
        "not a continuously stalled state",
        "distinguished by burst-like obstruction rather than steady blockage",
    ],
    "current_sensor_bias": [
        "not mainly a mechanical transmission defect",
        "distinguished by feedback inconsistency rather than load-path stress",
    ],
    "speed_sensor_scale": [
        "not mainly a position-offset pattern",
        "distinguished by speed-loop distortion rather than voltage shortage",
    ],
    "position_sensor_bias": [
        "not mainly a speed-scale distortion",
        "distinguished by persistent position offset rather than transmission slack",
    ],
    "winding_resistance_rise": [
        "not merely a bus-headroom collapse",
        "distinguished by imbalance and copper dissipation growth",
    ],
    "bus_voltage_sag_fault": [
        "not mainly a weakened-flux pattern",
        "distinguished by supply limitation rather than current-sensing bias",
    ],
    "backlash_growth": [
        "not dominated by sensor-side bias",
        "not mainly a thermal derating pattern",
        "distinguished by transmission slack rather than electrical shortage",
    ],
    "thermal_saturation": [
        "not primarily a supply sag pattern",
        "distinguished by thermal derating rather than isolated sensor corruption",
    ],
    "motor_encoder_freeze": [
        "not a static position-bias pattern",
        "distinguished by discontinuous feedback evolution rather than steady offset",
    ],
    "partial_demagnetization": [
        "not mainly a bus-voltage collapse pattern",
        "distinguished by weakened flux rather than pure supply shortage",
    ],
    "inverter_voltage_loss": [
        "not mainly a demagnetization signature",
        "distinguished by commanded-applied voltage mismatch rather than pure flux loss",
    ],
    "bearing_defect": [
        "not mainly a backlash dead-zone pattern",
        "distinguished by impulsive vibration rather than position-loop bias",
    ],
}


CONDITION_CUES: dict[str, str] = {
    "nominal_multistep": "multi-step command regime",
    "low_speed_multistep": "low-speed multi-step regime",
    "high_speed_multistep": "high-speed multi-step regime",
    "cruise_reversal": "cruise reversal regime",
    "voltage_margin_track": "low-voltage-margin tracking regime",
    "sine_tracking": "sinusoidal tracking regime",
    "hot_gust_mission": "hot gust-loaded regime",
    "cold_takeoff_reversal": "cold reversal regime",
}


@dataclass(frozen=True)
class TextBuildResult:
    text: str
    pieces: list[str]
    metadata: dict[str, str | float]


def _severity_bucket(severity: float | int | None) -> str | None:
    if severity is None:
        return None
    sev = float(severity)
    if sev < 0.15:
        return "trace intensity"
    if sev < 0.40:
        return "mild intensity"
    if sev < 0.70:
        return "moderate intensity"
    return "high intensity"


def _normalize_piece(piece: str, use_full_sentences: bool) -> str:
    piece = " ".join(str(piece).strip().split())
    if not piece:
        return ""
    if use_full_sentences:
        if piece[-1] not in ".!?":
            piece = piece[0].upper() + piece[1:] + "."
        else:
            piece = piece[0].upper() + piece[1:]
        return piece
    piece = piece.rstrip(" .;")
    return piece


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _select_items(
    rng: random.Random,
    items: list[str],
    max_items: int,
) -> list[str]:
    if max_items <= 0 or not items:
        return []
    if len(items) <= max_items:
        return list(items)
    return rng.sample(items, k=max_items)


def available_template_names() -> list[str]:
    return sorted(TEMPLATE_CONFIGS.keys())


def get_template_config(template_name: str | None = None) -> TemplateConfig:
    name = template_name or DEFAULT_TEMPLATE_NAME
    if name not in TEMPLATE_CONFIGS:
        raise KeyError(
            f"Unknown template_name={name!r}. Available: {', '.join(available_template_names())}"
        )
    return TEMPLATE_CONFIGS[name]


def build_text_views(
    scenario_name: str,
    *,
    condition_name: str | None = None,
    severity: float | int | None = None,
    source_scenario: str | None = None,
    template_name: str | None = None,
    seed: int | None = None,
) -> dict[str, str]:
    cfg = get_template_config(template_name)
    rng = random.Random(seed)

    family = FAMILY_BY_SCENARIO[scenario_name]
    location = LOCATION_BY_SCENARIO[scenario_name]
    boundary = BOUNDARY_BY_SCENARIO[scenario_name]

    mechanism_items = _select_items(
        rng,
        SCENARIO_MECHANISM_CUES.get(scenario_name, []),
        cfg.max_mechanism_items,
    )
    signal_items = _select_items(
        rng,
        SCENARIO_SIGNAL_CUES.get(scenario_name, []),
        cfg.max_signal_items,
    )
    contrast_items = _select_items(
        rng,
        SCENARIO_CONTRAST_CUES.get(scenario_name, []),
        cfg.max_contrast_items,
    )

    pieces: list[str] = []

    if cfg.include_label_tokens:
        pieces.append(f"{scenario_name.replace('_', ' ')}")
    if cfg.include_family_tokens:
        pieces.append(f"{family.replace('_', ' ')}")
    if cfg.include_location_tokens:
        pieces.append(f"{location.replace('_', ' ')}")
    if cfg.include_boundary_tokens:
        pieces.append(f"{boundary.replace('_', ' ')}")

    if cfg.include_condition and condition_name:
        cond = CONDITION_CUES.get(condition_name, condition_name.replace("_", " "))
        pieces.append(cond)

    if cfg.include_severity_bucket:
        sev_bucket = _severity_bucket(severity)
        if sev_bucket:
            pieces.append(sev_bucket)

    if cfg.include_mechanism:
        pieces.extend(mechanism_items)
    if cfg.include_signal_priors:
        pieces.extend(signal_items)
    if cfg.include_contrast:
        pieces.extend(contrast_items)

    pieces = [_normalize_piece(p, cfg.use_full_sentences) for p in pieces]
    pieces = _deduplicate_preserve_order(pieces)

    if cfg.shuffle_pieces:
        rng.shuffle(pieces)

    if cfg.use_full_sentences:
        text = " ".join(pieces)
    else:
        text = "; ".join(pieces)

    scenario_text = "" if not cfg.include_label_tokens else f"scenario {scenario_name.replace('_', ' ')}"
    family_text = "" if not cfg.include_family_tokens else f"family {family.replace('_', ' ')}"
    location_text = "" if not cfg.include_location_tokens else f"location {location.replace('_', ' ')}"
    boundary_text = "" if not cfg.include_boundary_tokens else f"boundary {boundary.replace('_', ' ')}"
    mechanism_text = "; ".join(_deduplicate_preserve_order([_normalize_piece(x, False) for x in mechanism_items]))
    evidence_text = "; ".join(_deduplicate_preserve_order([_normalize_piece(x, False) for x in signal_items]))
    contrast_text = "; ".join(_deduplicate_preserve_order([_normalize_piece(x, False) for x in contrast_items]))

    return {
        "scenario_text": scenario_text,
        "family_text": family_text,
        "location_text": location_text,
        "boundary_text": boundary_text,
        "mechanism_text": mechanism_text,
        "evidence_text": evidence_text,
        "contrast_text": contrast_text,
        "combined_text": text,
        "template_name": cfg.name,
    }


def build_text_record(
    scenario_name: str,
    *,
    condition_name: str | None = None,
    severity: float | int | None = None,
    source_scenario: str | None = None,
    template_name: str | None = None,
    seed: int | None = None,
) -> TextBuildResult:
    texts = build_text_views(
        scenario_name,
        condition_name=condition_name,
        severity=severity,
        source_scenario=source_scenario,
        template_name=template_name,
        seed=seed,
    )
    combined = texts["combined_text"]
    pieces = [p.strip() for p in combined.replace(".", ";").split(";") if p.strip()]
    return TextBuildResult(
        text=combined,
        pieces=pieces,
        metadata={
            "template_name": texts["template_name"],
            "scenario_name": scenario_name,
            "condition_name": condition_name or "",
            "source_scenario": source_scenario or "",
            "severity": float(severity) if severity is not None else -1.0,
        },
    )