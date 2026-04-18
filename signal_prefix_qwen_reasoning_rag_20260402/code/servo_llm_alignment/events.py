from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EventItem:
    signal: str
    direction: str
    score: float
    value: float
    baseline: float
    description: str


GENERIC_RULES = [
    ("phase_current_imbalance_est_a", "high", "phase current imbalance is elevated", "elec", "mean"),
    ("current_residual_a", "high", "current residual is elevated", "res", "abs_mean"),
    ("power_gap_w", "high", "electrical-to-mechanical power gap is elevated", "res", "mean"),
    ("back_emf_v", "low", "back EMF is reduced", "res", "mean"),
    ("voltage_margin_v", "low", "voltage margin is reduced", "res", "mean"),
    ("available_bus_voltage_v", "low", "available bus voltage is reduced", "elec", "mean"),
    ("pwm_duty", "high", "PWM duty is elevated", "elec", "mean"),
    ("vibration_band_mps2", "high", "band vibration energy is elevated", "vib", "mean"),
    ("vibration_envelope_mps2", "high", "vibration envelope is elevated", "vib", "mean"),
    ("vibration_shock_index", "high", "shock index is elevated", "vib", "mean"),
    ("shaft_twist_deg", "high", "shaft twist is elevated", "res", "abs_mean"),
    ("position_error_deg", "high", "position error is elevated", "res", "abs_mean"),
    ("speed_residual_deg_s", "high", "speed residual is elevated", "res", "abs_mean"),
    ("copper_loss_w", "high", "copper loss is elevated", "res", "mean"),
    ("winding_temp_rate_c_s", "high", "winding temperature rise is elevated", "therm", "mean"),
    ("winding_temp_c", "high", "winding temperature is elevated", "therm", "mean"),
    ("current_q_meas_a", "high", "q-axis current magnitude is elevated", "elec", "abs_mean"),
    ("current_limit_margin_a", "low", "current limit margin is reduced", "res", "mean"),
]

# Ordered, candidate-specific evidence templates. Earlier entries get higher priority.
SCENARIO_PRIORITY_RULES = {
    "inverter_voltage_loss": [
        ("voltage_cmd_gap_v", "high", "command-to-measured voltage gap is elevated", "derived", "mean"),
        ("pwm_duty", "high", "PWM duty is near saturation", "elec", "mean"),
        ("phase_current_imbalance_est_a", "high", "phase current imbalance is elevated", "elec", "mean"),
        ("voltage_margin_v", "low", "voltage margin is reduced", "res", "mean"),
        ("available_bus_voltage_v", "low", "available bus voltage is reduced", "elec", "mean"),
        ("power_gap_w", "high", "electrical-to-mechanical power gap is elevated", "res", "mean"),
        ("current_residual_a", "high", "current residual is elevated", "res", "abs_mean"),
    ],
    "winding_resistance_rise": [
        ("phase_current_imbalance_est_a", "high", "phase current imbalance is elevated", "elec", "mean"),
        ("copper_loss_w", "high", "copper loss is elevated", "res", "mean"),
        ("winding_temp_rate_c_s", "high", "winding temperature rise is elevated", "therm", "mean"),
        ("current_residual_a", "high", "current residual is elevated", "res", "abs_mean"),
        ("current_limit_margin_a", "low", "current limit margin is reduced", "res", "mean"),
        ("available_bus_voltage_v", "low", "available bus voltage is reduced", "elec", "mean"),
    ],
    "bus_voltage_sag_fault": [
        ("available_bus_voltage_v", "low", "available bus voltage is reduced", "elec", "mean"),
        ("voltage_margin_v", "low", "voltage margin is reduced", "res", "mean"),
        ("pwm_duty", "high", "PWM duty is near saturation", "elec", "mean"),
        ("voltage_cmd_gap_v", "high", "command-to-measured voltage gap is elevated", "derived", "mean"),
        ("power_gap_w", "high", "electrical-to-mechanical power gap is elevated", "res", "mean"),
    ],
    "partial_demagnetization": [
        ("back_emf_v", "low", "back EMF is reduced", "res", "mean"),
        ("current_q_meas_a", "high", "q-axis current magnitude is elevated", "elec", "abs_mean"),
        ("power_gap_w", "high", "electrical-to-mechanical power gap is elevated", "res", "mean"),
        ("current_residual_a", "high", "current residual is elevated", "res", "abs_mean"),
    ],
    "thermal_saturation": [
        ("winding_temp_c", "high", "winding temperature is elevated", "therm", "mean"),
        ("winding_temp_rate_c_s", "high", "winding temperature rise is elevated", "therm", "mean"),
        ("current_limit_margin_a", "low", "current limit margin is reduced", "res", "mean"),
        ("power_gap_w", "high", "electrical-to-mechanical power gap is elevated", "res", "mean"),
    ],
}

ELECTRICAL_SCENARIOS = set(SCENARIO_PRIORITY_RULES.keys())


class EventBuilder:
    def __init__(self, arrays: dict) -> None:
        self.arrays = arrays
        self.column_maps = self._build_column_maps()
        normal_mask = arrays["scenario_names"].astype(str) == "normal"
        self.baselines = self._compute_baselines(normal_mask)

    def _build_column_maps(self) -> dict[str, dict[str, int]]:
        return {
            "pos": {str(name): idx for idx, name in enumerate(self.arrays.get("pos_columns", []))},
            "elec": {str(name): idx for idx, name in enumerate(self.arrays.get("elec_columns", []))},
            "therm": {str(name): idx for idx, name in enumerate(self.arrays.get("therm_columns", []))},
            "vib": {str(name): idx for idx, name in enumerate(self.arrays.get("vib_columns", []))},
            "res": {str(name): idx for idx, name in enumerate(self.arrays.get("res_columns", []))},
            "ctx": {str(name): idx for idx, name in enumerate(self.arrays.get("ctx_columns", []))},
        }

    def _extract_series(self, index: int, group: str, signal: str) -> np.ndarray:
        if group == "derived":
            return self._extract_derived_series(index, signal)
        array_key = f"X_{group}"
        idx = self.column_maps[group][signal]
        return self.arrays[array_key][index, :, idx].astype(float)

    def _extract_derived_series(self, index: int, signal: str) -> np.ndarray:
        if signal == "voltage_cmd_gap_v":
            cmd = self.arrays["X_ctx"][index, :, self.column_maps["ctx"]["voltage_cmd_v"]].astype(float)
            meas = self.arrays["X_elec"][index, :, self.column_maps["elec"]["voltage_meas_v"]].astype(float)
            return np.abs(cmd - meas)
        raise KeyError(signal)

    @staticmethod
    def _reduce_series(series: np.ndarray, reducer: str) -> tuple[float, float]:
        if reducer == "mean":
            primary = float(np.mean(series))
            secondary = float(np.percentile(series, 90))
        elif reducer == "abs_mean":
            abs_series = np.abs(series)
            primary = float(np.mean(abs_series))
            secondary = float(np.percentile(abs_series, 90))
        else:
            raise ValueError(f"Unsupported reducer: {reducer}")
        return primary, secondary

    def _compute_baselines(self, normal_mask: np.ndarray) -> dict[tuple[str, str], tuple[float, float]]:
        baselines: dict[tuple[str, str], tuple[float, float]] = {}
        all_rules = []
        for signal, direction, description, group, reducer in GENERIC_RULES:
            all_rules.append((signal, direction, description, group, reducer))
        for rules in SCENARIO_PRIORITY_RULES.values():
            all_rules.extend(rules)
        seen = set()
        for signal, _direction, _description, group, reducer in all_rules:
            key = (group, signal)
            if key in seen:
                continue
            seen.add(key)
            if group != "derived" and signal not in self.column_maps[group]:
                continue
            values = []
            for idx in np.flatnonzero(normal_mask):
                series = self._extract_series(int(idx), group, signal)
                primary, _secondary = self._reduce_series(series, reducer)
                values.append(primary)
            if not values:
                baselines[key] = (0.0, 1.0)
            else:
                values_arr = np.asarray(values, dtype=float)
                baselines[key] = (float(np.mean(values_arr)), max(float(np.std(values_arr)), 1.0e-6))
        return baselines

    def _score_rule(self, index: int, signal: str, direction: str, description: str, group: str, reducer: str, threshold: float, bonus: float) -> EventItem | None:
        if group != "derived" and signal not in self.column_maps[group]:
            return None
        series = self._extract_series(index, group, signal)
        value, peak = self._reduce_series(series, reducer)
        baseline, scale = self.baselines[(group, signal)]
        signed_primary = (value - baseline) / scale
        signed_peak = (peak - baseline) / scale
        score_primary = signed_primary if direction == "high" else -signed_primary
        score_peak = signed_peak if direction == "high" else -signed_peak
        score = max(score_primary, 0.65 * score_peak) + bonus
        if score < threshold:
            return None
        return EventItem(signal=signal, direction=direction, score=float(score), value=float(value), baseline=float(baseline), description=description)

    def build(self, index: int, top_k: int = 6, candidate_scenarios: list[str] | None = None) -> list[EventItem]:
        candidate_scenarios = candidate_scenarios or []
        top_candidate = candidate_scenarios[0] if candidate_scenarios else ""
        events: list[EventItem] = []
        seen_signals = set()

        if top_candidate in SCENARIO_PRIORITY_RULES:
            for signal, direction, description, group, reducer in SCENARIO_PRIORITY_RULES[top_candidate]:
                item = self._score_rule(index, signal, direction, description, group, reducer, threshold=0.65, bonus=0.55)
                if item is not None:
                    events.append(item)
                    seen_signals.add(item.signal)

        generic_threshold = 0.95 if top_candidate in ELECTRICAL_SCENARIOS else 1.20
        generic_bonus_signals = set(signal for signal, *_rest in SCENARIO_PRIORITY_RULES.get(top_candidate, []))
        for signal, direction, description, group, reducer in GENERIC_RULES:
            if signal in seen_signals:
                continue
            bonus = 0.20 if signal in generic_bonus_signals else 0.0
            item = self._score_rule(index, signal, direction, description, group, reducer, threshold=generic_threshold, bonus=bonus)
            if item is not None:
                events.append(item)

        events.sort(key=lambda item: item.score, reverse=True)
        return events[:top_k]

    def build_summary(self, index: int, top_k: int = 6, candidate_scenarios: list[str] | None = None) -> str:
        events = self.build(index=index, top_k=top_k, candidate_scenarios=candidate_scenarios)
        if not events:
            return "No strong abnormal evidence exceeds the normal baseline by the current heuristic thresholds."
        return "; ".join(
            f"{item.description} (value={item.value:.4f}, baseline={item.baseline:.4f}, z={item.score:.2f})"
            for item in events
        )
