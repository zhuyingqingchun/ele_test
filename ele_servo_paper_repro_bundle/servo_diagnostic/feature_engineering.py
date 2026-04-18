from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ServoPlantParams
from .operating_conditions import create_operating_conditions
from .scenarios import create_scenarios


DEFAULT_SIGNAL_COLUMNS = [
    "theta_meas_deg",
    "theta_motor_meas_deg",
    "theta_e_obs_deg",
    "omega_motor_meas_deg_s",
    "omega_load_est_deg_s",
    "load_accel_est_deg_s2",
    "speed_residual_deg_s",
    "current_q_meas_a",
    "current_d_meas_a",
    "current_residual_a",
    "phase_current_a_est_a",
    "phase_current_b_est_a",
    "phase_current_c_est_a",
    "phase_current_neutral_est_a",
    "phase_current_imbalance_est_a",
    "phase_voltage_a_meas_v",
    "phase_voltage_b_meas_v",
    "phase_voltage_c_meas_v",
    "bus_current_meas_a",
    "available_bus_voltage_v",
    "voltage_meas_v",
    "pwm_duty",
    "back_emf_v",
    "torque_shaft_nm",
    "torque_load_nm",
    "shaft_twist_deg",
    "winding_temp_c",
    "housing_temp_c",
    "winding_temp_rate_c_s",
    "housing_temp_rate_c_s",
    "vibration_accel_mps2",
    "vibration_band_mps2",
    "vibration_envelope_mps2",
    "vibration_shock_index",
]


@dataclass(frozen=True)
class WindowConfig:
    window_size: int = 256
    stride: int = 64


@dataclass(frozen=True)
class WindowLabelAssignment:
    scenario: str
    fault_label: str
    fault_id: int
    source_scenario: str
    source_fault_label: str
    source_fault_id: int
    fault_active_center: bool
    fault_active_ratio: float
    window_label_state: str


SCENARIO_MAP = {scenario.name: scenario for scenario in create_scenarios()}
CONDITION_MAP = {condition.name: condition for condition in create_operating_conditions()}
BASE_CURRENT_LIMIT_A = ServoPlantParams().current_limit
FAULT_ACTIVE_RATIO_THRESHOLD = 0.15
FAULT_ACTIVE_EPS = 1.0e-9


def _float_row_value(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _bool_row_value(row: dict[str, str], key: str) -> bool:
    value = row.get(key, "")
    return str(value).strip().lower() in {"1", "1.0", "true", "yes"}


def _fault_started(time_s: float, start_s: float | None) -> bool:
    return start_s is not None and time_s + FAULT_ACTIVE_EPS >= start_s


def _periodic_fault_active(time_s: float, start_s: float | None, duration_s: float | None, period_s: float | None) -> bool:
    if start_s is None or duration_s is None or period_s is None:
        return False
    if time_s + FAULT_ACTIVE_EPS < start_s:
        return False
    phase = (time_s - start_s) % period_s
    return phase <= duration_s + FAULT_ACTIVE_EPS


def fault_active_for_row(row: dict[str, str]) -> bool:
    scenario_name = str(row["scenario"])
    if scenario_name == "normal":
        return False

    scenario = SCENARIO_MAP.get(scenario_name)
    if scenario is None:
        return str(row.get("fault_label", "normal")) != "normal"

    time_s = _float_row_value(row, "time_s", default=0.0)

    if scenario_name == "intermittent_jam_fault":
        return _periodic_fault_active(time_s, scenario.jam_start, scenario.jam_duration_s, scenario.jam_period_s)
    if scenario_name == "motor_encoder_freeze":
        return _periodic_fault_active(
            time_s,
            scenario.motor_encoder_freeze_start,
            scenario.motor_encoder_freeze_duration_s,
            scenario.motor_encoder_freeze_period_s,
        )
    if scenario_name == "thermal_saturation":
        if not _fault_started(time_s, scenario.thermal_derate_start):
            return False
        winding_temp_c = _float_row_value(row, "winding_temp_c")
        current_limit_a = _float_row_value(row, "current_limit_a", default=BASE_CURRENT_LIMIT_A)
        temp_triggered = math.isfinite(winding_temp_c) and winding_temp_c + FAULT_ACTIVE_EPS >= scenario.thermal_limit_temp_c
        derate_triggered = current_limit_a + FAULT_ACTIVE_EPS < BASE_CURRENT_LIMIT_A
        return temp_triggered or derate_triggered
    if scenario_name == "winding_resistance_rise":
        if not _fault_started(time_s, scenario.resistance_scale_start):
            return False
        if not _bool_row_value(row, "fault_electrical_active"):
            return False
        condition = CONDITION_MAP.get(str(row.get("condition_name", "")))
        nominal_bus_v = condition.bus_voltage if condition is not None else _float_row_value(row, "available_bus_voltage_v", default=28.0)
        phase_imbalance_a = _float_row_value(row, "phase_current_imbalance_est_a", default=0.0)
        current_residual_a = abs(_float_row_value(row, "current_residual_a", default=0.0))
        copper_loss_w = _float_row_value(row, "copper_loss_w", default=0.0)
        winding_temp_rate_c_s = _float_row_value(row, "winding_temp_rate_c_s", default=0.0)
        available_bus_v = _float_row_value(row, "available_bus_voltage_v", default=nominal_bus_v)
        voltage_drop_v = max(nominal_bus_v - available_bus_v, 0.0)

        symptom_count = 0
        symptom_count += int(phase_imbalance_a >= 0.28)
        symptom_count += int(current_residual_a >= 0.20)
        symptom_count += int(copper_loss_w >= 0.10)
        symptom_count += int(winding_temp_rate_c_s >= 0.035)
        symptom_count += int(voltage_drop_v >= 0.15)
        strong_evidence = (
            (phase_imbalance_a >= 0.40 and copper_loss_w >= 0.16)
            or (current_residual_a >= 0.30 and copper_loss_w >= 0.16)
        )
        return symptom_count >= 2 or strong_evidence
    if scenario_name == "partial_demagnetization":
        if not _fault_started(time_s, scenario.flux_scale_start):
            return False
        if not _bool_row_value(row, "fault_electrical_active"):
            return False
        back_emf_v = _float_row_value(row, "back_emf_v", default=0.0)
        omega_motor_deg_s = abs(_float_row_value(row, "omega_motor_meas_deg_s", default=0.0))
        current_q_a = abs(_float_row_value(row, "current_q_meas_a", default=0.0))
        current_residual_a = abs(_float_row_value(row, "current_residual_a", default=0.0))
        electrical_power_w = _float_row_value(row, "electrical_power_w", default=0.0)
        mechanical_power_w = _float_row_value(row, "mechanical_power_w", default=0.0)
        power_gap_w = abs(electrical_power_w - mechanical_power_w)
        bemf_per_speed = back_emf_v / max(omega_motor_deg_s, 180.0)

        symptom_count = 0
        symptom_count += int(bemf_per_speed <= 0.0105)
        symptom_count += int(current_q_a >= 0.18)
        symptom_count += int(current_residual_a >= 0.12)
        symptom_count += int(power_gap_w >= 0.45)
        strong_evidence = (
            (bemf_per_speed <= 0.0095 and current_q_a >= 0.16)
            or (current_residual_a >= 0.18 and power_gap_w >= 0.35)
        )
        return symptom_count >= 2 or strong_evidence
    if scenario_name == "inverter_voltage_loss":
        if not _fault_started(time_s, scenario.inverter_voltage_scale_start):
            return False
        if not _bool_row_value(row, "fault_electrical_active"):
            return False
        voltage_cmd_v = abs(_float_row_value(row, "voltage_cmd_v", default=0.0))
        voltage_meas_v = abs(_float_row_value(row, "voltage_meas_v", default=0.0))
        current_residual_a = abs(_float_row_value(row, "current_residual_a", default=0.0))
        phase_imbalance_a = _float_row_value(row, "phase_current_imbalance_est_a", default=0.0)
        electrical_power_w = _float_row_value(row, "electrical_power_w", default=0.0)
        mechanical_power_w = _float_row_value(row, "mechanical_power_w", default=0.0)
        power_gap_w = abs(electrical_power_w - mechanical_power_w)
        pwm_duty = _float_row_value(row, "pwm_duty", default=0.0)
        command_gap_v = max(voltage_cmd_v - voltage_meas_v, 0.0)

        symptom_count = 0
        symptom_count += int(command_gap_v >= 0.80)
        symptom_count += int(current_residual_a >= 0.35)
        symptom_count += int(phase_imbalance_a >= 0.22)
        symptom_count += int(power_gap_w >= 8.0)
        symptom_count += int(pwm_duty >= 0.76)
        strong_evidence = (
            (current_residual_a >= 0.50 and power_gap_w >= 12.0)
            or (command_gap_v >= 1.20 and phase_imbalance_a >= 0.24)
        )
        return symptom_count >= 2 or strong_evidence
    if scenario_name == "load_disturbance_severe":
        if not _fault_started(time_s, scenario.extra_load_start):
            return False
        if not _bool_row_value(row, "fault_load_active"):
            return False
        speed_residual_deg_s = abs(_float_row_value(row, "speed_residual_deg_s", default=0.0))
        shaft_twist_deg = abs(_float_row_value(row, "shaft_twist_deg", default=0.0))
        torque_load_nm = abs(_float_row_value(row, "torque_load_nm", default=0.0))
        torque_shaft_nm = abs(_float_row_value(row, "torque_shaft_nm", default=0.0))
        current_q_a = abs(_float_row_value(row, "current_q_meas_a", default=0.0))
        torque_gap_nm = abs(torque_shaft_nm - torque_load_nm)

        symptom_count = 0
        symptom_count += int(speed_residual_deg_s >= 7.5)
        symptom_count += int(shaft_twist_deg >= 0.16)
        symptom_count += int(torque_load_nm >= 0.18)
        symptom_count += int(current_q_a >= 0.14)
        symptom_count += int(torque_gap_nm >= 0.08)
        strong_evidence = (
            (speed_residual_deg_s >= 12.0 and shaft_twist_deg >= 0.22)
            or (current_q_a >= 0.20 and torque_load_nm >= 0.22)
        )
        return symptom_count >= 2 or strong_evidence
    if scenario_name == "speed_sensor_scale":
        if not _fault_started(time_s, scenario.speed_sensor_scale_start):
            return False
        if not _bool_row_value(row, "fault_speed_sensor_active"):
            return False
        omega_motor_meas = abs(_float_row_value(row, "omega_motor_meas_deg_s", default=0.0))
        omega_load_est = abs(_float_row_value(row, "omega_load_est_deg_s", default=0.0))
        speed_residual_deg_s = abs(_float_row_value(row, "speed_residual_deg_s", default=0.0))
        theta_motor_meas = _float_row_value(row, "theta_motor_meas_deg", default=0.0)
        theta_meas = _float_row_value(row, "theta_meas_deg", default=0.0)
        load_accel_est = abs(_float_row_value(row, "load_accel_est_deg_s2", default=0.0))
        encoder_mismatch_deg = abs(theta_motor_meas / max(ServoPlantParams().gear_ratio, 1.0) - theta_meas)
        speed_consistency_deg_s = abs(omega_motor_meas / max(ServoPlantParams().gear_ratio, 1.0) - omega_load_est)
        normalized_consistency = speed_consistency_deg_s / max(omega_load_est, 25.0)

        symptom_count = 0
        symptom_count += int(speed_residual_deg_s >= 11.5)
        symptom_count += int(speed_consistency_deg_s >= 4.2)
        symptom_count += int(normalized_consistency >= 0.12)
        symptom_count += int(encoder_mismatch_deg >= 0.20)
        symptom_count += int(load_accel_est >= 180.0 and speed_residual_deg_s >= 8.0)
        strong_evidence = (
            (speed_residual_deg_s >= 15.0 and speed_consistency_deg_s >= 5.0)
            or (normalized_consistency >= 0.15 and encoder_mismatch_deg >= 0.24)
            or (speed_consistency_deg_s >= 4.2 and encoder_mismatch_deg >= 0.20 and load_accel_est >= 180.0)
        )
        return symptom_count >= 2 or strong_evidence
    if scenario_name == "backlash_growth":
        if not _fault_started(time_s, scenario.backlash_scale_start):
            return False
        if not _bool_row_value(row, "fault_mechanical_active"):
            return False
        shaft_twist_deg = abs(_float_row_value(row, "shaft_twist_deg", default=0.0))
        position_error_deg = abs(_float_row_value(row, "position_error_deg", default=0.0))
        torque_shaft_nm = abs(_float_row_value(row, "torque_shaft_nm", default=0.0))
        torque_load_nm = abs(_float_row_value(row, "torque_load_nm", default=0.0))
        speed_residual_deg_s = abs(_float_row_value(row, "speed_residual_deg_s", default=0.0))
        torque_gap_nm = abs(torque_shaft_nm - torque_load_nm)

        symptom_count = 0
        symptom_count += int(shaft_twist_deg >= 0.12)
        symptom_count += int(position_error_deg >= 0.22)
        symptom_count += int(torque_gap_nm >= 0.06)
        symptom_count += int(speed_residual_deg_s >= 5.5)
        strong_evidence = (
            (shaft_twist_deg >= 0.18 and torque_gap_nm >= 0.09)
            or (position_error_deg >= 0.35 and speed_residual_deg_s >= 6.5)
        )
        return symptom_count >= 2 or strong_evidence
    if scenario_name == "bearing_defect":
        if not _fault_started(time_s, scenario.bearing_fault_start):
            return False
        if not _bool_row_value(row, "fault_bearing_active"):
            return False
        vibration_band = abs(_float_row_value(row, "vibration_band_mps2", default=0.0))
        vibration_envelope = _float_row_value(row, "vibration_envelope_mps2", default=0.0)
        vibration_shock = _float_row_value(row, "vibration_shock_index", default=0.0)
        vibration_accel = _float_row_value(row, "vibration_accel_mps2", default=0.0)

        symptom_count = 0
        symptom_count += int(vibration_band >= 0.10)
        symptom_count += int(vibration_envelope >= 1.10)
        symptom_count += int(vibration_shock >= 55.0)
        symptom_count += int(vibration_accel >= 3.20)
        strong_evidence = (
            (vibration_band >= 0.18 and vibration_shock >= 80.0)
            or (vibration_band >= 0.18 and vibration_accel >= 4.20)
        )
        return symptom_count >= 2 or strong_evidence

    start_markers = [
        scenario.extra_load_start,
        scenario.gust_start,
        scenario.friction_start,
        scenario.jam_start,
        scenario.current_sensor_bias_start,
        scenario.speed_sensor_scale_start,
        scenario.position_sensor_bias_start,
        scenario.resistance_scale_start,
        scenario.flux_scale_start,
        scenario.bus_voltage_scale_start,
        scenario.inverter_voltage_scale_start,
        scenario.backlash_scale_start,
        scenario.bearing_fault_start,
    ]
    return any(_fault_started(time_s, marker) for marker in start_markers)


def build_fault_active_mask(run_rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([fault_active_for_row(row) for row in run_rows], dtype=bool)


def assign_window_label(
    run_rows: list[dict[str, str]],
    active_mask: np.ndarray,
    start: int,
    end: int,
    active_ratio_threshold: float = FAULT_ACTIVE_RATIO_THRESHOLD,
    drop_ambiguous: bool = True,
) -> WindowLabelAssignment | None:
    center_offset = start + (end - start) // 2
    center_row = run_rows[center_offset]
    window_mask = active_mask[start:end]
    source_scenario = str(center_row["scenario"])
    source_fault_label = str(center_row["fault_label"])
    source_fault_id = int(center_row["fault_id"])
    center_local_index = (end - start) // 2
    fault_active_center = bool(window_mask[center_local_index]) if window_mask.size else False
    fault_active_ratio = float(window_mask.mean()) if window_mask.size else 0.0

    if source_fault_label == "normal":
        return WindowLabelAssignment(
            scenario="normal",
            fault_label="normal",
            fault_id=0,
            source_scenario=source_scenario,
            source_fault_label=source_fault_label,
            source_fault_id=source_fault_id,
            fault_active_center=False,
            fault_active_ratio=0.0,
            window_label_state="normal",
        )

    if fault_active_center or fault_active_ratio >= active_ratio_threshold:
        return WindowLabelAssignment(
            scenario=source_scenario,
            fault_label=source_fault_label,
            fault_id=source_fault_id,
            source_scenario=source_scenario,
            source_fault_label=source_fault_label,
            source_fault_id=source_fault_id,
            fault_active_center=fault_active_center,
            fault_active_ratio=fault_active_ratio,
            window_label_state="active_fault",
        )

    if fault_active_ratio <= FAULT_ACTIVE_EPS:
        return WindowLabelAssignment(
            scenario="normal",
            fault_label="normal",
            fault_id=0,
            source_scenario=source_scenario,
            source_fault_label=source_fault_label,
            source_fault_id=source_fault_id,
            fault_active_center=False,
            fault_active_ratio=0.0,
            window_label_state="pre_fault_normal",
        )

    if drop_ambiguous:
        return None

    assigned_fault = fault_active_ratio >= 0.5 * active_ratio_threshold
    return WindowLabelAssignment(
        scenario=source_scenario if assigned_fault else "normal",
        fault_label=source_fault_label if assigned_fault else "normal",
        fault_id=source_fault_id if assigned_fault else 0,
        source_scenario=source_scenario,
        source_fault_label=source_fault_label,
        source_fault_id=source_fault_id,
        fault_active_center=fault_active_center,
        fault_active_ratio=fault_active_ratio,
        window_label_state="transition_fault" if assigned_fault else "transition_normal",
    )


def load_dataset_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def group_rows_by_run(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["condition_name"], row["scenario"])
        grouped.setdefault(key, []).append(row)
    for run_rows in grouped.values():
        run_rows.sort(key=lambda item: int(item["sample_index"]))
    return grouped


def rows_to_signal_array(rows: list[dict[str, str]], signal_columns: list[str]) -> np.ndarray:
    return np.array([[float(row[column]) for column in signal_columns] for row in rows], dtype=float)


def build_window_samples(
    rows: list[dict[str, str]],
    signal_columns: list[str] | None = None,
    config: WindowConfig | None = None,
    active_ratio_threshold: float = FAULT_ACTIVE_RATIO_THRESHOLD,
    drop_ambiguous: bool = True,
) -> tuple[np.ndarray, list[dict[str, str | int | float]]]:
    signal_columns = signal_columns or DEFAULT_SIGNAL_COLUMNS
    config = config or WindowConfig()
    grouped = group_rows_by_run(rows)
    windows: list[np.ndarray] = []
    metadata: list[dict[str, str | int | float]] = []

    for (condition_name, _scenario_name), run_rows in grouped.items():
        signal_array = rows_to_signal_array(run_rows, signal_columns)
        total = signal_array.shape[0]
        if total < config.window_size:
            continue
        active_mask = build_fault_active_mask(run_rows)
        for start in range(0, total - config.window_size + 1, config.stride):
            end = start + config.window_size
            window = signal_array[start:end]
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
            windows.append(window)
            metadata.append(
                {
                    "condition_name": condition_name,
                    "scenario": label.scenario,
                    "fault_label": label.fault_label,
                    "fault_id": label.fault_id,
                    "source_scenario": label.source_scenario,
                    "source_fault_label": label.source_fault_label,
                    "source_fault_id": label.source_fault_id,
                    "fault_active_center": int(label.fault_active_center),
                    "fault_active_ratio": label.fault_active_ratio,
                    "window_label_state": label.window_label_state,
                    "window_start": start,
                    "window_end": end,
                    "time_start_s": float(run_rows[start]["time_s"]),
                    "time_end_s": float(run_rows[end - 1]["time_s"]),
                }
            )

    if not windows:
        return np.empty((0, config.window_size, len(signal_columns))), metadata
    return np.stack(windows, axis=0), metadata


def extract_statistical_features(window: np.ndarray, signal_columns: list[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    signal_map = {column_name: window[:, column_index] for column_index, column_name in enumerate(signal_columns)}
    for column_name, series in signal_map.items():
        features[f"{column_name}__mean"] = float(np.mean(series))
        features[f"{column_name}__std"] = float(np.std(series))
        features[f"{column_name}__rms"] = float(np.sqrt(np.mean(series**2)))
        features[f"{column_name}__min"] = float(np.min(series))
        features[f"{column_name}__max"] = float(np.max(series))
        features[f"{column_name}__ptp"] = float(np.ptp(series))
        features[f"{column_name}__slope"] = float(series[-1] - series[0])

    if all(name in signal_map for name in ("phase_current_a_est_a", "phase_current_b_est_a", "phase_current_c_est_a")):
        ia = signal_map["phase_current_a_est_a"]
        ib = signal_map["phase_current_b_est_a"]
        ic = signal_map["phase_current_c_est_a"]
        imbalance = np.sqrt(((ia - ib) ** 2 + (ib - ic) ** 2 + (ic - ia) ** 2) / 3.0)
        neutral = ia + ib + ic
        features["phase_current_imbalance__mean"] = float(np.mean(imbalance))
        features["phase_current_imbalance__max"] = float(np.max(imbalance))
        features["phase_current_neutral__rms"] = float(np.sqrt(np.mean(neutral**2)))

    if all(name in signal_map for name in ("winding_temp_c", "housing_temp_c")):
        thermal_gradient = signal_map["winding_temp_c"] - signal_map["housing_temp_c"]
        features["thermal_gradient__mean"] = float(np.mean(thermal_gradient))
        features["thermal_gradient__max"] = float(np.max(thermal_gradient))

    if all(name in signal_map for name in ("current_q_meas_a", "back_emf_v")):
        electro_proxy = signal_map["current_q_meas_a"] * signal_map["back_emf_v"]
        features["electro_proxy__mean"] = float(np.mean(electro_proxy))
        features["electro_proxy__std"] = float(np.std(electro_proxy))

    if all(name in signal_map for name in ("theta_meas_deg", "theta_motor_meas_deg")):
        encoder_mismatch = signal_map["theta_motor_meas_deg"] / 45.0 - signal_map["theta_meas_deg"]
        features["encoder_mismatch__rms"] = float(np.sqrt(np.mean(encoder_mismatch**2)))
        features["encoder_mismatch__max"] = float(np.max(np.abs(encoder_mismatch)))

    return features


def build_feature_rows(
    rows: list[dict[str, str]],
    signal_columns: list[str] | None = None,
    config: WindowConfig | None = None,
) -> list[dict[str, str | int | float]]:
    signal_columns = signal_columns or DEFAULT_SIGNAL_COLUMNS
    windows, metadata = build_window_samples(rows, signal_columns, config)
    feature_rows: list[dict[str, str | int | float]] = []
    for window, meta in zip(windows, metadata):
        record: dict[str, str | int | float] = dict(meta)
        record.update(extract_statistical_features(window, signal_columns))
        feature_rows.append(record)
    return feature_rows


def save_feature_rows(path: Path, feature_rows: list[dict[str, str | int | float]]) -> None:
    if not feature_rows:
        raise ValueError("No feature rows to save.")
    fieldnames = list(feature_rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(feature_rows)


def save_window_dataset(
    path: Path,
    windows: np.ndarray,
    metadata: list[dict[str, str | int | float]],
    signal_columns: list[str],
) -> None:
    fault_ids = np.array([int(item["fault_id"]) for item in metadata], dtype=np.int32)
    scenario_names = np.array([str(item["scenario"]) for item in metadata], dtype=object)
    condition_names = np.array([str(item["condition_name"]) for item in metadata], dtype=object)
    np.savez_compressed(
        path,
        X=windows,
        y_fault_id=fault_ids,
        scenario=scenario_names,
        condition=condition_names,
        signal_columns=np.array(signal_columns, dtype=object),
    )


def save_window_metadata(path: Path, metadata: list[dict[str, str | int | float]]) -> None:
    if not metadata:
        raise ValueError("No metadata rows to save.")
    fieldnames = list(metadata[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
