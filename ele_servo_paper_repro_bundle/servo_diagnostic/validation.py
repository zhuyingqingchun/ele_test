from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

import numpy as np

from .config import OperatingCondition, ServoPlantParams
from .controller_tuning import schedule_controller_gains
from .scenarios import create_scenarios
from .simulator import simulate_scenario


@dataclass(frozen=True)
class AcceptanceCriteria:
    condition_name: str
    profile_kind: str
    steady_state_error_deg_max: float | None = None
    settling_time_s_max: float | None = None
    tracking_rms_deg_max: float | None = None
    overshoot_deg_max: float | None = None
    max_output_rate_deg_s_min: float | None = None
    max_winding_temp_c_max: float | None = None
    bandwidth_hz_min: float | None = None
    phase_margin_deg_min: float | None = None
    disturbance_rejection_db_min: float | None = None


@dataclass(frozen=True)
class RunPerformance:
    condition_name: str
    profile_name: str
    steady_state_error_deg: float
    settling_time_s: float | None
    tracking_rms_deg: float
    overshoot_deg: float
    max_output_rate_deg_s: float
    max_winding_temp_c: float
    disturbance_rejection_db: float | None = None
    closed_loop_bandwidth_hz: float | None = None
    speed_loop_phase_margin_deg: float | None = None
    current_loop_phase_margin_deg: float | None = None


@dataclass(frozen=True)
class SinePoint:
    frequency_hz: float
    gain_db: float
    phase_deg: float


def create_acceptance_criteria() -> dict[str, AcceptanceCriteria]:
    return {
        "nominal_multistep": AcceptanceCriteria("nominal_multistep", "step", 0.25, 0.35, None, 0.5, 110.0, 55.0, 2.0, 45.0, 10.0),
        "low_speed_multistep": AcceptanceCriteria("low_speed_multistep", "step", 0.25, 0.38, None, 0.5, 100.0, 50.0, 2.0, 45.0, 10.0),
        "high_speed_multistep": AcceptanceCriteria("high_speed_multistep", "step", 0.30, 0.40, None, 0.6, 115.0, 60.0, 2.0, 45.0, 10.0),
        "cruise_reversal": AcceptanceCriteria("cruise_reversal", "tracking", None, None, 9.5, None, 110.0, 60.0, 2.0, 45.0, 8.0),
        "voltage_margin_track": AcceptanceCriteria("voltage_margin_track", "tracking", None, None, 6.5, None, 95.0, 65.0, 2.0, 40.0, 8.0),
        "sine_tracking": AcceptanceCriteria("sine_tracking", "tracking", None, None, 6.5, None, 95.0, 55.0, 2.0, 45.0, 8.0),
        "hot_gust_mission": AcceptanceCriteria("hot_gust_mission", "tracking", None, None, 3.5, None, 90.0, 85.0, 1.0, 40.0, 10.0),
        "cold_takeoff_reversal": AcceptanceCriteria("cold_takeoff_reversal", "tracking", None, None, 6.0, None, 95.0, 45.0, 2.0, 42.0, 8.0),
    }


def _series(rows: list[dict[str, float | str | int]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=float)


def _detect_plateaus(time_s: np.ndarray, theta_ref_deg: np.ndarray) -> list[tuple[int, int]]:
    if len(theta_ref_deg) == 0:
        return []
    boundaries = [0]
    change_idx = np.where(np.abs(np.diff(theta_ref_deg)) > 1e-9)[0]
    boundaries.extend((change_idx + 1).tolist())
    boundaries.append(len(theta_ref_deg))
    plateaus: list[tuple[int, int]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start < 20:
            continue
        if abs(theta_ref_deg[start]) < 0.5:
            continue
        if time_s[end - 1] - time_s[start] < 0.12:
            continue
        plateaus.append((start, end))
    return plateaus


def _step_metrics(rows: list[dict[str, float | str | int]]) -> tuple[float, float | None, float, float]:
    time_s = _series(rows, "time_s")
    theta_ref_deg = _series(rows, "theta_ref_deg")
    theta_meas_deg = _series(rows, "theta_meas_deg")
    plateaus = _detect_plateaus(time_s, theta_ref_deg)
    if not plateaus:
        tracking_rms = float(np.sqrt(np.mean((theta_ref_deg - theta_meas_deg) ** 2)))
        return 0.0, None, tracking_rms, 0.0

    first_start, first_end = plateaus[0]
    target = float(theta_ref_deg[first_start])
    band = max(0.15, 0.05 * abs(target))
    settle_time = None
    for idx in range(first_start, first_end):
        if np.all(np.abs(theta_meas_deg[idx:first_end] - target) <= band):
            settle_time = float(time_s[idx] - time_s[first_start])
            break

    steady_errors = []
    for start, end in plateaus:
        seg_err = theta_ref_deg[start:end] - theta_meas_deg[start:end]
        tail_len = max(10, (end - start) // 5)
        steady_errors.append(float(np.mean(seg_err[-tail_len:])))

    first_meas = theta_meas_deg[first_start:first_end]
    first_err = theta_ref_deg[first_start:first_end] - first_meas
    overshoot = float(max(0.0, np.max(first_meas) - target))

    return (
        float(np.mean(np.abs(steady_errors))),
        settle_time,
        float(np.sqrt(np.mean(first_err**2))),
        overshoot,
    )


def _tracking_rms(rows: list[dict[str, float | str | int]]) -> float:
    theta_ref_deg = _series(rows, "theta_ref_deg")
    theta_meas_deg = _series(rows, "theta_meas_deg")
    return float(np.sqrt(np.mean((theta_ref_deg - theta_meas_deg) ** 2)))


def _fit_sine_response(time_s: np.ndarray, signal: np.ndarray, omega: float) -> tuple[float, float]:
    reg = np.column_stack((np.sin(omega * time_s), np.cos(omega * time_s), np.ones_like(time_s)))
    coeffs, _, _, _ = np.linalg.lstsq(reg, signal, rcond=None)
    sin_c, cos_c = coeffs[0], coeffs[1]
    amplitude = float(np.hypot(sin_c, cos_c))
    phase = float(np.arctan2(cos_c, sin_c))
    return amplitude, phase


def sweep_closed_loop_frequency_response(
    params: ServoPlantParams,
    condition: OperatingCondition,
    frequencies_hz: Iterable[float] | None = None,
    amplitude_deg: float = 1.0,
) -> list[SinePoint]:
    frequencies = list(frequencies_hz or [0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    normal = next(s for s in create_scenarios() if s.name == "normal")
    points: list[SinePoint] = []
    amplitude_rad = np.deg2rad(amplitude_deg)

    for freq_hz in frequencies:
        warmup_s = 0.6
        duration_s = warmup_s + max(3.0, 6.0 / freq_hz)
        sim_params = replace(params, final_time=duration_s)

        def reference_fn(t: float, freq_hz: float = freq_hz) -> float:
            if t < warmup_s:
                return 0.0
            return amplitude_rad * np.sin(2.0 * np.pi * freq_hz * (t - warmup_s))

        rows = simulate_scenario(sim_params, normal, condition, reference_fn=reference_fn)
        time_s = _series(rows, "time_s")
        theta_ref_deg = _series(rows, "theta_ref_deg")
        theta_meas_deg = _series(rows, "theta_meas_deg")
        analysis_window = time_s >= max(warmup_s + 1.0 / freq_hz, duration_s - 2.5 / freq_hz)
        t_sel = time_s[analysis_window] - warmup_s
        ref_sel = theta_ref_deg[analysis_window]
        meas_sel = theta_meas_deg[analysis_window]
        omega = 2.0 * np.pi * freq_hz
        ref_amp, ref_phase = _fit_sine_response(t_sel, ref_sel, omega)
        out_amp, out_phase = _fit_sine_response(t_sel, meas_sel, omega)
        gain_db = 20.0 * np.log10(max(out_amp, 1e-9) / max(ref_amp, 1e-9))
        phase_deg = float(np.rad2deg(((out_phase - ref_phase + np.pi) % (2.0 * np.pi)) - np.pi))
        points.append(SinePoint(frequency_hz=freq_hz, gain_db=float(gain_db), phase_deg=phase_deg))

    return points


def estimate_closed_loop_bandwidth_hz(points: list[SinePoint]) -> float | None:
    for point in points:
        if point.gain_db <= -3.0:
            return point.frequency_hz
    return None


def _phase_margin_from_loop(loop_response: np.ndarray) -> float | None:
    magnitudes = np.abs(loop_response)
    if np.all(magnitudes < 1.0) or np.all(magnitudes > 1.0):
        return None
    idx = int(np.argmin(np.abs(20.0 * np.log10(np.maximum(magnitudes, 1e-12)))))
    phase_deg = float(np.rad2deg(np.angle(loop_response[idx])))
    return 180.0 + phase_deg


def estimate_loop_margins(params: ServoPlantParams, condition: OperatingCondition) -> tuple[float | None, float | None]:
    gains = schedule_controller_gains(params, condition)
    temp_delta = condition.winding_temp_init_c - 25.0
    resistance = params.phase_resistance * (1.0 + params.resistance_temp_coeff * temp_delta)
    flux_scale = max(0.72, 1.0 + params.flux_temp_coeff * temp_delta)
    flux = params.flux_linkage * flux_scale
    inverter_pole = 1.0 / max(params.inverter_tau, 1e-6)
    omega = np.logspace(0.0, 5.0, 600)
    s = 1j * omega

    current_loop = (gains.current_kp + gains.current_ki / s) / ((params.q_axis_inductance * s + resistance) * (s / inverter_pole + 1.0))
    current_pm = _phase_margin_from_loop(current_loop)

    torque_constant = 1.5 * params.pole_pairs * flux
    j_eq = params.motor_inertia + params.load_inertia / (params.gear_ratio**2)
    b_eq = params.motor_viscous + (params.load_viscous + params.aero_damping) / (params.gear_ratio**2)
    current_bw = omega[int(np.argmin(np.abs(20.0 * np.log10(np.maximum(np.abs(current_loop), 1e-12))))) ]
    current_closed = current_bw / (s + current_bw)
    speed_plant = torque_constant * current_closed / (j_eq * s + b_eq)
    speed_loop = (gains.speed_kp + gains.speed_ki / s) * speed_plant
    speed_pm = _phase_margin_from_loop(speed_loop)
    return current_pm, speed_pm


def estimate_disturbance_rejection_db(rows: list[dict[str, float | str | int]]) -> float:
    theta_ref_deg = _series(rows, "theta_ref_deg")
    theta_meas_deg = _series(rows, "theta_meas_deg")
    error = theta_ref_deg - theta_meas_deg
    active = np.array([float(row["fault_load_active"]) for row in rows], dtype=float) > 0.0
    if not np.any(active):
        return float("nan")
    rms_error = float(np.sqrt(np.mean(error[active] ** 2)))
    command_span = max(float(np.max(theta_ref_deg) - np.min(theta_ref_deg)), 1e-6)
    return float(-20.0 * np.log10(max(rms_error, 1e-6) / command_span))


def evaluate_normal_run(rows: list[dict[str, float | str | int]], condition_name: str) -> RunPerformance:
    steady_state_error_deg, settling_time_s, tracking_rms_deg, overshoot_deg = _step_metrics(rows)
    if condition_name not in {"nominal_multistep", "low_speed_multistep", "high_speed_multistep"}:
        tracking_rms_deg = _tracking_rms(rows)
    max_output_rate_deg_s = float(np.max(np.abs(_series(rows, "omega_load_true_deg_s"))))
    max_winding_temp_c = float(np.max(_series(rows, "winding_temp_c")))
    return RunPerformance(
        condition_name=condition_name,
        profile_name=str(rows[0]["profile_name"]),
        steady_state_error_deg=steady_state_error_deg,
        settling_time_s=settling_time_s,
        tracking_rms_deg=tracking_rms_deg,
        overshoot_deg=overshoot_deg,
        max_output_rate_deg_s=max_output_rate_deg_s,
        max_winding_temp_c=max_winding_temp_c,
    )


def evaluate_acceptance(
    performance: RunPerformance,
    criteria: AcceptanceCriteria,
    disturbance_rejection_db: float | None,
    closed_loop_bandwidth_hz: float | None,
    current_loop_phase_margin_deg: float | None,
    speed_loop_phase_margin_deg: float | None,
) -> dict[str, float | str | int]:
    metrics = asdict(performance)
    metrics["disturbance_rejection_db"] = disturbance_rejection_db
    metrics["closed_loop_bandwidth_hz"] = closed_loop_bandwidth_hz
    metrics["current_loop_phase_margin_deg"] = current_loop_phase_margin_deg
    metrics["speed_loop_phase_margin_deg"] = speed_loop_phase_margin_deg

    checks: list[bool] = []
    if criteria.steady_state_error_deg_max is not None:
        checks.append(performance.steady_state_error_deg <= criteria.steady_state_error_deg_max)
    if criteria.settling_time_s_max is not None and performance.settling_time_s is not None:
        checks.append(performance.settling_time_s <= criteria.settling_time_s_max)
    if criteria.tracking_rms_deg_max is not None:
        checks.append(performance.tracking_rms_deg <= criteria.tracking_rms_deg_max)
    if criteria.overshoot_deg_max is not None:
        checks.append(performance.overshoot_deg <= criteria.overshoot_deg_max)
    if criteria.max_output_rate_deg_s_min is not None:
        checks.append(performance.max_output_rate_deg_s >= criteria.max_output_rate_deg_s_min)
    if criteria.max_winding_temp_c_max is not None:
        checks.append(performance.max_winding_temp_c <= criteria.max_winding_temp_c_max)
    if criteria.bandwidth_hz_min is not None and closed_loop_bandwidth_hz is not None:
        checks.append(closed_loop_bandwidth_hz >= criteria.bandwidth_hz_min)
    if criteria.phase_margin_deg_min is not None and speed_loop_phase_margin_deg is not None:
        checks.append(speed_loop_phase_margin_deg >= criteria.phase_margin_deg_min)
    if criteria.disturbance_rejection_db_min is not None and disturbance_rejection_db is not None:
        checks.append(disturbance_rejection_db >= criteria.disturbance_rejection_db_min)

    return {
        "condition_name": criteria.condition_name,
        "profile_kind": criteria.profile_kind,
        **metrics,
        "pass": int(all(checks) if checks else 0),
    }


def _get_rows_for_key(rows_by_run: dict[str, list[dict[str, float | str | int]]], key: str) -> list[dict[str, float | str | int]]:
    """Return rows for a run key.

    This function supports both single-replicate keys (e.g. "cond__normal") and
    multi-replicate keys (e.g. "cond__normal__rep00"). It prefers the exact key,
    then falls back to the first matching replicate.
    """

    if key in rows_by_run:
        return rows_by_run[key]

    # Fallback: try finding a replicate suffix (e.g., __rep00)
    prefix = key + "__rep"
    for candidate_key in rows_by_run:
        if candidate_key.startswith(prefix):
            return rows_by_run[candidate_key]

    raise KeyError(key)


def build_validation_reports(
    rows_by_run: dict[str, list[dict[str, float | str | int]]],
    params: ServoPlantParams,
    conditions: list[OperatingCondition],
) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | str | int]]]:
    criteria_map = create_acceptance_criteria()
    acceptance_rows: list[dict[str, float | str | int]] = []
    frequency_rows: list[dict[str, float | str | int]] = []

    for condition in conditions:
        normal_rows = _get_rows_for_key(rows_by_run, f"{condition.name}__normal")
        load_rows = None
        try:
            load_rows = _get_rows_for_key(rows_by_run, f"{condition.name}__load_disturbance_mild")
        except KeyError:
            load_rows = None
        disturbance_rejection_db = estimate_disturbance_rejection_db(load_rows) if load_rows is not None else None
        perf = evaluate_normal_run(normal_rows, condition.name)
        freq_points = sweep_closed_loop_frequency_response(params, condition)
        bandwidth_hz = estimate_closed_loop_bandwidth_hz(freq_points)
        current_pm, speed_pm = estimate_loop_margins(params, condition)
        acceptance_rows.append(
            evaluate_acceptance(
                perf,
                criteria_map[condition.name],
                disturbance_rejection_db,
                bandwidth_hz,
                current_pm,
                speed_pm,
            )
        )
        for point in freq_points:
            frequency_rows.append(
                {
                    "condition_name": condition.name,
                    "frequency_hz": point.frequency_hz,
                    "gain_db": point.gain_db,
                    "phase_deg": point.phase_deg,
                    "closed_loop_bandwidth_hz": bandwidth_hz if bandwidth_hz is not None else "",
                    "current_loop_phase_margin_deg": current_pm if current_pm is not None else "",
                    "speed_loop_phase_margin_deg": speed_pm if speed_pm is not None else "",
                }
            )

    return acceptance_rows, frequency_rows
