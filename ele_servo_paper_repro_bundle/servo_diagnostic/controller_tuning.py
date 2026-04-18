from __future__ import annotations

from dataclasses import dataclass

from .config import OperatingCondition, ServoPlantParams


@dataclass(frozen=True)
class ScheduledControllerGains:
    pos_kp: float
    pos_kd: float
    speed_kp: float
    speed_ki: float
    current_kp: float
    current_ki: float
    current_d_kp: float
    current_d_ki: float
    pos_speed_limit: float
    reference_rate_limit: float
    reference_filter_tau: float
    disturbance_observer_gain: float
    disturbance_observer_tau: float
    disturbance_comp_limit: float


def schedule_controller_gains(params: ServoPlantParams, condition: OperatingCondition) -> ScheduledControllerGains:
    voltage_ratio = condition.bus_voltage / max(params.bus_voltage, 1e-6)
    ambient = condition.ambient_temp_c

    pos_kp = params.pos_kp
    pos_kd = params.pos_kd
    speed_kp = params.speed_kp
    speed_ki = params.speed_ki
    current_kp = params.current_kp
    current_ki = params.current_ki
    current_d_kp = params.current_d_kp
    current_d_ki = params.current_d_ki
    pos_speed_limit = params.pos_speed_limit
    reference_rate_limit = 50.0 * params.pos_speed_limit
    reference_filter_tau = 0.0
    disturbance_observer_gain = 0.0
    disturbance_observer_tau = 0.05
    disturbance_comp_limit = 0.0

    if condition.profile_name in {"aggressive_step", "reversal"}:
        pos_kp *= 1.04
        speed_kp *= 1.05
        speed_ki *= 1.05
    elif condition.profile_name == "mission_mix":
        pos_kd *= 1.08
        speed_kp *= 0.98

    if voltage_ratio < 0.95:
        pos_kd *= 1.12
        speed_kp *= 1.08
        speed_ki *= 1.05
        current_kp *= 1.06
        current_ki *= 1.06
        current_d_kp *= 1.06
        current_d_ki *= 1.06
        pos_speed_limit *= 0.96

    if ambient < -10.0:
        pos_kp *= 1.03
        pos_kd *= 1.12
        speed_kp *= 1.08
        speed_ki *= 1.10
        current_kp *= 1.10
        current_ki *= 1.12
        current_d_kp *= 1.10
        current_d_ki *= 1.12
    elif ambient > 45.0:
        pos_kp *= 0.96
        pos_kd *= 1.10
        speed_kp *= 0.95
        speed_ki *= 0.95
        current_kp *= 0.96
        current_ki *= 0.96
        current_d_kp *= 0.96
        current_d_ki *= 0.96
        pos_speed_limit *= 0.94

    if condition.name == "voltage_margin_track":
        pos_kp *= 1.60
        pos_kd *= 1.50
        speed_kp *= 1.70
        speed_ki *= 1.70
        current_kp *= 1.25
        current_ki *= 1.25
        current_d_kp *= 1.25
        current_d_ki *= 1.25
        pos_speed_limit *= 1.55
        reference_rate_limit = 1.2
        reference_filter_tau = 0.010
    elif condition.name == "hot_gust_mission":
        pos_kp *= 1.40
        pos_kd *= 1.40
        speed_kp *= 1.60
        speed_ki *= 1.60
        current_kp *= 1.18
        current_ki *= 1.18
        current_d_kp *= 1.18
        current_d_ki *= 1.18
        pos_speed_limit *= 1.45
        reference_rate_limit = 2.0
        reference_filter_tau = 0.012
        disturbance_observer_gain = 0.0012
        disturbance_observer_tau = 0.030
        disturbance_comp_limit = 1.8
    elif condition.name == "cold_takeoff_reversal":
        pos_kp *= 1.40
        pos_kd *= 1.35
        speed_kp *= 1.50
        speed_ki *= 1.50
        current_kp *= 1.18
        current_ki *= 1.18
        current_d_kp *= 1.18
        current_d_ki *= 1.18
        pos_speed_limit *= 1.45
        reference_rate_limit = 1.8
        reference_filter_tau = 0.008

    return ScheduledControllerGains(
        pos_kp=pos_kp,
        pos_kd=pos_kd,
        speed_kp=speed_kp,
        speed_ki=speed_ki,
        current_kp=current_kp,
        current_ki=current_ki,
        current_d_kp=current_d_kp,
        current_d_ki=current_d_ki,
        pos_speed_limit=pos_speed_limit,
        reference_rate_limit=reference_rate_limit,
        reference_filter_tau=reference_filter_tau,
        disturbance_observer_gain=disturbance_observer_gain,
        disturbance_observer_tau=disturbance_observer_tau,
        disturbance_comp_limit=disturbance_comp_limit,
    )

