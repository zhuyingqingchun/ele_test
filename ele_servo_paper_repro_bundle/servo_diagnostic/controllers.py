from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import OperatingCondition, ServoPlantParams
from .controller_tuning import ScheduledControllerGains, schedule_controller_gains

ReferenceFunction = Callable[[float], float]


def clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def first_order_update(state_value: float, target_value: float, tau: float, dt: float) -> float:
    if tau <= 0.0:
        return target_value
    alpha = min(dt / tau, 1.0)
    return state_value + alpha * (target_value - state_value)


def command_profile(profile_name: str, t: float) -> float:
    import numpy as np

    if profile_name == "multi_step":
        if t < 0.4:
            return 0.0
        if t < 1.4:
            return np.deg2rad(15.0)
        if t < 2.4:
            return np.deg2rad(-8.0)
        if t < 3.2:
            return np.deg2rad(12.0)
        return np.deg2rad(4.0)

    if profile_name == "aggressive_step":
        if t < 0.3:
            return 0.0
        if t < 1.2:
            return np.deg2rad(20.0)
        if t < 2.2:
            return np.deg2rad(-15.0)
        if t < 3.0:
            return np.deg2rad(18.0)
        return np.deg2rad(-5.0)

    if profile_name == "reversal":
        if t < 0.4:
            return 0.0
        if t < 1.1:
            return np.deg2rad(10.0)
        if t < 1.8:
            return np.deg2rad(-10.0)
        if t < 2.5:
            return np.deg2rad(14.0)
        if t < 3.2:
            return np.deg2rad(-14.0)
        return np.deg2rad(6.0)

    if profile_name == "sine_sweep":
        envelope = min(max((t - 0.2) / 0.8, 0.0), 1.0)
        return np.deg2rad(12.0) * envelope * np.sin(2.0 * np.pi * (0.4 + 0.35 * t) * t)

    if profile_name == "mission_mix":
        if t < 0.6:
            return np.deg2rad(8.0) * (t / 0.6)
        if t < 1.8:
            return np.deg2rad(8.0) + np.deg2rad(4.0) * np.sin(2.0 * np.pi * 0.9 * (t - 0.6))
        if t < 2.8:
            return np.deg2rad(-12.0)
        return np.deg2rad(5.0) * np.sin(2.0 * np.pi * 1.4 * (t - 2.8))

    raise ValueError(f"Unsupported profile_name: {profile_name}")


def build_reference_function(profile_name: str) -> ReferenceFunction:
    return lambda t: command_profile(profile_name, t)


def anti_windup_integrator(
    integral: float,
    error: float,
    dt: float,
    allow_integrate: bool,
) -> float:
    if allow_integrate:
        return integral + error * dt
    return integral


@dataclass
class ControlOutput:
    theta_ref: float
    position_error: float
    omega_load_ref: float
    omega_motor_ref: float
    i_d_ref: float
    i_q_ref: float
    i_ref: float
    v_d_cmd: float
    v_q_cmd: float
    u_cmd: float


class CascadedController:
    def __init__(
        self,
        params: ServoPlantParams,
        condition: OperatingCondition,
        reference_fn: ReferenceFunction | None = None,
    ) -> None:
        self.params = params
        self.condition = condition
        self.reference_fn = reference_fn or build_reference_function(condition.profile_name)
        self.gains: ScheduledControllerGains = schedule_controller_gains(params, condition)
        self.int_speed = 0.0
        self.int_iq = 0.0
        self.int_id = 0.0
        self.shaped_theta_ref: float | None = None
        self.disturbance_comp_iq = 0.0
        self.prev_omega_motor_meas = 0.0
        self.omega_dot_est = 0.0

    def _shape_reference(self, raw_theta_ref: float) -> float:
        if self.shaped_theta_ref is None:
            self.shaped_theta_ref = raw_theta_ref
            return raw_theta_ref
        filtered_target = first_order_update(
            self.shaped_theta_ref,
            raw_theta_ref,
            self.gains.reference_filter_tau,
            self.params.dt,
        )
        max_delta = self.gains.reference_rate_limit * self.params.dt
        delta = clip(filtered_target - self.shaped_theta_ref, -max_delta, max_delta)
        self.shaped_theta_ref += delta
        return self.shaped_theta_ref

    def _update_load_torque_observer(self, omega_motor_meas: float, i_q_meas: float) -> float:
        if self.gains.disturbance_observer_gain <= 0.0 or self.gains.disturbance_comp_limit <= 0.0:
            self.disturbance_comp_iq = 0.0
            self.prev_omega_motor_meas = omega_motor_meas
            return 0.0

        raw_accel = (omega_motor_meas - self.prev_omega_motor_meas) / self.params.dt
        self.prev_omega_motor_meas = omega_motor_meas
        self.omega_dot_est = first_order_update(
            self.omega_dot_est,
            raw_accel,
            self.gains.disturbance_observer_tau,
            self.params.dt,
        )

        flux = self.params.flux_linkage * max(0.72, 1.0 + self.params.flux_temp_coeff * (self.condition.winding_temp_init_c - 25.0))
        torque_constant = max(1.5 * self.params.pole_pairs * flux, 1e-6)
        j_eq = self.params.motor_inertia + self.params.load_inertia / (self.params.gear_ratio**2)
        b_eq = self.params.motor_viscous + (self.params.load_viscous + self.params.aero_damping) / (self.params.gear_ratio**2)
        torque_em_est = torque_constant * i_q_meas
        load_torque_est = torque_em_est - j_eq * self.omega_dot_est - b_eq * omega_motor_meas
        target_comp = self.gains.disturbance_observer_gain * load_torque_est / torque_constant
        target_comp = clip(target_comp, -self.gains.disturbance_comp_limit, self.gains.disturbance_comp_limit)
        self.disturbance_comp_iq = first_order_update(
            self.disturbance_comp_iq,
            target_comp,
            self.gains.disturbance_observer_tau,
            self.params.dt,
        )
        return self.disturbance_comp_iq

    def compute(
        self,
        t: float,
        theta_meas: float,
        omega_load_est: float,
        omega_motor_meas: float,
        i_d_meas: float,
        i_q_meas: float,
        available_bus_voltage: float,
        available_current_limit: float,
    ) -> ControlOutput:
        import math

        raw_theta_ref = self.reference_fn(t)
        theta_ref = self._shape_reference(raw_theta_ref)
        position_error = theta_ref - theta_meas
        omega_load_ref = clip(
            self.gains.pos_kp * position_error - self.gains.pos_kd * omega_load_est,
            -self.gains.pos_speed_limit,
            self.gains.pos_speed_limit,
        )
        omega_motor_ref = omega_load_ref * self.params.gear_ratio

        speed_error = omega_motor_ref - omega_motor_meas
        disturbance_comp = self._update_load_torque_observer(omega_motor_meas, i_q_meas)
        self.int_speed = clip(self.int_speed + speed_error * self.params.dt, -10.0, 10.0)
        iq_limit = max(float(available_current_limit), 0.1)
        i_q_raw = self.gains.speed_kp * speed_error + self.gains.speed_ki * self.int_speed + disturbance_comp
        i_q_ref = clip(i_q_raw, -iq_limit, iq_limit)
        if i_q_ref != i_q_raw:
            self.int_speed -= speed_error * self.params.dt
        i_d_ref = 0.0

        omega_e = self.params.pole_pairs * omega_motor_meas
        id_error = i_d_ref - i_d_meas
        iq_error = i_q_ref - i_q_meas
        v_d_raw = (
            self.gains.current_d_kp * id_error
            + self.gains.current_d_ki * self.int_id
            - omega_e * self.params.q_axis_inductance * i_q_meas
        )
        v_q_raw = (
            self.gains.current_kp * iq_error
            + self.gains.current_ki * self.int_iq
            + omega_e * (self.params.d_axis_inductance * i_d_meas + self.params.flux_linkage)
        )

        if self.condition.name == "voltage_margin_track":
            ff_scale = min(available_bus_voltage / max(self.condition.bus_voltage, 1e-6), 1.0)
            v_q_raw += ff_scale * 0.08 * self.params.flux_linkage * omega_e

        if self.condition.name == "cold_takeoff_reversal":
            reversal_weight = 1.0 if omega_motor_ref * omega_motor_meas < 0.0 else 0.0
            deadzone_comp = 0.22 * reversal_weight * np.sign(omega_motor_ref) if abs(omega_motor_ref) > 1e-6 else 0.0
            friction_comp = 0.10 * np.tanh(omega_motor_ref / max(self.params.gear_ratio, 1e-6))
            i_q_ref = clip(i_q_ref + deadzone_comp + friction_comp, -iq_limit, iq_limit)
            iq_error = i_q_ref - i_q_meas
            v_q_raw = (
                self.gains.current_kp * iq_error
                + self.gains.current_ki * self.int_iq
                + omega_e * (self.params.d_axis_inductance * i_d_meas + self.params.flux_linkage)
            )

        voltage_limit = max(float(available_bus_voltage), 1e-6) * self.params.modulation_index_limit / math.sqrt(3.0)
        voltage_mag = math.hypot(v_d_raw, v_q_raw)
        scale = 1.0 if voltage_mag <= voltage_limit else voltage_limit / max(voltage_mag, 1e-9)
        v_d_cmd = v_d_raw * scale
        v_q_cmd = v_q_raw * scale
        allow_integrate = scale >= 0.999
        self.int_id = anti_windup_integrator(self.int_id, id_error, self.params.dt, allow_integrate)
        self.int_iq = anti_windup_integrator(self.int_iq, iq_error, self.params.dt, allow_integrate)

        return ControlOutput(
            theta_ref=theta_ref,
            position_error=position_error,
            omega_load_ref=omega_load_ref,
            omega_motor_ref=omega_motor_ref,
            i_d_ref=i_d_ref,
            i_q_ref=i_q_ref,
            i_ref=i_q_ref,
            v_d_cmd=v_d_cmd,
            v_q_cmd=v_q_cmd,
            u_cmd=math.hypot(v_d_cmd, v_q_cmd),
        )


