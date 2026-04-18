from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FaultScenario, OperatingCondition, ServoPlantParams


@dataclass
class PlantState:
    i_d: float = 0.0
    i_q: float = 0.0
    omega_m: float = 0.0
    theta_m: float = 0.0
    omega_l: float = 0.0
    theta_l: float = 0.0
    v_d_applied: float = 0.0
    v_q_applied: float = 0.0
    winding_temp_c: float = 25.0
    housing_temp_c: float = 25.0
    prev_shaft_torque: float = 0.0
    phase_current_a_sensor_state: float = 0.0
    phase_current_b_sensor_state: float = 0.0
    phase_current_c_sensor_state: float = 0.0
    phase_voltage_a_sensor_state: float = 0.0
    phase_voltage_b_sensor_state: float = 0.0
    phase_voltage_c_sensor_state: float = 0.0
    motor_speed_sensor_state: float = 0.0
    position_sensor_state: float = 0.0
    motor_position_sensor_state: float = 0.0
    voltage_sensor_state: float = 0.0
    bus_current_sensor_state: float = 0.0
    load_speed_estimate_state: float = 0.0
    load_accel_estimate_state: float = 0.0
    electrical_angle_obs_state: float = 0.0
    prev_position_measurement: float = 0.0
    prev_motor_encoder_measurement: float = 0.0
    prev_load_speed_estimate: float = 0.0
    vibration_band_state: float = 0.0
    vibration_envelope_state: float = 0.0
    thermal_preheat_applied: bool = False


@dataclass
class MeasurementSnapshot:
    phase_current_a_meas: float
    phase_current_b_meas: float
    phase_current_c_meas: float
    i_d_meas: float
    i_q_meas: float
    i_meas: float
    phase_voltage_a_meas: float
    phase_voltage_b_meas: float
    phase_voltage_c_meas: float
    omega_m_meas: float
    theta_meas: float
    theta_motor_meas: float
    omega_l_est: float
    load_accel_est: float
    u_meas: float
    bus_current_meas: float
    theta_e_obs: float
    encoder_count: int
    motor_encoder_count: int
    sensor_active: bool
    current_sensor_fault_active: bool
    speed_sensor_fault_active: bool
    position_sensor_fault_active: bool
    motor_encoder_fault_active: bool


@dataclass
class PlantStepSignals:
    torque_em: float
    torque_shaft: float
    torque_load: float
    torque_friction: float
    back_emf_v: float
    available_bus_voltage_v: float
    pwm_duty: float
    electrical_power_w: float
    mechanical_power_w: float
    copper_loss_w: float
    iron_loss_w: float
    supply_current_est_a: float
    bus_current_true_a: float
    winding_temp_c: float
    housing_temp_c: float
    motor_accel_rad_s2: float
    load_accel_rad_s2: float
    shaft_twist_rad: float
    vibration_accel_mps2: float
    phase_current_a: float
    phase_current_b: float
    phase_current_c: float
    phase_voltage_a: float
    phase_voltage_b: float
    phase_voltage_c: float
    phase_current_neutral_a: float
    phase_current_imbalance_a: float
    winding_temp_rate_c_s: float
    housing_temp_rate_c_s: float
    vibration_band_mps2: float
    vibration_envelope_mps2: float
    vibration_shock_index: float
    airspeed_mps: float
    load_fault_active: bool
    friction_fault_active: bool
    jam_fault_active: bool
    electrical_fault_active: bool
    mechanical_fault_active: bool
    demagnetization_fault_active: bool
    inverter_fault_active: bool
    bearing_fault_active: bool


def ramp_progress(start_time: float | None, ramp_s: float, t: float) -> float:
    if start_time is None or t < start_time:
        return 0.0
    if ramp_s <= 0.0:
        return 1.0
    return min((t - start_time) / ramp_s, 1.0)


def ramp_scale(start_time: float | None, ramp_s: float, target_scale: float, t: float) -> float:
    progress = ramp_progress(start_time, ramp_s, t)
    return 1.0 + progress * (target_scale - 1.0)


def ramp_addition(start_time: float | None, ramp_s: float, target_value: float, t: float) -> float:
    return ramp_progress(start_time, ramp_s, t) * target_value


def apply_backlash(relative_angle: float, gap: float) -> float:
    if abs(relative_angle) <= gap:
        return 0.0
    return np.sign(relative_angle) * (abs(relative_angle) - gap)


def first_order_update(state_value: float, target_value: float, tau: float, dt: float) -> float:
    if tau <= 0.0:
        return target_value
    alpha = min(dt / tau, 1.0)
    return state_value + alpha * (target_value - state_value)


def quantize(value: float, resolution: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if resolution <= 0.0:
        return value
    return round(value / resolution) * resolution


def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def dq_to_abc(d_value: float, q_value: float, theta_e: float) -> tuple[float, float, float]:
    cos_t = np.cos(theta_e)
    sin_t = np.sin(theta_e)
    alpha = d_value * cos_t - q_value * sin_t
    beta = d_value * sin_t + q_value * cos_t
    phase_a = alpha
    phase_b = -0.5 * alpha + (np.sqrt(3.0) / 2.0) * beta
    phase_c = -0.5 * alpha - (np.sqrt(3.0) / 2.0) * beta
    return float(phase_a), float(phase_b), float(phase_c)


def abc_to_dq(phase_a: float, phase_b: float, phase_c: float, theta_e: float) -> tuple[float, float]:
    cos_a = np.cos(theta_e)
    sin_a = np.sin(theta_e)
    cos_b = np.cos(theta_e - 2.0 * np.pi / 3.0)
    sin_b = np.sin(theta_e - 2.0 * np.pi / 3.0)
    cos_c = np.cos(theta_e + 2.0 * np.pi / 3.0)
    sin_c = np.sin(theta_e + 2.0 * np.pi / 3.0)
    d_value = (2.0 / 3.0) * (phase_a * cos_a + phase_b * cos_b + phase_c * cos_c)
    q_value = -(2.0 / 3.0) * (phase_a * sin_a + phase_b * sin_b + phase_c * sin_c)
    return float(d_value), float(q_value)


def periodic_window_active(start_time: float | None, duration_s: float | None, period_s: float | None, t: float) -> bool:
    if start_time is None or t < start_time:
        return False
    if duration_s is None:
        return True
    if period_s is None:
        return start_time <= t < start_time + duration_s
    cycle_t = (t - start_time) % period_s
    return cycle_t < duration_s


def jam_window_active(scenario: FaultScenario, t: float) -> bool:
    return periodic_window_active(scenario.jam_start, scenario.jam_duration_s, scenario.jam_period_s, t)


def condition_airspeed(condition: OperatingCondition, t: float) -> float:
    base = condition.airspeed
    if condition.airspeed_profile_name == "steady":
        return base
    if condition.airspeed_profile_name == "gusting":
        return max(5.0, base + 8.0 * np.sin(2.0 * np.pi * 0.25 * t) + 3.5 * np.sin(2.0 * np.pi * 1.1 * t))
    if condition.airspeed_profile_name == "climb":
        return base + 12.0 * min(max(t / 2.0, 0.0), 1.0)
    raise ValueError(f"Unsupported airspeed profile: {condition.airspeed_profile_name}")


class DiagnosticServoPlant:
    def __init__(self, params: ServoPlantParams, condition: OperatingCondition) -> None:
        self.params = params
        self.condition = condition
        self.state = PlantState(
            winding_temp_c=condition.winding_temp_init_c,
            housing_temp_c=condition.housing_temp_init_c,
        )
        self.jammed_angle: float | None = None

    def electrical_angle(self) -> float:
        return wrap_to_pi(self.params.pole_pairs * self.state.theta_m)

    def phase_currents(self) -> tuple[float, float, float]:
        return dq_to_abc(self.state.i_d, self.state.i_q, self.electrical_angle())

    def phase_voltages(self) -> tuple[float, float, float]:
        return dq_to_abc(self.state.v_d_applied, self.state.v_q_applied, self.electrical_angle())

    def faulted_phase_currents(self, scenario: FaultScenario, t: float) -> tuple[float, float, float]:
        phase_current_a, phase_current_b, phase_current_c = self.phase_currents()
        imbalance_progress = ramp_progress(scenario.resistance_scale_start, scenario.resistance_ramp_s, t)
        if imbalance_progress <= 0.0 or scenario.resistance_phase_imbalance_gain <= 0.0:
            return phase_current_a, phase_current_b, phase_current_c
        imbalance_amp = (
            scenario.resistance_phase_imbalance_gain
            * imbalance_progress
            * max(scenario.resistance_scale - 1.0, 0.0)
            * max(abs(self.state.i_q) + 0.35 * abs(self.state.i_d), 0.7)
        )
        phase_currents = np.array(
            [
                phase_current_a + imbalance_amp,
                phase_current_b - 0.35 * imbalance_amp,
                phase_current_c - 0.65 * imbalance_amp,
            ],
            dtype=np.float64,
        )
        phase_currents -= phase_currents.mean()
        return float(phase_currents[0]), float(phase_currents[1]), float(phase_currents[2])

    def get_available_bus_voltage(self, scenario: FaultScenario, t: float) -> float:
        voltage_scale = ramp_scale(scenario.bus_voltage_scale_start, scenario.bus_voltage_ramp_s, scenario.bus_voltage_scale, t)
        return self.condition.bus_voltage * voltage_scale

    def thermal_overtemp(self, scenario: FaultScenario, t: float) -> float:
        if scenario.thermal_derate_start is None or t < scenario.thermal_derate_start:
            return 0.0
        return max(self.state.winding_temp_c - scenario.thermal_limit_temp_c, 0.0)

    def get_available_current_limit(self, scenario: FaultScenario, t: float) -> float:
        over_temp = self.thermal_overtemp(scenario, t)
        if over_temp <= 0.0 or scenario.thermal_current_limit_gain <= 0.0:
            return self.params.current_limit
        scale = max(
            scenario.thermal_current_limit_min_scale,
            1.0 - scenario.thermal_current_limit_gain * over_temp,
        )
        return self.params.current_limit * scale

    def effective_electrical_params(self, scenario: FaultScenario, t: float) -> tuple[float, float, float]:
        resistance_scale = ramp_scale(scenario.resistance_scale_start, scenario.resistance_ramp_s, scenario.resistance_scale, t)
        flux_scale = ramp_scale(scenario.flux_scale_start, scenario.flux_ramp_s, scenario.flux_scale, t)
        winding_temp_delta = self.state.winding_temp_c - 25.0
        effective_resistance = self.params.phase_resistance * (1.0 + self.params.resistance_temp_coeff * winding_temp_delta)
        effective_resistance *= resistance_scale

        base_flux_scale = max(0.72, 1.0 + self.params.flux_temp_coeff * winding_temp_delta)
        effective_flux = self.params.flux_linkage * base_flux_scale * flux_scale

        over_temp = self.thermal_overtemp(scenario, t)
        if over_temp > 0.0:
            effective_resistance *= 1.0 + scenario.thermal_resistance_rise_gain * over_temp
            effective_flux *= max(0.5, 1.0 - scenario.thermal_flux_drop_gain * over_temp)

        return effective_resistance, effective_flux, over_temp

    def observe_electrical_angle(self, theta_motor_encoder: float) -> float:
        theta_e_raw = wrap_to_pi(self.params.pole_pairs * theta_motor_encoder)
        angle_error = wrap_to_pi(theta_e_raw - self.state.electrical_angle_obs_state)
        self.state.electrical_angle_obs_state = wrap_to_pi(
            self.state.electrical_angle_obs_state
            + min(self.params.dt / max(self.params.electrical_angle_observer_tau, 1e-6), 1.0) * angle_error
        )
        return self.state.electrical_angle_obs_state

    def measure(self, scenario: FaultScenario, t: float, rng: np.random.Generator) -> MeasurementSnapshot:
        current_fault_progress = ramp_progress(scenario.current_sensor_bias_start, scenario.current_sensor_bias_ramp_s, t)
        speed_fault_progress = ramp_progress(scenario.speed_sensor_scale_start, scenario.speed_sensor_scale_ramp_s, t)
        position_fault_progress = ramp_progress(scenario.position_sensor_bias_start, scenario.position_sensor_bias_ramp_s, t)
        inverter_fault_progress = ramp_progress(scenario.inverter_voltage_scale_start, scenario.inverter_voltage_ramp_s, t)
        motor_encoder_fault_active = periodic_window_active(
            scenario.motor_encoder_freeze_start,
            scenario.motor_encoder_freeze_duration_s,
            scenario.motor_encoder_freeze_period_s,
            t,
        )

        phase_current_a_true, phase_current_b_true, phase_current_c_true = self.faulted_phase_currents(scenario, t)
        self.state.phase_current_a_sensor_state = first_order_update(
            self.state.phase_current_a_sensor_state,
            phase_current_a_true,
            self.params.current_sensor_tau,
            self.params.dt,
        )
        self.state.phase_current_b_sensor_state = first_order_update(
            self.state.phase_current_b_sensor_state,
            phase_current_b_true,
            self.params.current_sensor_tau,
            self.params.dt,
        )
        self.state.phase_current_c_sensor_state = first_order_update(
            self.state.phase_current_c_sensor_state,
            phase_current_c_true,
            self.params.current_sensor_tau,
            self.params.dt,
        )
        self.state.motor_speed_sensor_state = first_order_update(
            self.state.motor_speed_sensor_state,
            self.state.omega_m,
            self.params.motor_speed_sensor_tau,
            self.params.dt,
        )
        self.state.position_sensor_state = first_order_update(
            self.state.position_sensor_state,
            self.state.theta_l,
            self.params.position_sensor_tau,
            self.params.dt,
        )
        self.state.motor_position_sensor_state = first_order_update(
            self.state.motor_position_sensor_state,
            self.state.theta_m,
            self.params.motor_position_sensor_tau,
            self.params.dt,
        )
        phase_voltage_a_true, phase_voltage_b_true, phase_voltage_c_true = self.phase_voltages()
        self.state.phase_voltage_a_sensor_state = first_order_update(
            self.state.phase_voltage_a_sensor_state,
            phase_voltage_a_true,
            self.params.voltage_sensor_tau,
            self.params.dt,
        )
        self.state.phase_voltage_b_sensor_state = first_order_update(
            self.state.phase_voltage_b_sensor_state,
            phase_voltage_b_true,
            self.params.voltage_sensor_tau,
            self.params.dt,
        )
        self.state.phase_voltage_c_sensor_state = first_order_update(
            self.state.phase_voltage_c_sensor_state,
            phase_voltage_c_true,
            self.params.voltage_sensor_tau,
            self.params.dt,
        )
        voltage_mag_true = float(np.hypot(self.state.v_d_applied, self.state.v_q_applied))
        inverter_ripple = (
            scenario.inverter_voltage_ripple_gain
            * inverter_fault_progress
            * voltage_mag_true
            * np.sin(2.0 * np.pi * (18.0 + 0.015 * abs(self.state.omega_m)) * t)
        )
        measured_voltage_target = max(
            0.0,
            voltage_mag_true * (1.0 - scenario.inverter_voltage_meas_drop_gain * inverter_fault_progress) + inverter_ripple,
        )
        self.state.voltage_sensor_state = first_order_update(
            self.state.voltage_sensor_state,
            measured_voltage_target,
            self.params.voltage_sensor_tau,
            self.params.dt,
        )

        available_bus_voltage = self.get_available_bus_voltage(scenario, t)
        bus_current_true = self.instantaneous_bus_current(available_bus_voltage)
        self.state.bus_current_sensor_state = first_order_update(
            self.state.bus_current_sensor_state,
            bus_current_true,
            self.params.current_sensor_tau,
            self.params.dt,
        )

        current_sensor_bias = current_fault_progress * scenario.current_sensor_bias
        speed_sensor_scale = 1.0 + speed_fault_progress * (scenario.speed_sensor_scale - 1.0)
        speed_sensor_bias = np.deg2rad(speed_fault_progress * scenario.speed_sensor_bias_deg_s)
        speed_sensor_ripple = (
            scenario.speed_sensor_ripple_gain
            * speed_fault_progress
            * max(abs(self.state.motor_speed_sensor_state), np.deg2rad(40.0))
            * np.sin(2.0 * np.pi * (4.0 + 0.01 * abs(self.state.omega_m)) * t + 0.35)
        )
        theta_bias = np.deg2rad(position_fault_progress * scenario.position_sensor_bias)
        load_encoder_resolution = np.deg2rad(self.params.encoder_resolution_deg)
        motor_encoder_resolution = np.deg2rad(self.params.motor_encoder_resolution_deg)

        analog_position = self.state.position_sensor_state + theta_bias + rng.normal(0.0, self.params.position_noise_std)
        theta_encoder = quantize(analog_position, load_encoder_resolution)
        encoder_count = int(round(theta_encoder / load_encoder_resolution)) if load_encoder_resolution > 0.0 else 0

        analog_motor_position = self.state.motor_position_sensor_state + rng.normal(0.0, self.params.position_noise_std)
        theta_motor_encoder_candidate = quantize(analog_motor_position, motor_encoder_resolution)
        if motor_encoder_fault_active:
            theta_motor_encoder = self.state.prev_motor_encoder_measurement
        else:
            theta_motor_encoder = theta_motor_encoder_candidate
            self.state.prev_motor_encoder_measurement = theta_motor_encoder
        motor_encoder_count = int(round(theta_motor_encoder / motor_encoder_resolution)) if motor_encoder_resolution > 0.0 else 0
        theta_e_obs = self.observe_electrical_angle(theta_motor_encoder)

        raw_load_speed = (theta_encoder - self.state.prev_position_measurement) / self.params.dt
        self.state.load_speed_estimate_state = first_order_update(
            self.state.load_speed_estimate_state,
            raw_load_speed,
            self.params.load_speed_estimator_tau,
            self.params.dt,
        )
        self.state.prev_position_measurement = theta_encoder
        raw_load_accel = (self.state.load_speed_estimate_state - self.state.prev_load_speed_estimate) / self.params.dt
        self.state.load_accel_estimate_state = first_order_update(
            self.state.load_accel_estimate_state,
            raw_load_accel,
            self.params.load_speed_estimator_tau,
            self.params.dt,
        )
        self.state.prev_load_speed_estimate = self.state.load_speed_estimate_state

        phase_current_a_meas = self.state.phase_current_a_sensor_state + current_sensor_bias + rng.normal(0.0, self.params.phase_current_noise_std)
        phase_current_b_meas = self.state.phase_current_b_sensor_state + rng.normal(0.0, self.params.phase_current_noise_std)
        phase_current_c_meas = self.state.phase_current_c_sensor_state + rng.normal(0.0, self.params.phase_current_noise_std)
        phase_voltage_ripple = inverter_ripple
        phase_voltage_asym = scenario.inverter_phase_asymmetry_gain * inverter_fault_progress * max(voltage_mag_true, 1.0)
        phase_voltage_a_meas = (
            self.state.phase_voltage_a_sensor_state * (1.0 - 0.60 * scenario.inverter_voltage_meas_drop_gain * inverter_fault_progress)
            + phase_voltage_ripple
            + phase_voltage_asym
            + rng.normal(0.0, self.params.voltage_noise_std)
        )
        phase_voltage_b_meas = (
            self.state.phase_voltage_b_sensor_state * (1.0 - 0.45 * scenario.inverter_voltage_meas_drop_gain * inverter_fault_progress)
            - 0.55 * phase_voltage_ripple
            - 0.35 * phase_voltage_asym
            + rng.normal(0.0, self.params.voltage_noise_std)
        )
        phase_voltage_c_meas = (
            self.state.phase_voltage_c_sensor_state * (1.0 - 0.75 * scenario.inverter_voltage_meas_drop_gain * inverter_fault_progress)
            - 0.45 * phase_voltage_ripple
            - 0.65 * phase_voltage_asym
            + rng.normal(0.0, self.params.voltage_noise_std)
        )
        i_d_meas, i_q_meas = abc_to_dq(phase_current_a_meas, phase_current_b_meas, phase_current_c_meas, theta_e_obs)

        current_fault_active = current_fault_progress > 0.0
        speed_fault_active = speed_fault_progress > 0.0
        position_fault_active = position_fault_progress > 0.0
        sensor_active = current_fault_active or speed_fault_active or position_fault_active or motor_encoder_fault_active

        return MeasurementSnapshot(
            phase_current_a_meas=phase_current_a_meas,
            phase_current_b_meas=phase_current_b_meas,
            phase_current_c_meas=phase_current_c_meas,
            i_d_meas=i_d_meas,
            i_q_meas=i_q_meas,
            i_meas=i_q_meas,
            phase_voltage_a_meas=phase_voltage_a_meas,
            phase_voltage_b_meas=phase_voltage_b_meas,
            phase_voltage_c_meas=phase_voltage_c_meas,
            omega_m_meas=self.state.motor_speed_sensor_state * speed_sensor_scale + speed_sensor_bias + speed_sensor_ripple + rng.normal(0.0, self.params.speed_noise_std),
            theta_meas=theta_encoder,
            theta_motor_meas=theta_motor_encoder,
            omega_l_est=self.state.load_speed_estimate_state,
            load_accel_est=self.state.load_accel_estimate_state,
            u_meas=self.state.voltage_sensor_state + rng.normal(0.0, self.params.voltage_noise_std),
            bus_current_meas=self.state.bus_current_sensor_state + rng.normal(0.0, self.params.bus_current_noise_std),
            theta_e_obs=theta_e_obs,
            encoder_count=encoder_count,
            motor_encoder_count=motor_encoder_count,
            sensor_active=sensor_active,
            current_sensor_fault_active=current_fault_active,
            speed_sensor_fault_active=speed_fault_active,
            position_sensor_fault_active=position_fault_active,
            motor_encoder_fault_active=motor_encoder_fault_active,
        )

    def instantaneous_bus_current(self, bus_voltage: float) -> float:
        electrical_power = 1.5 * (self.state.v_d_applied * self.state.i_d + self.state.v_q_applied * self.state.i_q)
        return electrical_power / max(bus_voltage, 1e-6)

    def step(self, scenario: FaultScenario, t: float, v_d_cmd: float, v_q_cmd: float, rng: np.random.Generator) -> PlantStepSignals:
        if not self.state.thermal_preheat_applied and scenario.thermal_preheat_delta_c > 0.0:
            self.state.winding_temp_c += scenario.thermal_preheat_delta_c
            self.state.housing_temp_c += 0.5 * scenario.thermal_preheat_delta_c
            self.state.thermal_preheat_applied = True

        available_bus_voltage_base = self.get_available_bus_voltage(scenario, t)
        resistance_progress = ramp_progress(scenario.resistance_scale_start, scenario.resistance_ramp_s, t)
        demagnetization_progress = ramp_progress(scenario.flux_scale_start, scenario.flux_ramp_s, t)
        demagnetization_severity = demagnetization_progress * max(1.0 - scenario.flux_scale, 0.0)
        inverter_progress = ramp_progress(scenario.inverter_voltage_scale_start, scenario.inverter_voltage_ramp_s, t)
        resistance_bus_drop = (
            scenario.resistance_voltage_collapse_gain
            * resistance_progress
            * max(scenario.resistance_scale - 1.0, 0.0)
            * (abs(self.state.i_q) + 0.5 * abs(self.state.i_d))
        )
        available_bus_voltage = max(0.58 * available_bus_voltage_base, available_bus_voltage_base - resistance_bus_drop)
        vector_limit = available_bus_voltage * self.params.modulation_index_limit / np.sqrt(3.0)
        vector_mag_cmd = float(np.hypot(v_d_cmd, v_q_cmd))
        if vector_mag_cmd > vector_limit:
            scale = vector_limit / max(vector_mag_cmd, 1e-9)
            v_d_target = v_d_cmd * scale
            v_q_target = v_q_cmd * scale
        else:
            v_d_target = v_d_cmd
            v_q_target = v_q_cmd
        inverter_scale = ramp_scale(
            scenario.inverter_voltage_scale_start,
            scenario.inverter_voltage_ramp_s,
            scenario.inverter_voltage_scale,
            t,
        )
        resistive_drop = scenario.resistance_voltage_drop_gain * resistance_progress * (abs(self.state.i_q) + 0.35 * abs(self.state.i_d))
        resistive_drop_scale = max(0.55, 1.0 - resistive_drop)
        v_d_target *= inverter_scale
        v_q_target *= inverter_scale
        inverter_voltage_ripple = (
            scenario.inverter_voltage_ripple_gain
            * inverter_progress
            * vector_limit
            * np.sin(2.0 * np.pi * (18.0 + 0.012 * abs(self.state.omega_m)) * t)
        )
        inverter_voltage_asym = (
            scenario.inverter_phase_asymmetry_gain
            * inverter_progress
            * vector_limit
            * np.sin(2.0 * np.pi * 3.0 * t + 0.35)
        )
        v_d_target += 0.20 * inverter_voltage_asym - 0.18 * inverter_voltage_ripple
        v_q_target += inverter_voltage_ripple - 0.12 * inverter_voltage_asym
        v_d_target *= resistive_drop_scale
        v_q_target *= resistive_drop_scale
        inverter_clamp = max(0.42, 1.0 - scenario.inverter_pwm_clamp_gain * inverter_progress)
        v_d_target = float(np.clip(v_d_target, -inverter_clamp * vector_limit, inverter_clamp * vector_limit))
        v_q_target = float(np.clip(v_q_target, -inverter_clamp * vector_limit, inverter_clamp * vector_limit))
        self.state.v_d_applied = first_order_update(self.state.v_d_applied, v_d_target, self.params.inverter_tau, self.params.dt)
        self.state.v_q_applied = first_order_update(self.state.v_q_applied, v_q_target, self.params.inverter_tau, self.params.dt)
        pwm_duty = float(np.hypot(self.state.v_d_applied, self.state.v_q_applied) / max(vector_limit, 1e-6))
        if inverter_progress > 0.0:
            pwm_duty = float(np.clip(pwm_duty + 0.22 * scenario.inverter_pwm_clamp_gain * inverter_progress, 0.0, 1.0))

        friction_coulomb = self.params.motor_coulomb + ramp_addition(scenario.friction_start, scenario.friction_ramp_s, scenario.extra_coulomb, t)
        friction_viscous = self.params.motor_viscous + ramp_addition(scenario.friction_start, scenario.friction_ramp_s, scenario.extra_viscous, t)
        friction_active = ramp_progress(scenario.friction_start, scenario.friction_ramp_s, t) > 0.0
        theta_motor_out = self.state.theta_m / self.params.gear_ratio
        omega_motor_out = self.state.omega_m / self.params.gear_ratio
        bearing_progress = ramp_progress(scenario.bearing_fault_start, scenario.bearing_ramp_s, t)
        bearing_torque = 0.0
        bearing_signature = 0.0
        bearing_harmonic = 0.0
        bearing_sideband = 0.0
        bearing_impulse = 0.0
        bearing_envelope = 0.0
        if bearing_progress > 0.0 and scenario.bearing_freq_hz > 0.0:
            elapsed = t - scenario.bearing_fault_start
            shaft_freq_hz = max(abs(omega_motor_out) / (2.0 * np.pi), 0.5)
            bpfo_phase = 2.0 * np.pi * scenario.bearing_freq_hz * elapsed
            harm_phase = 2.0 * bpfo_phase
            sideband_low_phase = 2.0 * np.pi * max(scenario.bearing_freq_hz - shaft_freq_hz, 1.0) * elapsed
            sideband_high_phase = 2.0 * np.pi * (scenario.bearing_freq_hz + shaft_freq_hz) * elapsed
            bearing_torque = bearing_progress * scenario.bearing_torque_amp * np.sin(bpfo_phase)
            bearing_harmonic = (
                bearing_progress
                * scenario.bearing_torque_amp
                * scenario.bearing_harmonic_ratio
                * np.sin(harm_phase + 0.35)
            )
            bearing_sideband = (
                bearing_progress
                * scenario.bearing_torque_amp
                * scenario.bearing_sideband_ratio
                * 0.5
                * (np.sin(sideband_low_phase + 0.15) + np.sin(sideband_high_phase - 0.20))
            )
            pulse_core = max(0.0, np.sin(bpfo_phase))
            impulse_shape = pulse_core**10
            bearing_impulse = (
                bearing_progress
                * scenario.bearing_torque_amp
                * scenario.bearing_impulse_gain
                * impulse_shape
            )
            bearing_envelope = bearing_progress * (1.0 + scenario.bearing_envelope_gain * (0.5 + 0.5 * np.sin(2.0 * np.pi * shaft_freq_hz * elapsed)))
            bearing_signature = bearing_impulse + 0.35 * abs(bearing_harmonic) + 0.30 * abs(bearing_sideband)
        torque_friction = (
            friction_coulomb * np.tanh(self.state.omega_m / self.params.friction_smoothing)
            + friction_viscous * self.state.omega_m
            + bearing_torque
            + 0.45 * bearing_harmonic
            + 0.40 * bearing_sideband
            + 0.30 * bearing_impulse
        )

        backlash_scale = ramp_scale(scenario.backlash_scale_start, scenario.backlash_ramp_s, scenario.backlash_scale, t)
        shaft_deflection = theta_motor_out - self.state.theta_l
        shaft_speed_delta = omega_motor_out - self.state.omega_l
        effective_deflection = apply_backlash(shaft_deflection, self.params.backlash * backlash_scale)
        shaft_torque = self.params.shaft_stiffness * effective_deflection + self.params.shaft_damping * shaft_speed_delta
        backlash_progress = ramp_progress(scenario.backlash_scale_start, scenario.backlash_ramp_s, t)
        if backlash_progress > 0.0:
            backlash_rattle = (
                scenario.backlash_rattle_amp
                * backlash_progress
                * np.sin(2.0 * np.pi * scenario.backlash_rattle_freq * t)
            )
            backlash_drag = (
                scenario.backlash_drag_gain
                * backlash_progress
                * np.tanh(4.0 * shaft_speed_delta)
                * (0.4 + abs(shaft_speed_delta))
            )
            shaft_torque += backlash_rattle - backlash_drag
        shaft_torque = float(np.clip(shaft_torque, -self.params.max_shaft_torque, self.params.max_shaft_torque))

        torque_gust = 0.0
        torque_sustained = 0.0
        torque_torsion = 0.0
        load_fault_active = ramp_progress(scenario.extra_load_start, scenario.extra_load_ramp_s, t) > 0.0 or (scenario.gust_start is not None and t >= scenario.gust_start)
        if scenario.gust_start is not None and t >= scenario.gust_start:
            torque_gust = scenario.gust_amp * np.sin(2.0 * np.pi * scenario.gust_freq * (t - scenario.gust_start))
        extra_load = ramp_addition(scenario.extra_load_start, scenario.extra_load_ramp_s, scenario.extra_load_torque, t)
        if scenario.extra_load_start is not None and t >= scenario.extra_load_start:
            sustained_progress = max(ramp_progress(scenario.extra_load_start, scenario.extra_load_ramp_s, t), 0.25)
            elapsed_load = t - scenario.extra_load_start
            torque_sustained = sustained_progress * (
                scenario.sustained_load_bias
                + scenario.sustained_load_wave_amp * np.sin(2.0 * np.pi * scenario.sustained_load_wave_freq * elapsed_load)
            )
            if scenario.load_torsion_amp > 0.0 and scenario.load_torsion_freq > 0.0:
                torque_torsion = sustained_progress * scenario.load_torsion_amp * (
                    np.sin(2.0 * np.pi * scenario.load_torsion_freq * elapsed_load)
                    + 0.35 * np.sin(2.0 * np.pi * 0.5 * scenario.load_torsion_freq * elapsed_load + 0.4)
                )

        airspeed = condition_airspeed(self.condition, t)
        torque_aero = self.params.aero_stiffness * (airspeed**2) * self.state.theta_l
        torque_damping = (self.params.aero_damping + self.params.load_viscous) * self.state.omega_l
        torque_load = self.condition.load_bias_torque + torque_aero + torque_damping + torque_gust + extra_load + torque_sustained + torque_torsion

        effective_resistance, effective_flux, over_temp = self.effective_electrical_params(scenario, t)
        omega_e = self.params.pole_pairs * self.state.omega_m
        back_emf_v = abs(omega_e * effective_flux)
        did = (
            self.state.v_d_applied
            - effective_resistance * self.state.i_d
            + omega_e * self.params.q_axis_inductance * self.state.i_q
        ) / self.params.d_axis_inductance
        diq = (
            self.state.v_q_applied
            - effective_resistance * self.state.i_q
            - omega_e * (self.params.d_axis_inductance * self.state.i_d + effective_flux)
        ) / self.params.q_axis_inductance
        demagnetization_current_drag = (
            scenario.flux_current_boost_gain
            * demagnetization_severity
            * (abs(omega_e) / max(self.params.pole_pairs * 50.0, 1.0) + 0.65 * abs(self.state.i_q))
        )
        diq -= demagnetization_current_drag * np.sign(self.state.i_q if abs(self.state.i_q) > 1.0e-6 else v_q_target)
        torque_em = 1.5 * self.params.pole_pairs * (
            effective_flux * self.state.i_q + (self.params.d_axis_inductance - self.params.q_axis_inductance) * self.state.i_d * self.state.i_q
        )
        torque_em *= max(0.52, 1.0 - scenario.flux_power_gap_gain * demagnetization_severity)
        motor_side_load = shaft_torque / max(self.params.gear_ratio * self.params.gear_efficiency, 1e-6)
        motor_side_load += scenario.flux_tracking_drag_gain * demagnetization_severity * (0.35 + 0.15 * abs(self.state.i_q))
        domega_m = (torque_em - torque_friction - motor_side_load) / self.params.motor_inertia

        jam_active = jam_window_active(scenario, t)
        if jam_active:
            if self.jammed_angle is None:
                self.jammed_angle = self.state.theta_l
            self.state.theta_l = self.jammed_angle
            self.state.omega_l = 0.0
            domega_l = 0.0
        else:
            self.jammed_angle = None
            domega_l = (self.params.gear_efficiency * shaft_torque - torque_load) / self.params.load_inertia
            self.state.omega_l += domega_l * self.params.dt
            self.state.theta_l += self.state.omega_l * self.params.dt

        dynamic_current_limit = max(0.5, 1.5 * self.get_available_current_limit(scenario, t))
        self.state.i_d = float(np.clip(self.state.i_d + did * self.params.dt, -dynamic_current_limit, dynamic_current_limit))
        self.state.i_q = float(np.clip(self.state.i_q + diq * self.params.dt, -dynamic_current_limit, dynamic_current_limit))
        self.state.omega_m = float(np.clip(self.state.omega_m + domega_m * self.params.dt, -400.0, 400.0))
        self.state.theta_m += self.state.omega_m * self.params.dt
        self.state.omega_l = float(np.clip(self.state.omega_l, -12.0, 12.0))

        phase_current_a, phase_current_b, phase_current_c = self.faulted_phase_currents(scenario, t)
        phase_voltage_a, phase_voltage_b, phase_voltage_c = self.phase_voltages()
        electrical_power_w = 1.5 * (self.state.v_d_applied * self.state.i_d + self.state.v_q_applied * self.state.i_q)
        bus_current_true = electrical_power_w / max(available_bus_voltage, 1e-6)
        copper_loss_w = 1.5 * effective_resistance * (self.state.i_d**2 + self.state.i_q**2)
        iron_loss_w = self.params.iron_loss_coeff * (self.state.omega_m**2)
        if resistance_progress > 0.0 and scenario.resistance_extra_heat_gain > 0.0:
            copper_loss_w *= 1.0 + scenario.resistance_extra_heat_gain * resistance_progress * max(scenario.resistance_scale - 1.0, 0.0)
        thermal_progress = ramp_progress(scenario.thermal_derate_start, 0.0, t)
        if thermal_progress > 0.0:
            copper_loss_w *= 1.0 + scenario.thermal_copper_loss_gain * thermal_progress
            iron_loss_w *= 1.0 + scenario.thermal_iron_loss_gain * thermal_progress
        mech_loss_w = abs(torque_friction * self.state.omega_m)
        cooling_scale = max(scenario.cooling_fault_scale, 1.0e-3)
        winding_to_housing_resistance = self.params.winding_to_housing_resistance_c_per_w / cooling_scale
        housing_to_ambient_resistance = self.params.housing_to_ambient_resistance_c_per_w / cooling_scale
        winding_to_housing_w = (self.state.winding_temp_c - self.state.housing_temp_c) / winding_to_housing_resistance
        housing_to_ambient_w = (self.state.housing_temp_c - self.condition.ambient_temp_c) / housing_to_ambient_resistance

        dtemp_winding = (copper_loss_w + iron_loss_w + mech_loss_w - winding_to_housing_w) / self.params.winding_thermal_capacity_j_per_c
        dtemp_housing = (winding_to_housing_w - housing_to_ambient_w) / self.params.housing_thermal_capacity_j_per_c
        winding_temp_rate_c_s = dtemp_winding
        housing_temp_rate_c_s = dtemp_housing
        self.state.winding_temp_c += dtemp_winding * self.params.dt
        self.state.housing_temp_c += dtemp_housing * self.params.dt

        mechanical_power_w = torque_em * self.state.omega_m
        torque_ripple = (shaft_torque - self.state.prev_shaft_torque) / self.params.dt
        vibration_band_input = (
            0.008 * torque_ripple
            + 0.0025 * domega_m
            + 0.0012 * domega_l
            + 9.0 * bearing_torque
            + 14.0 * bearing_harmonic
            + 18.0 * bearing_sideband
            + 72.0 * bearing_signature
        )
        self.state.vibration_band_state = first_order_update(
            self.state.vibration_band_state,
            vibration_band_input,
            0.0065,
            self.params.dt,
        )
        self.state.vibration_envelope_state = first_order_update(
            self.state.vibration_envelope_state,
            bearing_envelope * abs(self.state.vibration_band_state) + 2.5 * bearing_signature,
            0.012,
            self.params.dt,
        )
        vibration_shock_index = (
            abs(torque_ripple) / (abs(shaft_torque) + 0.05)
            + 5.0 * bearing_signature
            + 1.2 * abs(bearing_harmonic)
            + 1.5 * abs(bearing_sideband)
        )
        vibration_shock_index = float(np.clip(vibration_shock_index, 0.0, 500.0))
        vibration_accel = 0.003 * abs(domega_m) + 0.001 * abs(domega_l) + 0.008 * abs(torque_ripple)
        vibration_accel += 0.04 * over_temp
        vibration_accel += 14.0 * abs(bearing_torque) + 18.0 * abs(bearing_harmonic) + 24.0 * abs(bearing_sideband) + 175.0 * bearing_signature
        vibration_accel += rng.normal(0.0, self.params.vibration_noise_std)
        self.state.prev_shaft_torque = shaft_torque

        demagnetization_fault_active = ramp_progress(scenario.flux_scale_start, scenario.flux_ramp_s, t) > 0.0
        inverter_fault_active = ramp_progress(scenario.inverter_voltage_scale_start, scenario.inverter_voltage_ramp_s, t) > 0.0
        bearing_fault_active = bearing_progress > 0.0
        electrical_fault_active = (
            ramp_progress(scenario.resistance_scale_start, scenario.resistance_ramp_s, t) > 0.0
            or ramp_progress(scenario.bus_voltage_scale_start, scenario.bus_voltage_ramp_s, t) > 0.0
            or demagnetization_fault_active
            or inverter_fault_active
            or over_temp > 0.0
        )
        mechanical_fault_active = ramp_progress(scenario.backlash_scale_start, scenario.backlash_ramp_s, t) > 0.0 or jam_active or bearing_fault_active

        phase_current_neutral_a = phase_current_a + phase_current_b + phase_current_c
        phase_current_imbalance_a = float(np.sqrt(((phase_current_a - phase_current_b) ** 2 + (phase_current_b - phase_current_c) ** 2 + (phase_current_c - phase_current_a) ** 2) / 3.0))
        return PlantStepSignals(
            torque_em=torque_em,
            torque_shaft=shaft_torque,
            torque_load=torque_load,
            torque_friction=torque_friction,
            back_emf_v=back_emf_v,
            available_bus_voltage_v=available_bus_voltage,
            pwm_duty=pwm_duty,
            electrical_power_w=electrical_power_w,
            mechanical_power_w=mechanical_power_w,
            copper_loss_w=copper_loss_w,
            iron_loss_w=iron_loss_w,
            supply_current_est_a=bus_current_true,
            bus_current_true_a=bus_current_true,
            winding_temp_c=self.state.winding_temp_c,
            housing_temp_c=self.state.housing_temp_c,
            motor_accel_rad_s2=domega_m,
            load_accel_rad_s2=domega_l,
            shaft_twist_rad=shaft_deflection,
            vibration_accel_mps2=vibration_accel,
            phase_current_a=phase_current_a,
            phase_current_b=phase_current_b,
            phase_current_c=phase_current_c,
            phase_voltage_a=phase_voltage_a,
            phase_voltage_b=phase_voltage_b,
            phase_voltage_c=phase_voltage_c,
            phase_current_neutral_a=phase_current_neutral_a,
            phase_current_imbalance_a=phase_current_imbalance_a,
            winding_temp_rate_c_s=winding_temp_rate_c_s,
            housing_temp_rate_c_s=housing_temp_rate_c_s,
            vibration_band_mps2=self.state.vibration_band_state,
            vibration_envelope_mps2=self.state.vibration_envelope_state,
            vibration_shock_index=vibration_shock_index,
            airspeed_mps=airspeed,
            load_fault_active=load_fault_active,
            friction_fault_active=friction_active,
            jam_fault_active=jam_active,
            electrical_fault_active=electrical_fault_active,
            mechanical_fault_active=mechanical_fault_active,
            demagnetization_fault_active=demagnetization_fault_active,
            inverter_fault_active=inverter_fault_active,
            bearing_fault_active=bearing_fault_active,
        )
