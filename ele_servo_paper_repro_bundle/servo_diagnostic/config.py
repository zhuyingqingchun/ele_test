from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ServoPlantParams:
    dt: float = 0.001
    final_time: float = 4.0
    random_seed: int = 7

    bus_voltage: float = 28.0
    inverter_tau: float = 0.0015
    modulation_index_limit: float = 0.95

    phase_resistance: float = 0.65
    d_axis_inductance: float = 0.0035
    q_axis_inductance: float = 0.0038
    flux_linkage: float = 0.055
    pole_pairs: int = 4
    resistance_temp_coeff: float = 0.0039
    flux_temp_coeff: float = -0.00045

    motor_inertia: float = 6.0e-5
    motor_viscous: float = 4.0e-4
    motor_coulomb: float = 0.015
    friction_smoothing: float = 2.0

    load_inertia: float = 2.5e-3
    load_viscous: float = 0.04
    gear_ratio: float = 45.0
    gear_efficiency: float = 0.92
    shaft_stiffness: float = 260.0
    shaft_damping: float = 0.55
    backlash: float = np.deg2rad(0.12)
    max_shaft_torque: float = 24.0

    aero_stiffness: float = 0.0028
    aero_damping: float = 0.06

    pos_kp: float = 18.0
    pos_kd: float = 1.2
    speed_kp: float = 0.05
    speed_ki: float = 1.0
    current_kp: float = 1.5
    current_ki: float = 60.0
    current_d_kp: float = 1.2
    current_d_ki: float = 50.0

    pos_speed_limit: float = np.deg2rad(220.0)
    current_limit: float = 8.0

    ambient_temp_c: float = 25.0
    winding_temp_init_c: float = 25.0
    housing_temp_init_c: float = 25.0
    winding_thermal_capacity_j_per_c: float = 45.0
    housing_thermal_capacity_j_per_c: float = 120.0
    winding_to_housing_resistance_c_per_w: float = 0.75
    housing_to_ambient_resistance_c_per_w: float = 1.8
    iron_loss_coeff: float = 2.5e-5

    current_noise_std: float = 0.03
    speed_noise_std: float = np.deg2rad(3.0)
    position_noise_std: float = np.deg2rad(0.04)
    voltage_noise_std: float = 0.05
    phase_current_noise_std: float = 0.02
    bus_current_noise_std: float = 0.03
    vibration_noise_std: float = 0.2
    current_sensor_tau: float = 0.0015
    motor_speed_sensor_tau: float = 0.004
    position_sensor_tau: float = 0.003
    motor_position_sensor_tau: float = 0.0015
    voltage_sensor_tau: float = 0.002
    load_speed_estimator_tau: float = 0.01
    electrical_angle_observer_tau: float = 0.002
    encoder_resolution_deg: float = 0.02
    motor_encoder_resolution_deg: float = 0.05


@dataclass
class FaultScenario:
    name: str
    fault_label: str
    fault_id: int
    extra_load_start: float | None = None
    extra_load_torque: float = 0.0
    extra_load_ramp_s: float = 0.0
    sustained_load_bias: float = 0.0
    sustained_load_wave_amp: float = 0.0
    sustained_load_wave_freq: float = 0.0
    load_torsion_amp: float = 0.0
    load_torsion_freq: float = 0.0
    gust_start: float | None = None
    gust_amp: float = 0.0
    gust_freq: float = 5.0
    friction_start: float | None = None
    extra_coulomb: float = 0.0
    extra_viscous: float = 0.0
    friction_ramp_s: float = 0.0
    jam_start: float | None = None
    jam_duration_s: float | None = None
    jam_period_s: float | None = None
    current_sensor_bias_start: float | None = None
    current_sensor_bias: float = 0.0
    current_sensor_bias_ramp_s: float = 0.0
    speed_sensor_scale_start: float | None = None
    speed_sensor_scale: float = 1.0
    speed_sensor_scale_ramp_s: float = 0.0
    speed_sensor_bias_deg_s: float = 0.0
    speed_sensor_ripple_gain: float = 0.0
    position_sensor_bias_start: float | None = None
    position_sensor_bias: float = 0.0
    position_sensor_bias_ramp_s: float = 0.0
    motor_encoder_freeze_start: float | None = None
    motor_encoder_freeze_duration_s: float | None = None
    motor_encoder_freeze_period_s: float | None = None
    resistance_scale_start: float | None = None
    resistance_scale: float = 1.0
    resistance_ramp_s: float = 0.0
    resistance_voltage_drop_gain: float = 0.0
    resistance_voltage_collapse_gain: float = 0.0
    resistance_phase_imbalance_gain: float = 0.0
    resistance_extra_heat_gain: float = 0.0
    flux_scale_start: float | None = None
    flux_scale: float = 1.0
    flux_ramp_s: float = 0.0
    flux_current_boost_gain: float = 0.0
    flux_power_gap_gain: float = 0.0
    flux_tracking_drag_gain: float = 0.0
    bus_voltage_scale_start: float | None = None
    bus_voltage_scale: float = 1.0
    bus_voltage_ramp_s: float = 0.0
    inverter_voltage_scale_start: float | None = None
    inverter_voltage_scale: float = 1.0
    inverter_voltage_ramp_s: float = 0.0
    inverter_voltage_meas_drop_gain: float = 0.0
    inverter_voltage_ripple_gain: float = 0.0
    inverter_phase_asymmetry_gain: float = 0.0
    inverter_pwm_clamp_gain: float = 0.0
    backlash_scale_start: float | None = None
    backlash_scale: float = 1.0
    backlash_ramp_s: float = 0.0
    backlash_drag_gain: float = 0.0
    backlash_rattle_amp: float = 0.0
    backlash_rattle_freq: float = 0.0
    bearing_fault_start: float | None = None
    bearing_torque_amp: float = 0.0
    bearing_freq_hz: float = 0.0
    bearing_ramp_s: float = 0.0
    bearing_harmonic_ratio: float = 0.0
    bearing_sideband_ratio: float = 0.0
    bearing_impulse_gain: float = 0.0
    bearing_envelope_gain: float = 0.0
    thermal_derate_start: float | None = None
    thermal_limit_temp_c: float = 90.0
    thermal_current_limit_gain: float = 0.0
    thermal_current_limit_min_scale: float = 1.0
    thermal_flux_drop_gain: float = 0.0
    thermal_resistance_rise_gain: float = 0.0
    thermal_preheat_delta_c: float = 0.0
    cooling_fault_scale: float = 1.0
    thermal_copper_loss_gain: float = 0.0
    thermal_iron_loss_gain: float = 0.0


@dataclass(frozen=True)
class OperatingCondition:
    name: str
    condition_id: int
    airspeed: float
    bus_voltage: float
    load_bias_torque: float
    profile_name: str
    ambient_temp_c: float = 25.0
    winding_temp_init_c: float = 25.0
    housing_temp_init_c: float = 25.0
    airspeed_profile_name: str = "steady"
