from __future__ import annotations

from .config import FaultScenario, OperatingCondition, ServoPlantParams
from .controllers import CascadedController, ReferenceFunction
from .plant import DiagnosticServoPlant, wrap_to_pi


def simulate_scenario(
    params: ServoPlantParams,
    scenario: FaultScenario,
    condition: OperatingCondition,
    reference_fn: ReferenceFunction | None = None,
) -> list[dict[str, float | str | int]]:
    import numpy as np

    seed = params.random_seed + scenario.fault_id * 1000 + condition.condition_id * 100
    rng = np.random.default_rng(seed)
    plant = DiagnosticServoPlant(params, condition)
    controller = CascadedController(params, condition, reference_fn=reference_fn)
    time_axis = np.arange(0.0, params.final_time + params.dt, params.dt)
    rows: list[dict[str, float | str | int]] = []

    for sample_index, t in enumerate(time_axis):
        available_bus_voltage = plant.get_available_bus_voltage(scenario, float(t))
        available_current_limit = plant.get_available_current_limit(scenario, float(t))
        measurement = plant.measure(scenario, float(t), rng)
        control = controller.compute(
            t=float(t),
            theta_meas=measurement.theta_meas,
            omega_load_est=measurement.omega_l_est,
            omega_motor_meas=measurement.omega_m_meas,
            i_d_meas=measurement.i_d_meas,
            i_q_meas=measurement.i_q_meas,
            available_bus_voltage=available_bus_voltage,
            available_current_limit=available_current_limit,
        )
        signals = plant.step(scenario, float(t), control.v_d_cmd, control.v_q_cmd, rng)

        theta_e_true = wrap_to_pi(params.pole_pairs * plant.state.theta_m)

        rows.append(
            {
                "scenario": scenario.name,
                "fault_label": scenario.fault_label,
                "fault_id": scenario.fault_id,
                "condition_name": condition.name,
                "condition_id": condition.condition_id,
                "profile_name": condition.profile_name,
                "airspeed_mps": float(condition.airspeed),
                "airspeed_actual_mps": float(signals.airspeed_mps),
                "ambient_temp_c": float(condition.ambient_temp_c),
                "bus_voltage_v": float(condition.bus_voltage),
                "sample_index": sample_index,
                "time_s": float(t),
                "theta_ref_deg": float(np.rad2deg(control.theta_ref)),
                "theta_true_deg": float(np.rad2deg(plant.state.theta_l)),
                "theta_meas_deg": float(np.rad2deg(measurement.theta_meas)),
                "encoder_count": int(measurement.encoder_count),
                "motor_encoder_count": int(measurement.motor_encoder_count),
                "theta_motor_meas_deg": float(np.rad2deg(measurement.theta_motor_meas)),
                "theta_e_true_deg": float(np.rad2deg(theta_e_true)),
                "theta_e_obs_deg": float(np.rad2deg(measurement.theta_e_obs)),
                "omega_motor_true_deg_s": float(np.rad2deg(plant.state.omega_m)),
                "omega_motor_meas_deg_s": float(np.rad2deg(measurement.omega_m_meas)),
                "omega_load_true_deg_s": float(np.rad2deg(plant.state.omega_l)),
                "omega_load_est_deg_s": float(np.rad2deg(measurement.omega_l_est)),
                "load_accel_est_deg_s2": float(np.rad2deg(measurement.load_accel_est)),
                "motor_speed_ref_deg_s": float(np.rad2deg(control.omega_motor_ref)),
                "current_true_a": float(plant.state.i_q),
                "current_meas_a": float(measurement.i_q_meas),
                "current_ref_a": float(control.i_q_ref),
                "current_limit_a": float(available_current_limit),
                "current_d_true_a": float(plant.state.i_d),
                "current_q_true_a": float(plant.state.i_q),
                "current_d_meas_a": float(measurement.i_d_meas),
                "current_q_meas_a": float(measurement.i_q_meas),
                "current_residual_a": float(control.i_q_ref - measurement.i_q_meas),
                "speed_residual_deg_s": float(np.rad2deg(control.omega_motor_ref - measurement.omega_m_meas)),
                "voltage_cmd_v": float(control.u_cmd),
                "vd_cmd_v": float(control.v_d_cmd),
                "vq_cmd_v": float(control.v_q_cmd),
                "voltage_meas_v": float(measurement.u_meas),
                "phase_voltage_a_meas_v": float(measurement.phase_voltage_a_meas),
                "phase_voltage_b_meas_v": float(measurement.phase_voltage_b_meas),
                "phase_voltage_c_meas_v": float(measurement.phase_voltage_c_meas),
                "available_bus_voltage_v": float(signals.available_bus_voltage_v),
                "pwm_duty": float(signals.pwm_duty),
                "back_emf_v": float(signals.back_emf_v),
                "electrical_power_w": float(signals.electrical_power_w),
                "mechanical_power_w": float(signals.mechanical_power_w),
                "copper_loss_w": float(signals.copper_loss_w),
                "iron_loss_w": float(signals.iron_loss_w),
                "supply_current_est_a": float(signals.supply_current_est_a),
                "bus_current_true_a": float(signals.bus_current_true_a),
                "bus_current_meas_a": float(measurement.bus_current_meas),
                "phase_current_a_true_a": float(signals.phase_current_a),
                "phase_current_b_true_a": float(signals.phase_current_b),
                "phase_current_c_true_a": float(signals.phase_current_c),
                "phase_current_a_est_a": float(measurement.phase_current_a_meas),
                "phase_current_b_est_a": float(measurement.phase_current_b_meas),
                "phase_current_c_est_a": float(measurement.phase_current_c_meas),
                "phase_current_neutral_true_a": float(signals.phase_current_neutral_a),
                "phase_current_imbalance_true_a": float(signals.phase_current_imbalance_a),
                "phase_current_neutral_est_a": float(measurement.phase_current_a_meas + measurement.phase_current_b_meas + measurement.phase_current_c_meas),
                "phase_current_imbalance_est_a": float(np.sqrt(((measurement.phase_current_a_meas - measurement.phase_current_b_meas) ** 2 + (measurement.phase_current_b_meas - measurement.phase_current_c_meas) ** 2 + (measurement.phase_current_c_meas - measurement.phase_current_a_meas) ** 2) / 3.0)),
                "phase_voltage_a_v": float(signals.phase_voltage_a),
                "phase_voltage_b_v": float(signals.phase_voltage_b),
                "phase_voltage_c_v": float(signals.phase_voltage_c),
                "torque_em_nm": float(signals.torque_em),
                "torque_shaft_nm": float(signals.torque_shaft),
                "torque_load_nm": float(signals.torque_load),
                "torque_friction_nm": float(signals.torque_friction),
                "position_error_deg": float(np.rad2deg(control.position_error)),
                "speed_ref_deg_s": float(np.rad2deg(control.omega_load_ref)),
                "motor_accel_deg_s2": float(np.rad2deg(signals.motor_accel_rad_s2)),
                "load_accel_deg_s2": float(np.rad2deg(signals.load_accel_rad_s2)),
                "shaft_twist_deg": float(np.rad2deg(signals.shaft_twist_rad)),
                "winding_temp_c": float(signals.winding_temp_c),
                "housing_temp_c": float(signals.housing_temp_c),
                "winding_temp_rate_c_s": float(signals.winding_temp_rate_c_s),
                "housing_temp_rate_c_s": float(signals.housing_temp_rate_c_s),
                "vibration_accel_mps2": float(signals.vibration_accel_mps2),
                "vibration_band_mps2": float(signals.vibration_band_mps2),
                "vibration_envelope_mps2": float(signals.vibration_envelope_mps2),
                "vibration_shock_index": float(signals.vibration_shock_index),
                "fault_load_active": int(signals.load_fault_active),
                "fault_friction_active": int(signals.friction_fault_active),
                "fault_jam_active": int(signals.jam_fault_active),
                "fault_sensor_active": int(measurement.sensor_active),
                "fault_current_sensor_active": int(measurement.current_sensor_fault_active),
                "fault_speed_sensor_active": int(measurement.speed_sensor_fault_active),
                "fault_position_sensor_active": int(measurement.position_sensor_fault_active),
                "fault_motor_encoder_active": int(measurement.motor_encoder_fault_active),
                "fault_electrical_active": int(signals.electrical_fault_active),
                "fault_mechanical_active": int(signals.mechanical_fault_active),
                "fault_demagnetization_active": int(signals.demagnetization_fault_active),
                "fault_inverter_active": int(signals.inverter_fault_active),
                "fault_bearing_active": int(signals.bearing_fault_active),
            }
        )

    return rows
