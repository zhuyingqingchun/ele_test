from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle



def draw_block(ax: plt.Axes, x: float, y: float, width: float, height: float, title: str, body: str) -> None:
    rect = Rectangle((x, y), width, height, linewidth=1.4, edgecolor="black", facecolor="#f6efe1")
    ax.add_patch(rect)
    ax.text(x + width / 2.0, y + height * 0.67, title, ha="center", va="center", fontsize=11, weight="bold")
    ax.text(x + width / 2.0, y + height * 0.33, body, ha="center", va="center", fontsize=9.5)



def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], label: str = "") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", linewidth=1.4, color="black"),
    )
    if label:
        ax.text((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5 + 0.18, label, ha="center", fontsize=9.5)



def plot_control_topology(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 5.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    for x, y in ((1.0, 3.7), (4.4, 2.6), (7.8, 1.5)):
        circle = Circle((x, y), 0.18, fill=False, linewidth=1.4)
        ax.add_patch(circle)
        ax.text(x - 0.03, y + 0.10, "+", fontsize=11)
        ax.text(x - 0.03, y - 0.16, "-", fontsize=11)

    draw_block(ax, 1.5, 2.95, 2.0, 1.45, "Position PD", "theta_ref -> omega_ref")
    draw_block(ax, 4.9, 1.9, 2.0, 1.45, "Speed PI", "omega_ref -> i_ref")
    draw_block(ax, 8.3, 0.85, 2.0, 1.45, "Current PI", "i_ref -> u_cmd")
    draw_block(ax, 10.9, 0.85, 1.8, 1.45, "PWM Driver", "u_cmd -> u_a")
    draw_block(ax, 13.0, 0.6, 1.7, 1.95, "Servo Plant", "R-L motor\ngear + load")

    draw_arrow(ax, (0.1, 3.7), (0.82, 3.7), "theta_ref")
    draw_arrow(ax, (1.18, 3.7), (1.5, 3.7), "e_theta")
    draw_arrow(ax, (3.5, 3.7), (4.22, 2.6), "omega_ref")
    draw_arrow(ax, (4.58, 2.6), (4.9, 2.6), "e_omega")
    draw_arrow(ax, (6.9, 2.6), (7.62, 1.5), "i_ref")
    draw_arrow(ax, (7.98, 1.5), (8.3, 1.5), "e_i")
    draw_arrow(ax, (10.3, 1.5), (10.9, 1.5), "u_cmd")
    draw_arrow(ax, (12.7, 1.5), (13.0, 1.5), "u_a")
    draw_arrow(ax, (14.7, 1.5), (15.0, 1.5), "theta, omega, i")

    ax.plot([14.4, 14.4], [1.5, 4.65], color="black", linewidth=1.2)
    ax.plot([14.4, 1.0], [4.65, 4.65], color="black", linewidth=1.2)
    draw_arrow(ax, (1.0, 4.65), (1.0, 3.88), "theta_meas")

    ax.plot([13.95, 13.95], [1.5, 2.6], color="black", linewidth=1.2)
    ax.plot([13.95, 4.4], [2.6, 2.6], color="black", linewidth=1.2)
    draw_arrow(ax, (4.4, 2.6), (4.4, 2.42), "omega_m_meas")

    ax.plot([13.45, 13.45], [1.5, 0.22], color="black", linewidth=1.2)
    ax.plot([13.45, 7.8], [0.22, 0.22], color="black", linewidth=1.2)
    draw_arrow(ax, (7.8, 0.22), (7.8, 1.32), "i_meas")

    ax.text(7.5, 5.12, "Current Control Topology", ha="center", fontsize=16, weight="bold")
    ax.text(
        7.5,
        4.72,
        "Outer position loop, middle speed loop, inner current loop, then PWM and physical servo plant",
        ha="center",
        fontsize=10.5,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def plot_signal_topology(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    draw_block(ax, 0.6, 5.6, 2.0, 1.0, "Command Profile", "theta_ref")
    draw_block(ax, 3.0, 5.55, 2.2, 1.1, "Controller Stack", "omega_ref, i_ref, u_cmd")
    draw_block(ax, 5.8, 5.55, 2.0, 1.1, "PWM + Bus", "u_a, duty, Vbus")
    draw_block(ax, 8.3, 5.35, 2.3, 1.5, "Electrical Subsystem", "i_a, phase currents,\nback-EMF, power")
    draw_block(ax, 11.2, 5.35, 2.2, 1.5, "Mechanical Subsystem", "omega_m, theta_m,\nshaft twist, torque")
    draw_block(ax, 13.8, 5.35, 1.8, 1.5, "Load/Aero", "theta_l, omega_l,\nload torque")

    draw_block(ax, 8.6, 3.15, 2.2, 1.0, "Thermal Branch", "winding temp,\nhousing temp")
    draw_block(ax, 11.5, 3.15, 2.2, 1.0, "Vibration Branch", "accel proxy,\nshock level")
    draw_block(ax, 1.0, 1.0, 2.8, 1.5, "Measured Control Signals", "theta_meas\nomega_m_meas\ni_meas\nu_meas")
    draw_block(ax, 4.4, 0.85, 3.0, 1.8, "Electrical Diagnostic Signals", "phase currents\nback_emf_v\npwm_duty\nelectrical_power_w\nsupply_current_est_a")
    draw_block(ax, 8.1, 0.85, 3.0, 1.8, "Mechanical Diagnostic Signals", "torque_em_nm\ntorque_shaft_nm\ntorque_load_nm\nshaft_twist_deg\nmotor_accel_deg_s2")
    draw_block(ax, 11.8, 0.85, 3.0, 1.8, "Health Indicators", "winding_temp_c\nhousing_temp_c\nvibration_accel_mps2\nresiduals")

    draw_arrow(ax, (2.6, 6.1), (3.0, 6.1), "theta_ref")
    draw_arrow(ax, (5.2, 6.1), (5.8, 6.1), "u_cmd")
    draw_arrow(ax, (7.8, 6.1), (8.3, 6.1), "u_a")
    draw_arrow(ax, (10.6, 6.1), (11.2, 6.1), "torque_em")
    draw_arrow(ax, (13.4, 6.1), (13.8, 6.1), "theta_l")

    draw_arrow(ax, (9.4, 5.35), (9.6, 4.15), "losses")
    draw_arrow(ax, (12.3, 5.35), (12.5, 4.15), "torque ripple")

    draw_arrow(ax, (8.9, 5.35), (5.9, 2.65), "phase I, V, power")
    draw_arrow(ax, (12.0, 5.35), (9.2, 2.65), "speed, twist, torque")
    draw_arrow(ax, (14.3, 5.35), (2.4, 2.5), "theta, omega")
    draw_arrow(ax, (9.7, 3.15), (13.2, 2.65), "temperatures")
    draw_arrow(ax, (12.6, 3.15), (13.2, 2.65), "vibration")
    draw_arrow(ax, (3.8, 1.75), (4.4, 1.75), "measured channels")
    draw_arrow(ax, (7.4, 1.75), (8.1, 1.75), "torque channels")
    draw_arrow(ax, (11.1, 1.75), (11.8, 1.75), "state indicators")

    ax.text(8.0, 6.95, "Current Signal Topology", ha="center", fontsize=16, weight="bold")
    ax.text(
        8.0,
        6.6,
        "Signal flow from reference and controller outputs to electrical, mechanical, thermal, vibration, and diagnostic channels",
        ha="center",
        fontsize=10.5,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def plot_architecture(output_path: Path) -> None:
    plot_control_topology(output_path)



def plot_scenario_overview(rows_by_scenario: dict[str, list[dict[str, float | str | int]]], output_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    for scenario_name, rows in rows_by_scenario.items():
        time_axis = np.array([float(row["time_s"]) for row in rows])
        theta = np.array([float(row["theta_meas_deg"]) for row in rows])
        omega = np.array([float(row["omega_motor_meas_deg_s"]) for row in rows])
        current = np.array([float(row["current_meas_a"]) for row in rows])
        torque = np.array([float(row["torque_load_nm"]) for row in rows])

        axes[0].plot(time_axis, theta, linewidth=1.6, label=scenario_name)
        axes[1].plot(time_axis, omega, linewidth=1.4, label=scenario_name)
        axes[2].plot(time_axis, current, linewidth=1.4, label=scenario_name)
        axes[3].plot(time_axis, torque, linewidth=1.4, label=scenario_name)

    axes[0].set_ylabel("theta_meas (deg)")
    axes[1].set_ylabel("omega_m_meas (deg/s)")
    axes[2].set_ylabel("current_meas (A)")
    axes[3].set_ylabel("load torque (N m)")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)

    axes[0].legend(loc="upper right", ncol=2)
    fig.suptitle("Diagnostic signal overview across fault scenarios", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def plot_single_scenario(rows: list[dict[str, float | str | int]], output_path: Path) -> None:
    time_axis = np.array([float(row["time_s"]) for row in rows])
    theta_ref = np.array([float(row["theta_ref_deg"]) for row in rows])
    theta_meas = np.array([float(row["theta_meas_deg"]) for row in rows])
    omega = np.array([float(row["omega_motor_meas_deg_s"]) for row in rows])
    current = np.array([float(row["current_meas_a"]) for row in rows])
    voltage = np.array([float(row["voltage_meas_v"]) for row in rows])

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time_axis, theta_ref, label="theta_ref", linewidth=1.5)
    axes[0].plot(time_axis, theta_meas, label="theta_meas", linewidth=1.5)
    axes[1].plot(time_axis, omega, label="omega_m_meas", color="#d62728", linewidth=1.4)
    axes[2].plot(time_axis, current, label="current_meas", color="#2ca02c", linewidth=1.4)
    axes[2].plot(time_axis, voltage, label="voltage_meas", color="#9467bd", linewidth=1.1, alpha=0.7)

    axes[0].set_ylabel("Angle (deg)")
    axes[1].set_ylabel("Speed (deg/s)")
    axes[2].set_ylabel("Current / Voltage")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right")

    fig.suptitle(f"Representative diagnostic signals: {rows[0]['scenario']}", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def plot_fault_vs_normal(
    normal_rows: list[dict[str, float | str | int]],
    fault_rows: list[dict[str, float | str | int]],
    output_path: Path,
) -> None:
    time_axis = np.array([float(row["time_s"]) for row in normal_rows])
    theta_ref = np.array([float(row["theta_ref_deg"]) for row in normal_rows])

    normal_theta = np.array([float(row["theta_meas_deg"]) for row in normal_rows])
    fault_theta = np.array([float(row["theta_meas_deg"]) for row in fault_rows])
    normal_omega = np.array([float(row["omega_motor_meas_deg_s"]) for row in normal_rows])
    fault_omega = np.array([float(row["omega_motor_meas_deg_s"]) for row in fault_rows])
    normal_current = np.array([float(row["current_meas_a"]) for row in normal_rows])
    fault_current = np.array([float(row["current_meas_a"]) for row in fault_rows])
    normal_torque = np.array([float(row["torque_load_nm"]) for row in normal_rows])
    fault_torque = np.array([float(row["torque_load_nm"]) for row in fault_rows])

    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(time_axis, theta_ref, color="black", linestyle="--", linewidth=1.0, label="theta_ref")
    axes[0].plot(time_axis, normal_theta, color="#1f77b4", linewidth=1.6, label="normal")
    axes[0].plot(time_axis, fault_theta, color="#d62728", linewidth=1.4, label="fault")
    axes[1].plot(time_axis, normal_omega, color="#1f77b4", linewidth=1.4, label="normal")
    axes[1].plot(time_axis, fault_omega, color="#d62728", linewidth=1.4, label="fault")
    axes[2].plot(time_axis, normal_current, color="#1f77b4", linewidth=1.4, label="normal")
    axes[2].plot(time_axis, fault_current, color="#d62728", linewidth=1.4, label="fault")
    axes[3].plot(time_axis, normal_torque, color="#1f77b4", linewidth=1.4, label="normal")
    axes[3].plot(time_axis, fault_torque, color="#d62728", linewidth=1.4, label="fault")

    axes[0].set_ylabel("theta (deg)")
    axes[1].set_ylabel("omega_m (deg/s)")
    axes[2].set_ylabel("current (A)")
    axes[3].set_ylabel("load torque (N m)")
    axes[3].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(loc="upper right")

    condition_name = str(fault_rows[0]["condition_name"])
    scenario_name = str(fault_rows[0]["scenario"])
    fault_label = str(fault_rows[0]["fault_label"])
    profile_name = str(fault_rows[0]["profile_name"])
    fig.suptitle(
        f"Normal vs Fault: {condition_name} | {scenario_name} ({fault_label}) | {profile_name}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
