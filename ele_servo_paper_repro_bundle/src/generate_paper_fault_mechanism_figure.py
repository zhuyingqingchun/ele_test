from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from servo_diagnostic.config import ServoPlantParams
from servo_diagnostic.operating_conditions import create_operating_conditions
from servo_diagnostic.scenarios import create_scenarios
from servo_diagnostic.simulator import simulate_scenario


NORMAL_COLOR = "#1f77b4"
FAULT_COLOR = "#d62728"
MARKER_COLOR = "#444444"


def find_condition(name: str):
    for condition in create_operating_conditions():
        if condition.name == name:
            return condition
    raise ValueError(f"Unknown condition: {name}")


def find_scenario(name: str):
    for scenario in create_scenarios():
        if scenario.name == name:
            return scenario
    raise ValueError(f"Unknown scenario: {name}")


def values(rows: list[dict[str, float | str | int]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=np.float64)


def position_sensing_deviation(rows: list[dict[str, float | str | int]]) -> np.ndarray:
    return values(rows, "theta_meas_deg") - values(rows, "theta_true_deg")


def robust_limits(normal: np.ndarray, fault: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([normal, fault])
    lo, hi = np.quantile(combined, [0.01, 0.99])
    if np.isclose(lo, hi):
        span = max(abs(lo), 1.0) * 0.15
        return lo - span, hi + span
    margin = (hi - lo) * 0.12
    return lo - margin, hi + margin


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a paper figure for typical servo fault mechanisms.")
    parser.add_argument("--condition", default="sine_tracking")
    parser.add_argument(
        "--output",
        default="/mnt/PRO6000_disk/swd/ele_servo_gpt5/论文/figures/figure2_typical_fault_mechanisms_sine_tracking.png",
    )
    args = parser.parse_args()

    params = ServoPlantParams()
    condition = find_condition(args.condition)
    normal_rows = simulate_scenario(params, find_scenario("normal"), condition)
    time_axis = values(normal_rows, "time_s")

    plot_specs = [
        {
            "scenario": "winding_resistance_rise",
            "fault_start": 1.0,
            "panels": [
                ("Winding Resistance Rise\nPhase Current Imbalance", "Imbalance (A)", lambda rows: values(rows, "phase_current_imbalance_est_a")),
                ("Winding Temperature Rate", "Temp Rate (C/s)", lambda rows: values(rows, "winding_temp_rate_c_s")),
            ],
        },
        {
            "scenario": "bearing_defect",
            "fault_start": 1.4,
            "panels": [
                ("Bearing Defect\nVibration Envelope", "Envelope (m/s^2)", lambda rows: values(rows, "vibration_envelope_mps2")),
                ("Friction Torque", "Torque (N m)", lambda rows: values(rows, "torque_friction_nm")),
            ],
        },
        {
            "scenario": "position_sensor_bias",
            "fault_start": 1.4,
            "panels": [
                ("Position Sensor Bias\nSensing Deviation", "Deviation (deg)", position_sensing_deviation),
                ("Current Residual", "Residual (A)", lambda rows: values(rows, "current_residual_a")),
            ],
        },
    ]

    fig, axes = plt.subplots(len(plot_specs), 2, figsize=(12.5, 9.2), sharex=True)

    for row_idx, spec in enumerate(plot_specs):
        fault_rows = simulate_scenario(params, find_scenario(spec["scenario"]), condition)
        for col_idx, (title, ylabel, extractor) in enumerate(spec["panels"]):
            ax = axes[row_idx, col_idx]
            normal_signal = extractor(normal_rows)
            fault_signal = extractor(fault_rows)

            ax.plot(time_axis, normal_signal, color=NORMAL_COLOR, linewidth=1.6, label="normal")
            ax.plot(time_axis, fault_signal, color=FAULT_COLOR, linewidth=1.4, label="fault")
            ax.axvline(spec["fault_start"], color=MARKER_COLOR, linestyle="--", linewidth=1.0)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(ylabel)
            ax.set_ylim(*robust_limits(normal_signal, fault_signal))
            ax.grid(True, linestyle="--", alpha=0.3)
            ax.legend(loc="upper right", fontsize=8, frameon=True)

            if col_idx == 0:
                ymax = ax.get_ylim()[1]
                ax.text(
                    spec["fault_start"] + 0.05,
                    ymax - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.08,
                    "fault start",
                    fontsize=8,
                    color=MARKER_COLOR,
                )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    for ax in axes.flat:
        ax.set_xlim(0.0, 4.0)

    fig.suptitle("Typical Fault Mechanism Responses Under Sine Tracking Condition", fontsize=14, y=0.995)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()
