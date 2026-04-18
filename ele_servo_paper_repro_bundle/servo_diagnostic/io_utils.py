from __future__ import annotations

import csv
from pathlib import Path

import numpy as np



def save_dataset(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def summarize_rows(rows: list[dict[str, float | str | int]]) -> dict[str, float]:
    theta_error = np.array([float(row["position_error_deg"]) for row in rows])
    current = np.array([float(row["current_meas_a"]) for row in rows])
    speed = np.array([float(row["omega_motor_meas_deg_s"]) for row in rows])
    return {
        "max_abs_position_error_deg": float(np.max(np.abs(theta_error))),
        "rms_current_a": float(np.sqrt(np.mean(current**2))),
        "max_abs_speed_deg_s": float(np.max(np.abs(speed))),
    }



def write_summary(path: Path, rows_by_run: dict[str, list[dict[str, float | str | int]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "condition_name",
                "condition_id",
                "replicate_id",
                "run_seed",
                "profile_name",
                "scenario",
                "fault_label",
                "fault_id",
                "max_abs_position_error_deg",
                "rms_current_a",
                "max_abs_speed_deg_s",
            ]
        )
        for run_name, rows in rows_by_run.items():
            del run_name
            stats = summarize_rows(rows)
            writer.writerow(
                [
                    rows[0]["condition_name"],
                    rows[0]["condition_id"],
                    rows[0].get("replicate_id", 0),
                    rows[0].get("run_seed", ""),
                    rows[0]["profile_name"],
                    rows[0]["scenario"],
                    rows[0]["fault_label"],
                    rows[0]["fault_id"],
                    f"{stats['max_abs_position_error_deg']:.5f}",
                    f"{stats['rms_current_a']:.5f}",
                    f"{stats['max_abs_speed_deg_s']:.5f}",
                ]
            )
