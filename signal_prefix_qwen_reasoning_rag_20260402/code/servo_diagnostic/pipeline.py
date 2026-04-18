from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import ServoPlantParams
from .io_utils import save_dataset, write_summary
from .operating_conditions import create_operating_conditions
from .plots import (
    plot_architecture,
    plot_control_topology,
    plot_fault_vs_normal,
    plot_scenario_overview,
    plot_signal_topology,
    plot_single_scenario,
)
from .scenarios import create_scenarios
from .simulator import simulate_scenario
from .validation import build_validation_reports

def run_pipeline(
    base_dir: Path | None = None,
    params: ServoPlantParams | None = None,
    repeat_count: int = 1,
    seed_stride: int = 10000,
    save_per_run: bool = True,
) -> dict[str, Path | int]:
    params = params or ServoPlantParams()
    base_dir = base_dir or Path(__file__).resolve().parent.parent
    dataset_dir = base_dir / "diagnostic_datasets"
    comparison_dir = base_dir / "comparison_plots"
    dataset_dir.mkdir(exist_ok=True)
    comparison_dir.mkdir(exist_ok=True)
    if repeat_count < 1:
        raise ValueError("repeat_count must be >= 1.")

    scenarios = create_scenarios()
    conditions = create_operating_conditions()
    all_rows: list[dict[str, float | str | int]] = []
    rows_by_run: dict[str, list[dict[str, float | str | int]]] = {}
    rows_for_overview: dict[str, list[dict[str, float | str | int]]] = {}

    for replicate_id in range(repeat_count):
        run_params = replace(params, random_seed=params.random_seed + replicate_id * seed_stride)
        for condition in conditions:
            for scenario in scenarios:
                rows = simulate_scenario(run_params, scenario, condition)
                run_name = f"{condition.name}__{scenario.name}"
                run_key = run_name if repeat_count == 1 else f"{run_name}__rep{replicate_id:02d}"
                for row in rows:
                    row["replicate_id"] = replicate_id
                    row["run_seed"] = run_params.random_seed
                    row["run_name"] = run_key
                rows_by_run[run_key] = rows
                all_rows.extend(rows)
                if save_per_run:
                    save_dataset(dataset_dir / f"{run_key}.csv", rows)
                if condition.name == "nominal_multistep" and scenario.name not in rows_for_overview:
                    rows_for_overview[scenario.name] = rows

    for condition in conditions:
        normal_key = f"{condition.name}__normal" if repeat_count == 1 else f"{condition.name}__normal__rep00"
        normal_rows = rows_by_run[normal_key]
        for scenario in scenarios:
            if scenario.name == "normal":
                continue
            fault_key = f"{condition.name}__{scenario.name}" if repeat_count == 1 else f"{condition.name}__{scenario.name}__rep00"
            fault_rows = rows_by_run[fault_key]
            plot_fault_vs_normal(
                normal_rows,
                fault_rows,
                comparison_dir / f"{condition.name}__{scenario.name}_vs_normal.png",
            )

    combined_dataset = dataset_dir / "servo_fault_diagnosis_dataset_full.csv"
    summary_path = dataset_dir / "dataset_summary.csv"
    acceptance_path = dataset_dir / "performance_acceptance_report.csv"
    frequency_path = dataset_dir / "frequency_response_report.csv"
    architecture_path = base_dir / "servo_diagnostic_architecture.png"
    control_topology_path = base_dir / "servo_control_topology.png"
    signal_topology_path = base_dir / "servo_signal_topology.png"
    overview_path = base_dir / "servo_diagnostic_overview.png"
    jam_plot_path = base_dir / "servo_jam_signals.png"

    save_dataset(combined_dataset, all_rows)
    write_summary(summary_path, rows_by_run)
    acceptance_rows, frequency_rows = build_validation_reports(rows_by_run, params, conditions)
    save_dataset(acceptance_path, acceptance_rows)
    save_dataset(frequency_path, frequency_rows)
    plot_architecture(architecture_path)
    plot_control_topology(control_topology_path)
    plot_signal_topology(signal_topology_path)
    plot_scenario_overview(rows_for_overview, overview_path)
    jam_key = "nominal_multistep__jam_fault" if repeat_count == 1 else "nominal_multistep__jam_fault__rep00"
    plot_single_scenario(rows_by_run[jam_key], jam_plot_path)

    return {
        "combined_dataset": combined_dataset,
        "summary": summary_path,
        "acceptance_report": acceptance_path,
        "frequency_report": frequency_path,
        "architecture_plot": architecture_path,
        "control_topology_plot": control_topology_path,
        "signal_topology_plot": signal_topology_path,
        "overview_plot": overview_path,
        "jam_plot": jam_plot_path,
        "comparison_dir": comparison_dir,
        "repeat_count": repeat_count,
    }
