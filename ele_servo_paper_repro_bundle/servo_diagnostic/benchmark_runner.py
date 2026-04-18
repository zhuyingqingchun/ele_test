from __future__ import annotations

from pathlib import Path

from .benchmark_utils import EvaluationResult, save_results_csv
from .cnn_baseline import evaluate_cnn1d_holdout
from .tree_baseline import evaluate_tree_holdout


def evaluate_cnn_over_conditions(
    dataset_path: Path,
    holdout_conditions: list[str],
    output_dir: Path,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> list[EvaluationResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[EvaluationResult] = []
    for holdout_condition in holdout_conditions:
        result = evaluate_cnn1d_holdout(
            dataset_path=dataset_path,
            holdout_condition=holdout_condition,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_json=output_dir / f"cnn1d__{holdout_condition}.json",
        )
        results.append(result)
    save_results_csv(output_dir / "cnn1d_leave_one_condition_summary.csv", results)
    return results


def evaluate_tree_over_conditions(
    feature_csv: Path,
    holdout_conditions: list[str],
    output_dir: Path,
    preferred_backend: str = "auto",
) -> list[EvaluationResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[EvaluationResult] = []
    for holdout_condition in holdout_conditions:
        result = evaluate_tree_holdout(
            feature_csv=feature_csv,
            holdout_condition=holdout_condition,
            preferred_backend=preferred_backend,
            output_json=output_dir / f"tree__{holdout_condition}.json",
        )
        results.append(result)
    save_results_csv(output_dir / "tree_leave_one_condition_summary.csv", results)
    return results
