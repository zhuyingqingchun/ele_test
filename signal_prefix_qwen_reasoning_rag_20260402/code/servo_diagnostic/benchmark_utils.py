from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix


@dataclass
class EvaluationResult:
    model_name: str
    backend: str
    holdout_condition: str
    train_samples: int
    test_samples: int
    accuracy: float
    labels: list[str]
    confusion: list[list[int]]
    extra: dict[str, float | int | str]


def load_window_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def split_window_dataset(data: dict[str, np.ndarray], holdout_condition: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    condition = data["condition"].astype(str)
    train_mask = condition != holdout_condition
    test_mask = condition == holdout_condition
    if not np.any(train_mask) or not np.any(test_mask):
        raise ValueError(f"Invalid holdout condition: {holdout_condition}")
    train = {key: value[train_mask] if value.shape[0] == condition.shape[0] else value for key, value in data.items()}
    test = {key: value[test_mask] if value.shape[0] == condition.shape[0] else value for key, value in data.items()}
    return train, test


def standardize_window_channels(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std, mean.squeeze(), std.squeeze()


def load_feature_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    feature_columns = [column for column in rows[0].keys() if "__" in column]
    return rows, feature_columns


def split_feature_rows(rows: list[dict[str, str]], holdout_condition: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows = [row for row in rows if row["condition_name"] != holdout_condition]
    test_rows = [row for row in rows if row["condition_name"] == holdout_condition]
    if not train_rows or not test_rows:
        raise ValueError(f"Invalid holdout condition: {holdout_condition}")
    return train_rows, test_rows


def feature_rows_to_matrix(rows: list[dict[str, str]], feature_columns: list[str]) -> np.ndarray:
    return np.array([[float(row[column]) for column in feature_columns] for row in rows], dtype=float)


def rows_to_fault_labels(rows: list[dict[str, str]]) -> list[str]:
    return [row["fault_label"] for row in rows]


def rows_to_fault_ids(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([int(row["fault_id"]) for row in rows], dtype=np.int64)


def build_result(
    model_name: str,
    backend: str,
    holdout_condition: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    train_samples: int,
    test_samples: int,
    extra: dict[str, float | int | str] | None = None,
) -> EvaluationResult:
    accuracy = float(np.mean(y_true == y_pred))
    confusion = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names))).tolist()
    return EvaluationResult(
        model_name=model_name,
        backend=backend,
        holdout_condition=holdout_condition,
        train_samples=train_samples,
        test_samples=test_samples,
        accuracy=accuracy,
        labels=label_names,
        confusion=confusion,
        extra=extra or {},
    )


def save_result_json(path: Path, result: EvaluationResult) -> None:
    path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")


def save_results_csv(path: Path, results: list[EvaluationResult]) -> None:
    rows = []
    for result in results:
        row = {
            "model_name": result.model_name,
            "backend": result.backend,
            "holdout_condition": result.holdout_condition,
            "train_samples": result.train_samples,
            "test_samples": result.test_samples,
            "accuracy": result.accuracy,
        }
        row.update(result.extra)
        rows.append(row)
    if not rows:
        raise ValueError("No evaluation rows to save.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
