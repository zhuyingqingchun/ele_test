from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DiagnosisResult:
    accuracy: float
    labels: list[str]
    confusion: list[list[int]]
    predictions: list[str]
    truths: list[str]


def load_feature_table(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    feature_columns = [
        column
        for column in rows[0].keys()
        if "__" in column
    ]
    return rows, feature_columns


def split_rows_by_condition(
    rows: list[dict[str, str]],
    holdout_condition: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows = [row for row in rows if row["condition_name"] != holdout_condition]
    test_rows = [row for row in rows if row["condition_name"] == holdout_condition]
    if not train_rows or not test_rows:
        raise ValueError(f"Invalid holdout condition: {holdout_condition}")
    return train_rows, test_rows


def rows_to_matrix(rows: list[dict[str, str]], feature_columns: list[str]) -> np.ndarray:
    return np.array([[float(row[column]) for column in feature_columns] for row in rows], dtype=float)


def rows_to_labels(rows: list[dict[str, str]]) -> list[str]:
    return [row["fault_label"] for row in rows]


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def transform_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def fit_nearest_centroid(X: np.ndarray, labels: list[str]) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for sample, label in zip(X, labels):
        grouped[label].append(sample)
    ordered_labels = sorted(grouped.keys())
    centroids = np.stack([np.mean(grouped[label], axis=0) for label in ordered_labels], axis=0)
    return ordered_labels, centroids


def predict_nearest_centroid(X: np.ndarray, labels: list[str], centroids: np.ndarray) -> list[str]:
    distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    indices = np.argmin(distances, axis=1)
    return [labels[index] for index in indices]


def evaluate_predictions(truths: list[str], predictions: list[str]) -> DiagnosisResult:
    ordered_labels = sorted(set(truths) | set(predictions))
    label_to_index = {label: index for index, label in enumerate(ordered_labels)}
    confusion = [[0 for _ in ordered_labels] for _ in ordered_labels]
    correct = 0
    for truth, pred in zip(truths, predictions):
        confusion[label_to_index[truth]][label_to_index[pred]] += 1
        if truth == pred:
            correct += 1
    accuracy = correct / max(len(truths), 1)
    return DiagnosisResult(
        accuracy=accuracy,
        labels=ordered_labels,
        confusion=confusion,
        predictions=predictions,
        truths=truths,
    )


def run_simple_diagnosis(feature_csv: Path, holdout_condition: str) -> DiagnosisResult:
    rows, feature_columns = load_feature_table(feature_csv)
    train_rows, test_rows = split_rows_by_condition(rows, holdout_condition)
    X_train = rows_to_matrix(train_rows, feature_columns)
    X_test = rows_to_matrix(test_rows, feature_columns)
    y_train = rows_to_labels(train_rows)
    y_test = rows_to_labels(test_rows)

    mean, std = fit_standardizer(X_train)
    X_train_std = transform_standardize(X_train, mean, std)
    X_test_std = transform_standardize(X_test, mean, std)
    centroid_labels, centroids = fit_nearest_centroid(X_train_std, y_train)
    predictions = predict_nearest_centroid(X_test_std, centroid_labels, centroids)
    return evaluate_predictions(y_test, predictions)


def save_diagnosis_result(path: Path, result: DiagnosisResult) -> None:
    payload = {
        "accuracy": result.accuracy,
        "labels": result.labels,
        "confusion": result.confusion,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
