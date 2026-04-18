from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


@dataclass
class HierarchicalEvaluationResult:
    model_name: str
    backend: str
    holdout_condition: str
    train_samples: int
    test_samples: int
    binary_accuracy: float
    family_accuracy: float
    scenario_accuracy: float
    binary_labels: list[str]
    binary_confusion: list[list[int]]
    family_labels: list[str]
    family_confusion: list[list[int]]
    scenario_labels: list[str]
    scenario_confusion: list[list[int]]


def load_feature_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open('r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    feature_columns = [column for column in rows[0].keys() if '__' in column]
    return rows, feature_columns


def split_feature_rows(rows: list[dict[str, str]], holdout_condition: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_rows = [row for row in rows if row['condition_name'] != holdout_condition]
    test_rows = [row for row in rows if row['condition_name'] == holdout_condition]
    if not train_rows or not test_rows:
        raise ValueError(f'Invalid holdout condition: {holdout_condition}')
    return train_rows, test_rows


def rows_to_matrix(rows: list[dict[str, str]], feature_columns: list[str]) -> np.ndarray:
    return np.array([[float(row[column]) for column in feature_columns] for row in rows], dtype=float)


def fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def transform_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def make_stage_model(backend: str):
    if backend == 'mlp':
        return MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=300, random_state=7)
    if backend != 'rf':
        raise ValueError(f'Unsupported backend: {backend}')
    return RandomForestClassifier(
        n_estimators=180,
        max_depth=20,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=7,
        n_jobs=1,
    )


def _encode_labels(labels: list[str]) -> tuple[np.ndarray, list[str], dict[str, int]]:
    ordered = sorted(set(labels))
    mapping = {label: idx for idx, label in enumerate(ordered)}
    encoded = np.array([mapping[label] for label in labels], dtype=np.int64)
    return encoded, ordered, mapping


def _confusion(y_true: list[str], y_pred: list[str], labels: list[str]) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    true_idx = np.array([index[label] for label in y_true], dtype=np.int64)
    pred_idx = np.array([index[label] for label in y_pred], dtype=np.int64)
    return confusion_matrix(true_idx, pred_idx, labels=np.arange(len(labels))).tolist()


class HierarchicalFaultDiagnoser:
    def __init__(self, backend: str = 'rf') -> None:
        self.backend = backend
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.binary_model = make_stage_model(backend)
        self.family_model = make_stage_model(backend)
        self.family_stage_models: dict[str, object] = {}
        self.family_default_scenarios: dict[str, str] = {}

    def fit(self, train_rows: list[dict[str, str]], feature_columns: list[str]) -> None:
        X_train = rows_to_matrix(train_rows, feature_columns)
        self.mean, self.std = fit_standardizer(X_train)
        X_train_std = transform_standardize(X_train, self.mean, self.std)

        binary_labels = ['fault' if row['fault_label'] != 'normal' else 'normal' for row in train_rows]
        y_binary, _, _ = _encode_labels(binary_labels)
        self.binary_model.fit(X_train_std, y_binary)
        self.binary_label_order = ['fault', 'normal'] if set(binary_labels) == {'fault', 'normal'} else sorted(set(binary_labels))

        fault_rows = [row for row in train_rows if row['fault_label'] != 'normal']
        X_fault = rows_to_matrix(fault_rows, feature_columns)
        X_fault_std = transform_standardize(X_fault, self.mean, self.std)
        family_labels = [row['fault_label'] for row in fault_rows]
        y_family, family_order, _ = _encode_labels(family_labels)
        self.family_model.fit(X_fault_std, y_family)
        self.family_label_order = family_order

        scenario_by_family: dict[str, list[dict[str, str]]] = {}
        for row in fault_rows:
            scenario_by_family.setdefault(row['fault_label'], []).append(row)

        for family, rows in scenario_by_family.items():
            scenarios = sorted({row['scenario'] for row in rows})
            if len(scenarios) == 1:
                self.family_default_scenarios[family] = scenarios[0]
                continue
            model = make_stage_model(self.backend)
            X_family = rows_to_matrix(rows, feature_columns)
            X_family_std = transform_standardize(X_family, self.mean, self.std)
            y_scenario, scenario_order, _ = _encode_labels([row['scenario'] for row in rows])
            model.fit(X_family_std, y_scenario)
            self.family_stage_models[family] = (model, scenario_order)

    def predict(self, rows: list[dict[str, str]], feature_columns: list[str]) -> tuple[list[str], list[str], list[str]]:
        if self.mean is None or self.std is None:
            raise RuntimeError('Model not fitted.')
        X = rows_to_matrix(rows, feature_columns)
        X_std = transform_standardize(X, self.mean, self.std)

        binary_raw = self.binary_model.predict(X_std)
        binary_predictions = ['normal' if int(idx) == 1 and self.binary_label_order == ['fault', 'normal'] else 'fault' for idx in binary_raw]
        if self.binary_label_order != ['fault', 'normal']:
            binary_predictions = [self.binary_label_order[int(idx)] for idx in binary_raw]

        family_predictions: list[str] = []
        scenario_predictions: list[str] = []
        for sample, binary_pred in zip(X_std, binary_predictions):
            if binary_pred == 'normal':
                family_predictions.append('normal')
                scenario_predictions.append('normal')
                continue
            family_idx = int(self.family_model.predict(sample.reshape(1, -1))[0])
            family_pred = self.family_label_order[family_idx]
            family_predictions.append(family_pred)
            if family_pred in self.family_default_scenarios:
                scenario_predictions.append(self.family_default_scenarios[family_pred])
                continue
            stage_model, scenario_order = self.family_stage_models[family_pred]
            scenario_idx = int(stage_model.predict(sample.reshape(1, -1))[0])
            scenario_predictions.append(scenario_order[scenario_idx])
        return binary_predictions, family_predictions, scenario_predictions


def evaluate_hierarchical_holdout(
    feature_csv: Path,
    holdout_condition: str,
    backend: str = 'rf',
    output_json: Path | None = None,
) -> HierarchicalEvaluationResult:
    rows, feature_columns = load_feature_rows(feature_csv)
    train_rows, test_rows = split_feature_rows(rows, holdout_condition)

    diagnoser = HierarchicalFaultDiagnoser(backend=backend)
    diagnoser.fit(train_rows, feature_columns)
    binary_pred, family_pred, scenario_pred = diagnoser.predict(test_rows, feature_columns)

    binary_true = ['fault' if row['fault_label'] != 'normal' else 'normal' for row in test_rows]
    family_true = [row['fault_label'] for row in test_rows]
    scenario_true = [row['scenario'] for row in test_rows]

    binary_labels = sorted(set(binary_true) | set(binary_pred))
    family_labels = sorted(set(family_true) | set(family_pred))
    scenario_labels = sorted(set(scenario_true) | set(scenario_pred))

    result = HierarchicalEvaluationResult(
        model_name='hierarchical_fault_diagnoser',
        backend=backend,
        holdout_condition=holdout_condition,
        train_samples=len(train_rows),
        test_samples=len(test_rows),
        binary_accuracy=float(np.mean(np.array(binary_true) == np.array(binary_pred))),
        family_accuracy=float(np.mean(np.array(family_true) == np.array(family_pred))),
        scenario_accuracy=float(np.mean(np.array(scenario_true) == np.array(scenario_pred))),
        binary_labels=binary_labels,
        binary_confusion=_confusion(binary_true, binary_pred, binary_labels),
        family_labels=family_labels,
        family_confusion=_confusion(family_true, family_pred, family_labels),
        scenario_labels=scenario_labels,
        scenario_confusion=_confusion(scenario_true, scenario_pred, scenario_labels),
    )
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding='utf-8')
    return result


def evaluate_hierarchical_over_conditions(
    feature_csv: Path,
    holdout_conditions: list[str],
    output_dir: Path,
    backend: str = 'rf',
) -> list[HierarchicalEvaluationResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for holdout_condition in holdout_conditions:
        result = evaluate_hierarchical_holdout(
            feature_csv=feature_csv,
            holdout_condition=holdout_condition,
            backend=backend,
            output_json=output_dir / f'hierarchical__{holdout_condition}.json',
        )
        results.append(result)

    summary_rows = [
        {
            'model_name': result.model_name,
            'backend': result.backend,
            'holdout_condition': result.holdout_condition,
            'train_samples': result.train_samples,
            'test_samples': result.test_samples,
            'binary_accuracy': result.binary_accuracy,
            'family_accuracy': result.family_accuracy,
            'scenario_accuracy': result.scenario_accuracy,
        }
        for result in results
    ]
    with (output_dir / 'hierarchical_leave_one_condition_summary.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    return results
