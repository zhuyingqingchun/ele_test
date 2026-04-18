from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .benchmark_utils import (
    EvaluationResult,
    build_result,
    feature_rows_to_matrix,
    load_feature_rows,
    rows_to_fault_ids,
    save_result_json,
    split_feature_rows,
)


def available_backends() -> dict[str, bool]:
    return {
        "xgboost": importlib.util.find_spec("xgboost") is not None,
        "lightgbm": importlib.util.find_spec("lightgbm") is not None,
        "sklearn_rf_fallback": True,
    }


def make_tree_model(preferred_backend: str = "auto"):
    backends = available_backends()
    if preferred_backend in ("auto", "xgboost") and backends["xgboost"]:
        from xgboost import XGBClassifier

        return "xgboost", XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=7,
        )
    if preferred_backend in ("auto", "lightgbm") and backends["lightgbm"]:
        from lightgbm import LGBMClassifier

        return "lightgbm", LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=7,
        )
    return "sklearn_rf_fallback", RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=2,
        random_state=7,
        n_jobs=1,
    )


def evaluate_tree_holdout(
    feature_csv: Path,
    holdout_condition: str,
    preferred_backend: str = "auto",
    output_json: Path | None = None,
) -> EvaluationResult:
    rows, feature_columns = load_feature_rows(feature_csv)
    train_rows, test_rows = split_feature_rows(rows, holdout_condition)
    X_train = feature_rows_to_matrix(train_rows, feature_columns)
    X_test = feature_rows_to_matrix(test_rows, feature_columns)
    y_train_raw = rows_to_fault_ids(train_rows)
    y_test_raw = rows_to_fault_ids(test_rows)

    label_ids = sorted({int(label) for label in np.concatenate([y_train_raw, y_test_raw])})
    label_names = [f"fault_{label_id}" for label_id in label_ids]
    label_to_index = {label_id: idx for idx, label_id in enumerate(label_ids)}
    y_train = np.array([label_to_index[int(label)] for label in y_train_raw], dtype=np.int64)
    y_test = np.array([label_to_index[int(label)] for label in y_test_raw], dtype=np.int64)

    backend, model = make_tree_model(preferred_backend)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = build_result(
        model_name="tree_feature_baseline",
        backend=backend,
        holdout_condition=holdout_condition,
        y_true=y_test,
        y_pred=y_pred,
        label_names=label_names,
        train_samples=X_train.shape[0],
        test_samples=X_test.shape[0],
        extra={"num_features": X_train.shape[1]},
    )
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        save_result_json(output_json, result)
    return result
