from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import CNN1DClassifier

from .benchmark_utils import EvaluationResult, build_result, load_window_npz, save_result_json, split_window_dataset, standardize_window_channels


class TorchCNNTrainer:
    def __init__(self, num_channels: int, num_classes: int, learning_rate: float = 1e-3) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN1DClassifier(num_channels, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 8, batch_size: int = 64) -> dict[str, float | int | str]:
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        final_loss = 0.0
        self.model.train()
        for _ in range(epochs):
            running_loss = 0.0
            total = 0
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                loss.backward()
                self.optimizer.step()
                running_loss += float(loss.item()) * batch_X.size(0)
                total += batch_X.size(0)
            final_loss = running_loss / max(total, 1)
        return {"epochs": epochs, "batch_size": batch_size, "final_train_loss": final_loss, "device": str(self.device)}

    def predict(self, X_test: np.ndarray, batch_size: int = 256) -> np.ndarray:
        dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        outputs: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for (batch_X,) in loader:
                logits = self.model(batch_X.to(self.device))
                outputs.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0)


def evaluate_cnn1d_holdout(
    dataset_path: Path,
    holdout_condition: str,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    output_json: Path | None = None,
) -> EvaluationResult:
    data = load_window_npz(dataset_path)
    train, test = split_window_dataset(data, holdout_condition)
    label_ids = np.unique(data["y_fault_id"].astype(np.int64))
    label_names = [f"fault_{int(label_id)}" for label_id in label_ids]
    label_to_index = {int(label_id): idx for idx, label_id in enumerate(label_ids)}

    X_train = train["X"].astype(np.float32)
    X_test = test["X"].astype(np.float32)
    y_train_raw = train["y_fault_id"].astype(np.int64)
    y_test_raw = test["y_fault_id"].astype(np.int64)
    y_train = np.array([label_to_index[int(label)] for label in y_train_raw], dtype=np.int64)
    y_test = np.array([label_to_index[int(label)] for label in y_test_raw], dtype=np.int64)
    X_train_std, X_test_std, _, _ = standardize_window_channels(X_train, X_test)

    trainer = TorchCNNTrainer(num_channels=X_train_std.shape[2], num_classes=len(label_names), learning_rate=learning_rate)
    extra = trainer.fit(X_train_std, y_train, epochs=epochs, batch_size=batch_size)
    y_pred = trainer.predict(X_test_std)
    result = build_result(
        model_name="cnn1d",
        backend="pytorch",
        holdout_condition=holdout_condition,
        y_true=y_test,
        y_pred=y_pred,
        label_names=label_names,
        train_samples=X_train.shape[0],
        test_samples=X_test.shape[0],
        extra=extra,
    )
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        save_result_json(output_json, result)
    return result
