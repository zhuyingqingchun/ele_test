from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments_smoke_20260316.train_timeseries_baseline_randomsplit import (
    apply_standardization,
    collect_predictions,
    compute_confusion_matrix,
    compute_standardization_stats,
    load_label_names,
    load_split,
    load_signal_columns,
    select_named_columns,
    write_confusion_matrix_csv,
    zero_named_columns,
)
from experiments_smoke_20260316.traditional_sequence_models import build_baseline_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Path to *_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    dataset_dir = Path(report["dataset"])
    checkpoint_path = Path(report["best_checkpoint"])
    model_name = str(report["model"])
    keep_columns = [str(x) for x in report.get("keep_columns", [])]
    zero_columns = [str(x) for x in report.get("zero_columns", [])]

    train_x, train_y = load_split(dataset_dir / "train")
    test_x, test_y = load_split(dataset_dir / "test")
    signal_columns = load_signal_columns(dataset_dir)
    label_names = load_label_names(dataset_dir)

    train_x, signal_columns = select_named_columns(train_x, signal_columns, keep_columns)
    test_x, _ = select_named_columns(test_x, load_signal_columns(dataset_dir), keep_columns)

    train_x = zero_named_columns(train_x, signal_columns, zero_columns)
    test_x = zero_named_columns(test_x, signal_columns, zero_columns)

    mean, std = compute_standardization_stats(train_x)
    test_x = apply_standardization(test_x, mean, std)

    num_classes = int(torch.max(train_y).item() + 1)
    num_channels = int(test_x.shape[2])
    device = torch.device(args.device)

    model = build_baseline_model(model_name, num_channels, num_classes).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256)
    y_true, y_pred = collect_predictions(model, test_loader, device)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    out_path = report_path.with_name(report_path.stem.replace("_report", "_test_confusion_matrix") + ".csv")
    write_confusion_matrix_csv(out_path, cm, label_names)
    print(out_path)


if __name__ == "__main__":
    main()
