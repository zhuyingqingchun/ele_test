from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments_smoke_20260316.traditional_sequence_models import build_baseline_model


def load_split(split_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.load(split_dir / "data.npy")).float()
    y = torch.from_numpy(np.load(split_dir / "labels.npy")).long()
    return x, y


def compute_standardization_stats(train_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=(0, 1), keepdim=True)
    std = train_x.std(dim=(0, 1), keepdim=True)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return mean, std


def apply_standardization(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def load_signal_columns(dataset_dir: Path) -> list[str]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return [str(x) for x in metadata.get("signal_columns", [])]


def load_label_names(dataset_dir: Path) -> list[str]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    label_map = metadata.get("label_map", {})
    if not label_map:
        return []
    return [str(label_map[str(i)]) for i in sorted(int(k) for k in label_map.keys())]


def select_named_columns(
    x: torch.Tensor,
    signal_columns: list[str],
    keep_columns: list[str],
) -> tuple[torch.Tensor, list[str]]:
    if not keep_columns:
        return x, signal_columns
    column_to_idx = {name: idx for idx, name in enumerate(signal_columns)}
    missing = [name for name in keep_columns if name not in column_to_idx]
    if missing:
        raise ValueError(f"Missing keep-columns in dataset metadata: {missing}")
    idxs = [column_to_idx[name] for name in keep_columns]
    return x[:, :, idxs], list(keep_columns)


def zero_named_columns(x: torch.Tensor, signal_columns: list[str], zero_columns: list[str]) -> torch.Tensor:
    if not zero_columns:
        return x
    column_to_idx = {name: idx for idx, name in enumerate(signal_columns)}
    idxs = [column_to_idx[name] for name in zero_columns if name in column_to_idx]
    if not idxs:
        return x
    out = x.clone()
    out[:, :, idxs] = 0.0
    return out


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * int(y.numel())
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
    return loss_sum / max(1, total), correct / max(1, total)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            y_pred.append(preds)
            y_true.append(y.cpu())
    return torch.cat(y_true), torch.cat(y_pred)


def compute_confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_idx, pred_idx in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(true_idx), int(pred_idx)] += 1
    return cm


def write_confusion_matrix_csv(path: Path, cm: np.ndarray, label_names: list[str]) -> None:
    header_names = label_names if label_names and len(label_names) == cm.shape[0] else [str(i) for i in range(cm.shape[0])]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual/pred", *header_names])
        for idx, row in enumerate(cm.tolist()):
            writer.writerow([header_names[idx], *row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset directory with train/val/test splits.")
    parser.add_argument(
        "--model",
        choices=["cnn_tcn", "bilstm", "resnet_fcn", "transformer", "dual_branch_xattn"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--keep-columns", type=str, default="")
    parser.add_argument("--zero-columns", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = Path(args.dataset)
    train_x, train_y = load_split(dataset_dir / "train")
    val_x, val_y = load_split(dataset_dir / "val")
    test_x, test_y = load_split(dataset_dir / "test")
    signal_columns = load_signal_columns(dataset_dir)
    label_names = load_label_names(dataset_dir)
    keep_columns = [x.strip() for x in args.keep_columns.split(",") if x.strip()]
    zero_columns = [x.strip() for x in args.zero_columns.split(",") if x.strip()]

    train_x, signal_columns = select_named_columns(train_x, signal_columns, keep_columns)
    val_x, _ = select_named_columns(val_x, load_signal_columns(dataset_dir), keep_columns)
    test_x, _ = select_named_columns(test_x, load_signal_columns(dataset_dir), keep_columns)

    train_x = zero_named_columns(train_x, signal_columns, zero_columns)
    val_x = zero_named_columns(val_x, signal_columns, zero_columns)
    test_x = zero_named_columns(test_x, signal_columns, zero_columns)

    mean, std = compute_standardization_stats(train_x)
    train_x = apply_standardization(train_x, mean, std)
    val_x = apply_standardization(val_x, mean, std)
    test_x = apply_standardization(test_x, mean, std)

    num_classes = int(torch.max(train_y).item() + 1)
    num_channels = int(train_x.shape[2])

    device = torch.device(args.device)
    model = build_baseline_model(args.model, num_channels, num_classes).to(device)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or f"{args.model}_randomsplit"
    best_path = save_dir / f"{run_name}.pt"
    report_path = save_dir / f"{run_name}_report.json"
    confusion_matrix_csv_path = save_dir / f"{run_name}_test_confusion_matrix.csv"

    best_val = -math.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * int(y.numel())
            total += int(y.numel())

        val_loss, val_acc = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, total),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": args.model,
                    "best_val_acc": best_val,
                    "num_classes": num_classes,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    test_y_true, test_y_pred = collect_predictions(model, test_loader, device)
    test_confusion_matrix = compute_confusion_matrix(test_y_true, test_y_pred, num_classes)
    write_confusion_matrix_csv(confusion_matrix_csv_path, test_confusion_matrix, label_names)

    report = {
        "dataset": str(dataset_dir),
        "model": args.model,
        "best_checkpoint": str(best_path),
        "test_confusion_matrix_csv": str(confusion_matrix_csv_path),
        "best_val_acc": float(best_val),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "standardization": "train_split_mean_std",
        "split_sizes": {
            "train": int(train_y.shape[0]),
            "val": int(val_y.shape[0]),
            "test": int(test_y.shape[0]),
        },
        "signal_columns": signal_columns,
        "keep_columns": keep_columns,
        "zero_columns": zero_columns,
        "history": history,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
