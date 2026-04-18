from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


def stratified_split(y: np.ndarray, ratios: tuple[float, float, float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = y.astype(np.int64)
    indices = np.arange(y.shape[0])
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    r_train, r_val, r_test = ratios
    if not math.isclose(r_train + r_val + r_test, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        total = r_train + r_val + r_test
        r_train, r_val, r_test = r_train / total, r_val / total, r_test / total

    for cls in np.unique(y):
        cls_idx = indices[y == cls]
        rng.shuffle(cls_idx)
        n = cls_idx.shape[0]
        n_train = int(math.floor(n * r_train))
        n_val = int(math.floor(n * r_val))
        train_idx.extend(cls_idx[:n_train].tolist())
        val_idx.extend(cls_idx[n_train : n_train + n_val].tolist())
        test_idx.extend(cls_idx[n_train + n_val :].tolist())

    train_idx = np.array(train_idx, dtype=np.int64)
    val_idx = np.array(val_idx, dtype=np.int64)
    test_idx = np.array(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def save_split(out_dir: Path, split_name: str, X: np.ndarray, y: np.ndarray) -> None:
    split_dir = out_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / "data.npy", X.astype(np.float32))
    np.save(split_dir / "labels.npy", y.astype(np.int64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--window-npz",
        type=Path,
        default=Path("derived_datasets/servo_window_handoff_dataset.npz"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("derived_datasets/servo_handoff_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments_smoke_20260316/traditional/window_dataset"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split", type=str, default="0.7,0.15,0.15")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    class_order = [str(x) for x in manifest["scenario_classes"]]
    class_to_idx = {name: idx for idx, name in enumerate(class_order)}

    data = np.load(args.window_npz, allow_pickle=True)
    X = data["X"].astype(np.float32)
    scenario = data["scenario"].astype(str)
    y = np.array([class_to_idx[s] for s in scenario], dtype=np.int64)

    split_parts = [float(x.strip()) for x in args.split.split(",") if x.strip()]
    if len(split_parts) != 3:
        raise SystemExit("--split must be 3 comma-separated ratios, e.g. 0.7,0.15,0.15")
    ratios = (split_parts[0], split_parts[1], split_parts[2])

    train_idx, val_idx, test_idx = stratified_split(y, ratios=ratios, seed=args.seed)
    save_split(args.output_dir, "train", X[train_idx], y[train_idx])
    save_split(args.output_dir, "val", X[val_idx], y[val_idx])
    save_split(args.output_dir, "test", X[test_idx], y[test_idx])

    metadata = {
        "source_npz": str(args.window_npz),
        "seed": int(args.seed),
        "split": {"ratios": list(ratios), "train": int(train_idx.size), "val": int(val_idx.size), "test": int(test_idx.size)},
        "num_classes": int(len(class_order)),
        "label_map": {str(i): name for i, name in enumerate(class_order)},
        "window_shape": [int(X.shape[1]), int(X.shape[2])],
        "signal_columns": [str(x) for x in data["signal_columns"].astype(str).tolist()] if "signal_columns" in data.files else [],
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
