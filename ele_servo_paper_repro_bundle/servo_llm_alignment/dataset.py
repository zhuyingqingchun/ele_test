from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from servo_diagnostic.multimodal_method import (
    MultimodalBatch,
    ServoMultimodalDataset,
    apply_normalization,
    collate_multimodal_batch,
    compute_normalization_stats,
    load_multimodal_arrays,
)


@dataclass(frozen=True)
class TextAlignmentRecord:
    index: int
    scenario: str
    family: str
    location: str
    boundary: str
    severity: float
    condition_name: str
    source_scenario: str
    texts: dict[str, str]


class Stage1AlignmentDataset(Dataset):
    def __init__(self, base_dataset: ServoMultimodalDataset, records: list[TextAlignmentRecord], indices: np.ndarray) -> None:
        self.base_dataset = base_dataset
        self.records = records
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> tuple[MultimodalBatch, TextAlignmentRecord]:
        item = int(self.indices[idx])
        return self.base_dataset[item], self.records[item]


def load_alignment_records(path: Path) -> list[TextAlignmentRecord]:
    records: list[TextAlignmentRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            # Only pass the fields that TextAlignmentRecord expects
            record_data = {
                "index": data["index"],
                "scenario": data["scenario"],
                "family": data["family"],
                "location": data["location"],
                "boundary": data["boundary"],
                "severity": data["severity"],
                "condition_name": data["condition_name"],
                "source_scenario": data["source_scenario"],
                "texts": data["texts"]
            }
            records.append(TextAlignmentRecord(**record_data))
    return records


def collate_stage1_batch(items: list[tuple[MultimodalBatch, TextAlignmentRecord]]) -> tuple[MultimodalBatch, list[TextAlignmentRecord]]:
    signal_items = [item[0] for item in items]
    text_items = [item[1] for item in items]
    return collate_multimodal_batch(signal_items), text_items


def stratified_split(labels: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(labels.shape[0], dtype=np.int64)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in sorted(set(labels.tolist())):
        class_idx = all_indices[labels == label].copy()
        rng.shuffle(class_idx)
        val_count = max(1, int(round(class_idx.shape[0] * val_ratio)))
        val_count = min(val_count, max(class_idx.shape[0] - 1, 1))
        val_parts.append(class_idx[:val_count])
        train_parts.append(class_idx[val_count:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def build_stage1_dataloaders(
    dataset_path: Path,
    corpus_path: Path,
    batch_size: int,
    val_ratio: float,
    seed: int,
    max_samples: int = 0,
    num_workers: int = 0,
) -> tuple[dict[str, np.ndarray], DataLoader, DataLoader]:
    arrays = load_multimodal_arrays(dataset_path)
    records = load_alignment_records(corpus_path)
    total_count = int(arrays["y_cls"].shape[0])
    if max_samples > 0:
        expected = min(max_samples, total_count)
        if len(records) < expected:
            raise ValueError("Corpus size is smaller than the requested max_samples subset.")
    elif len(records) != total_count:
        raise ValueError("Corpus size does not match dataset size.")

    if max_samples > 0:
        subset_idx = np.arange(min(max_samples, total_count), dtype=np.int64)
        arrays = dict(arrays)
        for key, value in list(arrays.items()):
            if isinstance(value, np.ndarray) and value.shape[0] == total_count:
                arrays[key] = value[subset_idx]
        records = [records[int(idx)] for idx in subset_idx]
    train_idx, val_idx = stratified_split(arrays["y_cls"].astype(np.int64), val_ratio=val_ratio, seed=seed)
    stats = compute_normalization_stats(arrays, train_idx)
    normalized = apply_normalization(arrays, stats)
    base_dataset = ServoMultimodalDataset(normalized, np.arange(arrays["y_cls"].shape[0], dtype=np.int64))

    train_dataset = Stage1AlignmentDataset(base_dataset, records, train_idx)
    val_dataset = Stage1AlignmentDataset(base_dataset, records, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_stage1_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_stage1_batch)

    metadata = {
        "class_names": arrays["class_names"],
        "family_names": arrays["family_names"],
        "location_names": arrays["location_names"],
        "boundary_names": arrays["boundary_names"],
        "scenario_to_family_idx": arrays["scenario_to_family_idx"],
        "train_idx": train_idx,
        "val_idx": val_idx,
    }
    return metadata, train_loader, val_loader
