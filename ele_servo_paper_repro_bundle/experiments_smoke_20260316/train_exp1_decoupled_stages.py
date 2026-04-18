from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from servo_diagnostic.multimodal_method import load_multimodal_arrays
from servo_llm_alignment.dataset import load_alignment_records
from servo_llm_alignment.text_encoder import MockFrozenTextEncoder, QwenFrozenTextEncoder

from experiments_smoke_20260316.exp1_decoupled_models import (
    Stage1DecoupledClassifier,
    Stage2DecoupledClassifier,
    Stage3DecoupledClassifier,
    Stage4DecoupledClassifier,
)


STRICT_DECOUPLED_COLUMNS = {
    "pos": ["theta_meas_deg", "theta_motor_meas_deg", "omega_motor_meas_deg_s", "encoder_count", "motor_encoder_count"],
    "electrical": [
        "current_meas_a",
        "current_d_meas_a",
        "current_q_meas_a",
        "bus_current_meas_a",
        "phase_voltage_a_meas_v",
        "phase_voltage_b_meas_v",
        "phase_voltage_c_meas_v",
        "voltage_meas_v",
        "available_bus_voltage_v",
    ],
    "thermal": ["winding_temp_c", "housing_temp_c"],
    "vibration": ["vibration_accel_mps2"],
}


@dataclass
class DecoupledBatch:
    pos: torch.Tensor
    electrical: torch.Tensor
    thermal: torch.Tensor
    vibration: torch.Tensor
    y_cls: torch.Tensor


FEATURE_MODE_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "base": {
        "name": "base",
        "description": "Use only the strict four-modality raw/near-raw time-series inputs.",
        "algorithms": {},
    },
    "modality_tf": {
        "name": "modality_tf",
        "description": "Apply modality-specific time/frequency feature augmentation before training.",
        "algorithms": {
            "pos": ["first_difference", "second_difference", "rolling_rms_velocity"],
            "electrical": ["rolling_rms", "fft_low_mid_high_band_energy"],
            "thermal": ["raw_only"],
            "vibration": ["envelope", "highpass_residual", "spectral_centroid"],
        },
    },
}


class DecoupledDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
        self.arrays = arrays
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> DecoupledBatch:
        item = int(self.indices[idx])
        return DecoupledBatch(
            pos=torch.from_numpy(self.arrays["X_pos"][item]).float(),
            electrical=torch.from_numpy(self.arrays["X_electrical"][item]).float(),
            thermal=torch.from_numpy(self.arrays["X_thermal"][item]).float(),
            vibration=torch.from_numpy(self.arrays["X_vibration"][item]).float(),
            y_cls=torch.tensor(self.arrays["y_cls"][item], dtype=torch.long),
        )


class StageTextDataset(Dataset):
    def __init__(self, base: DecoupledDataset, records: list, indices: np.ndarray) -> None:
        self.base = base
        self.records = records
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int):
        item = int(self.indices[idx])
        return self.base[item], self.records[item]


def collate_decoupled_batch(items: list[DecoupledBatch]) -> DecoupledBatch:
    return DecoupledBatch(
        pos=torch.stack([x.pos for x in items], dim=0),
        electrical=torch.stack([x.electrical for x in items], dim=0),
        thermal=torch.stack([x.thermal for x in items], dim=0),
        vibration=torch.stack([x.vibration for x in items], dim=0),
        y_cls=torch.stack([x.y_cls for x in items], dim=0),
    )


def collate_with_records(items):
    batches = [item[0] for item in items]
    recs = [item[1] for item in items]
    return collate_decoupled_batch(batches), recs


def move_batch_to_device(batch: DecoupledBatch, device: torch.device) -> DecoupledBatch:
    return DecoupledBatch(
        pos=batch.pos.to(device),
        electrical=batch.electrical.to(device),
        thermal=batch.thermal.to(device),
        vibration=batch.vibration.to(device),
        y_cls=batch.y_cls.to(device),
    )


def batch_input_dims(batch: DecoupledBatch) -> dict[str, int]:
    return {
        "pos": int(batch.pos.shape[-1]),
        "electrical": int(batch.electrical.shape[-1]),
        "thermal": int(batch.thermal.shape[-1]),
        "vibration": int(batch.vibration.shape[-1]),
    }


def _moving_average_same(x: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return x.astype(np.float32)
    pad_left = kernel // 2
    pad_right = kernel - 1 - pad_left
    padded = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="edge")
    prefix = np.concatenate(
        [np.zeros((padded.shape[0], 1, padded.shape[2]), dtype=padded.dtype), np.cumsum(padded, axis=1)],
        axis=1,
    )
    return ((prefix[:, kernel:, :] - prefix[:, :-kernel, :]) / float(kernel)).astype(np.float32)


def _first_difference(x: np.ndarray) -> np.ndarray:
    return np.diff(x, axis=1, prepend=x[:, :1, :]).astype(np.float32)


def _second_difference(x: np.ndarray) -> np.ndarray:
    d1 = _first_difference(x)
    return np.diff(d1, axis=1, prepend=d1[:, :1, :]).astype(np.float32)


def _rolling_rms(x: np.ndarray, kernel: int) -> np.ndarray:
    return np.sqrt(np.maximum(_moving_average_same(np.square(x).astype(np.float32), kernel), 0.0)).astype(np.float32)


def _low_order_dct_trend(x: np.ndarray, keep_bins: int) -> np.ndarray:
    n = int(x.shape[1])
    keep = max(1, min(int(keep_bins), n))
    t = np.arange(n, dtype=np.float32)
    basis = np.empty((keep, n), dtype=np.float32)
    basis[0] = 1.0 / np.sqrt(float(n))
    for k in range(1, keep):
        basis[k] = np.sqrt(2.0 / float(n)) * np.cos((np.pi / float(n)) * (t + 0.5) * k)
    mean = x.mean(axis=1, keepdims=True).astype(np.float32)
    centered = (x - mean).astype(np.float32)
    coeffs = np.einsum("kn,btc->bkc", basis, centered, optimize=True)
    recon = np.einsum("kn,bkc->bnc", basis, coeffs, optimize=True)
    return (recon + mean).astype(np.float32)


def _fft_band_energy_map(x: np.ndarray, num_bands: int = 3) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x, axis=1)).astype(np.float32)
    freq_bins = int(spec.shape[1])
    usable = max(freq_bins - 1, 1)
    edges = np.linspace(1, freq_bins, num_bands + 1, dtype=int)
    band_maps: list[np.ndarray] = []
    for start, end in zip(edges[:-1], edges[1:]):
        start = max(1, int(start))
        end = max(start + 1, int(end))
        band = spec[:, start:end, :].mean(axis=1, keepdims=True)
        band_maps.append(np.repeat(band, x.shape[1], axis=1))
    return np.concatenate(band_maps, axis=2).astype(np.float32)


def _spectral_centroid_map(x: np.ndarray) -> np.ndarray:
    spec = np.abs(np.fft.rfft(x, axis=1)).astype(np.float32)
    freqs = np.arange(spec.shape[1], dtype=np.float32).reshape(1, -1, 1)
    denom = np.maximum(spec.sum(axis=1, keepdims=True), 1.0e-6)
    centroid = (spec * freqs).sum(axis=1, keepdims=True) / denom
    centroid = centroid / max(float(spec.shape[1] - 1), 1.0)
    return np.repeat(centroid.astype(np.float32), x.shape[1], axis=1)


def _augment_position_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    vel = _first_difference(x)
    acc = _second_difference(x)
    motion_energy = _rolling_rms(vel, kernel=9)
    return np.concatenate([x, vel, acc, motion_energy], axis=2).astype(np.float32), [
        "raw",
        "delta",
        "delta2",
        "delta_rms",
    ]


def _augment_electrical_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    rms = _rolling_rms(x, kernel=9)
    band_maps = _fft_band_energy_map(x, num_bands=3)
    return np.concatenate([x, rms, band_maps], axis=2).astype(np.float32), [
        "raw",
        "rolling_rms",
        "fft_band_energy_low_mid_high",
    ]


def _augment_thermal_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    return x.astype(np.float32, copy=False), ["raw"]


def _augment_vibration_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    envelope = _moving_average_same(np.abs(x).astype(np.float32), kernel=9)
    lowpass = _moving_average_same(x, kernel=9)
    highpass = (x - lowpass).astype(np.float32)
    centroid = _spectral_centroid_map(x)
    return np.concatenate([x, envelope, highpass, centroid], axis=2).astype(np.float32), [
        "raw",
        "envelope",
        "highpass_residual",
        "spectral_centroid",
    ]


def apply_feature_mode(
    arrays: dict[str, np.ndarray],
    feature_mode: str,
    *,
    chunk_size: int = 2048,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if feature_mode == "base":
        return arrays, FEATURE_MODE_DESCRIPTIONS["base"]
    if feature_mode != "modality_tf":
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    augmented = dict(arrays)
    feature_meta = {
        "name": "modality_tf",
        "description": FEATURE_MODE_DESCRIPTIONS["modality_tf"]["description"],
        "algorithms": FEATURE_MODE_DESCRIPTIONS["modality_tf"]["algorithms"],
        "channel_expansion": {},
    }
    augmenters = {
        "X_pos": _augment_position_features,
        "X_electrical": _augment_electrical_features,
        "X_thermal": _augment_thermal_features,
        "X_vibration": _augment_vibration_features,
    }
    for key, fn in augmenters.items():
        source = arrays[key]
        chunks: list[np.ndarray] = []
        feature_groups: list[str] | None = None
        for start in range(0, int(source.shape[0]), int(chunk_size)):
            end = min(start + int(chunk_size), int(source.shape[0]))
            out_chunk, groups = fn(source[start:end].astype(np.float32, copy=False))
            chunks.append(out_chunk.astype(np.float32, copy=False))
            if feature_groups is None:
                feature_groups = groups
        augmented[key] = np.concatenate(chunks, axis=0).astype(np.float32)
        feature_meta["channel_expansion"][key] = {
            "original_dim": int(source.shape[2]),
            "augmented_dim": int(augmented[key].shape[2]),
            "feature_groups": feature_groups or ["raw"],
        }
    return augmented, feature_meta


def compute_normalization_stats(arrays: dict[str, np.ndarray], train_idx: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    stats = {}
    for key in ["X_pos", "X_electrical", "X_thermal", "X_vibration"]:
        train_values = arrays[key][train_idx]
        mean = train_values.mean(axis=(0, 1), keepdims=True).astype(np.float32)
        std = train_values.std(axis=(0, 1), keepdims=True).astype(np.float32)
        std[std < 1.0e-6] = 1.0
        stats[key] = {"mean": mean, "std": std}
    return stats


def apply_normalization(arrays: dict[str, np.ndarray], stats: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for key, value in arrays.items():
        if key in stats:
            out[key] = ((value - stats[key]["mean"]) / stats[key]["std"]).astype(np.float32)
        else:
            out[key] = value
    return out


def stratified_split_three_way(labels: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_indices = np.arange(labels.shape[0], dtype=np.int64)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(set(labels.tolist())):
        class_idx = all_indices[labels == label].copy()
        rng.shuffle(class_idx)
        n = int(class_idx.shape[0])
        val_count = max(1, int(round(n * val_ratio))) if val_ratio > 0 else 0
        remaining_after_val = max(n - val_count, 1)
        test_count = max(1, int(round(n * test_ratio))) if test_ratio > 0 else 0
        test_count = min(test_count, max(remaining_after_val - 1, 0))
        val_count = min(val_count, max(n - test_count - 1, 0))
        val_parts.append(class_idx[:val_count])
        test_parts.append(class_idx[val_count : val_count + test_count])
        train_parts.append(class_idx[val_count + test_count :])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def load_decoupled_arrays(dataset_path: Path, corpus_path: Path | None, max_samples: int):
    arrays = load_multimodal_arrays(dataset_path)
    records = None
    if corpus_path is not None:
        records = load_alignment_records(corpus_path)
    total_len = int(arrays["y_cls"].shape[0])
    if max_samples > 0:
        keep = min(int(max_samples), total_len)
        arrays = {k: (v[:keep] if isinstance(v, np.ndarray) and v.shape and v.shape[0] == total_len else v) for k, v in arrays.items()}
        if records is not None:
            records = records[:keep]

    def pick(source_key: str, column_key: str, wanted: list[str]) -> np.ndarray:
        source_cols = [str(x) for x in arrays[column_key].astype(str).tolist()]
        idx = [source_cols.index(name) for name in wanted]
        return arrays[source_key][:, :, idx].astype(np.float32)

    out = {
        "X_pos": pick("X_pos", "pos_columns", STRICT_DECOUPLED_COLUMNS["pos"]),
        "X_electrical": pick("X_elec", "elec_columns", STRICT_DECOUPLED_COLUMNS["electrical"]),
        "X_thermal": pick("X_therm", "therm_columns", STRICT_DECOUPLED_COLUMNS["thermal"]),
        "X_vibration": pick("X_vib", "vib_columns", STRICT_DECOUPLED_COLUMNS["vibration"]),
        "y_cls": arrays["y_cls"].astype(np.int64),
        "class_names": arrays["class_names"],
    }
    return out, records


def apply_decoupled_ablation(
    arrays: dict[str, np.ndarray],
    *,
    zero_pos: bool,
    zero_electrical: bool,
    zero_thermal: bool,
    zero_vibration: bool,
) -> dict[str, np.ndarray]:
    masked = dict(arrays)
    ablation_map = {
        "X_pos": zero_pos,
        "X_electrical": zero_electrical,
        "X_thermal": zero_thermal,
        "X_vibration": zero_vibration,
    }
    for key, enabled in ablation_map.items():
        if enabled and key in masked:
            masked[key] = np.zeros_like(masked[key])
    return masked


def build_text_cache(
    output_dir: Path,
    records: list,
    text_backbone: str,
    qwen_path: Path,
    device: torch.device,
    text_batch_size: int,
):
    if text_backbone == "qwen":
        text_encoder = QwenFrozenTextEncoder(str(qwen_path), device=device)
        text_dim = int(text_encoder.hidden_size)
    else:
        text_encoder = MockFrozenTextEncoder(hidden_size=768)
        text_dim = int(text_encoder.hidden_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"text_embeddings__{text_backbone}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False), text_dim
    texts = [getattr(r, "texts", {}).get("combined_text", "") or getattr(r, "text", "") for r in records]
    cache = text_encoder.encode_texts(texts, batch_size=text_batch_size)
    torch.save(cache, cache_path)
    return cache, text_dim


def compute_batch_loss(
    outputs,
    y_cls: torch.Tensor,
    label_smoothing: float,
    lambda_align: float,
    lambda_quality: float,
) -> tuple[torch.Tensor, float, float]:
    cls_loss = F.cross_entropy(outputs.logits, y_cls, label_smoothing=label_smoothing)
    align_val = float(outputs.align_loss.item()) if outputs.align_loss is not None else 0.0
    quality_val = float(outputs.quality_loss.item()) if getattr(outputs, "quality_loss", None) is not None else 0.0
    total = cls_loss
    if outputs.align_loss is not None:
        total = total + lambda_align * outputs.align_loss
    if getattr(outputs, "quality_loss", None) is not None:
        total = total + lambda_quality * outputs.quality_loss
    return total, align_val, quality_val


def evaluate(model, loader, device: torch.device, *, text_cache=None, label_smoothing: float, lambda_align: float, return_predictions: bool = False):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    align_sum = 0.0
    quality_sum = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, tuple):
                batch, recs = batch
                idx = torch.as_tensor([int(r.index) for r in recs], dtype=torch.long)
                text_emb = text_cache[idx].to(device)
            else:
                text_emb = None
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, text_emb) if text_emb is not None else model(batch)
            loss, align_val, quality_val = compute_batch_loss(
                outputs,
                batch.y_cls,
                label_smoothing=label_smoothing,
                lambda_align=lambda_align,
                lambda_quality=0.0,
            )
            preds = outputs.logits.argmax(dim=1)
            correct += int((preds == batch.y_cls).sum().item())
            total += int(batch.y_cls.numel())
            loss_sum += float(loss.item()) * int(batch.y_cls.numel())
            align_sum += align_val * int(batch.y_cls.numel())
            quality_sum += quality_val * int(batch.y_cls.numel())
            if return_predictions:
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(batch.y_cls.cpu().numpy().tolist())
    result = {
        "loss": loss_sum / max(1, total),
        "scenario_accuracy": correct / max(1, total),
        "align_loss": align_sum / max(1, total),
        "quality_loss": quality_sum / max(1, total),
    }
    if return_predictions:
        result["predictions"] = all_preds
        result["targets"] = all_targets
    return result


def load_init(model, init_path: Path) -> None:
    payload = torch.load(init_path, map_location="cpu", weights_only=False)
    state = payload.get("model") or payload.get("model_state") or payload.get("state_dict") or payload
    model.load_state_dict(state, strict=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "derived_datasets" / "servo_multimodal_handoff_dataset.npz")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "derived_datasets" / "stage1_alignment_corpus.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--lambda-align", type=float, default=0.20)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--text-backbone", choices=["qwen", "mock"], default="mock")
    parser.add_argument("--qwen-path", type=Path, default=Path("/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--fusion-ff", type=int, default=768)
    parser.add_argument("--llm-layers", type=int, default=4)
    parser.add_argument("--llm-heads", type=int, default=8)
    parser.add_argument("--llm-ff", type=int, default=768)
    parser.add_argument("--pool", choices=["cls", "mean", "attn", "text"], default="cls")
    parser.add_argument("--feature-mode", choices=["base", "modality_tf"], default="base")
    parser.add_argument("--feature-chunk-size", type=int, default=2048)
    parser.add_argument("--quality-aware-fusion", action="store_true")
    parser.add_argument("--quality-hidden-dim", type=int, default=128)
    parser.add_argument("--quality-drop-prob", type=float, default=0.0)
    parser.add_argument("--quality-min-gate", type=float, default=0.10)
    parser.add_argument("--lambda-quality", type=float, default=0.10)
    parser.add_argument("--zero-pos", action="store_true")
    parser.add_argument("--zero-electrical", action="store_true")
    parser.add_argument("--zero-thermal", action="store_true")
    parser.add_argument("--zero-vibration", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    need_text = int(args.stage) in (3, 4)
    arrays, records = load_decoupled_arrays(args.dataset, args.corpus if need_text else None, int(args.max_samples))
    arrays = apply_decoupled_ablation(
        arrays,
        zero_pos=bool(args.zero_pos),
        zero_electrical=bool(args.zero_electrical),
        zero_thermal=bool(args.zero_thermal),
        zero_vibration=bool(args.zero_vibration),
    )
    arrays, feature_meta = apply_feature_mode(arrays, str(args.feature_mode), chunk_size=int(args.feature_chunk_size))
    y_cls = arrays["y_cls"].astype(np.int64)
    train_idx, val_idx, test_idx = stratified_split_three_way(
        y_cls,
        float(args.val_ratio),
        float(args.test_ratio),
        int(args.seed),
    )
    stats = compute_normalization_stats(arrays, train_idx)
    normalized = apply_normalization(arrays, stats)
    full_indices = np.arange(y_cls.shape[0], dtype=np.int64)
    base_dataset = DecoupledDataset(normalized, full_indices)

    if need_text:
        if records is None:
            raise SystemExit("stage 3/4 require --corpus")
        train_loader = DataLoader(StageTextDataset(base_dataset, records, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_records)
        val_loader = DataLoader(StageTextDataset(base_dataset, records, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_records)
        test_loader = DataLoader(StageTextDataset(base_dataset, records, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_records)
    else:
        train_loader = DataLoader(DecoupledDataset(normalized, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_decoupled_batch)
        val_loader = DataLoader(DecoupledDataset(normalized, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_decoupled_batch)
        test_loader = DataLoader(DecoupledDataset(normalized, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_decoupled_batch)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    input_dims = batch_input_dims(base_dataset[0])
    num_classes = 17

    text_cache = None
    text_dim = 0
    if need_text:
        text_cache, text_dim = build_text_cache(args.output_dir, records, str(args.text_backbone), args.qwen_path, device, int(args.text_batch_size))

    if int(args.stage) == 1:
        model = Stage1DecoupledClassifier(
            input_dims,
            num_classes,
            model_dim=int(args.model_dim),
            token_dim=int(args.token_dim),
            quality_aware_fusion=bool(args.quality_aware_fusion),
            quality_hidden_dim=int(args.quality_hidden_dim),
            modality_drop_prob=float(args.quality_drop_prob),
            quality_min_gate=float(args.quality_min_gate),
        ).to(device)
    elif int(args.stage) == 2:
        model = Stage2DecoupledClassifier(
            input_dims,
            num_classes,
            model_dim=int(args.model_dim),
            token_dim=int(args.token_dim),
            num_layers=int(args.fusion_layers),
            nhead=int(args.fusion_heads),
            dim_feedforward=int(args.fusion_ff),
            pool=str(args.pool),
            quality_aware_fusion=bool(args.quality_aware_fusion),
            quality_hidden_dim=int(args.quality_hidden_dim),
            modality_drop_prob=float(args.quality_drop_prob),
            quality_min_gate=float(args.quality_min_gate),
        ).to(device)
    elif int(args.stage) == 3:
        model = Stage3DecoupledClassifier(
            input_dims,
            num_classes,
            text_dim=int(text_dim),
            model_dim=int(args.model_dim),
            token_dim=int(args.token_dim),
            quality_aware_fusion=bool(args.quality_aware_fusion),
            quality_hidden_dim=int(args.quality_hidden_dim),
            modality_drop_prob=float(args.quality_drop_prob),
            quality_min_gate=float(args.quality_min_gate),
        ).to(device)
    else:
        model = Stage4DecoupledClassifier(
            input_dims,
            num_classes,
            text_dim=int(text_dim),
            model_dim=int(args.model_dim),
            token_dim=int(args.token_dim),
            num_layers=int(args.llm_layers),
            nhead=int(args.llm_heads),
            dim_feedforward=int(args.llm_ff),
            pool=str(args.pool),
            quality_aware_fusion=bool(args.quality_aware_fusion),
            quality_hidden_dim=int(args.quality_hidden_dim),
            modality_drop_prob=float(args.quality_drop_prob),
            quality_min_gate=float(args.quality_min_gate),
        ).to(device)

    if args.init is not None:
        load_init(model, args.init)
    if args.freeze_backbone and hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = False

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    report_path = args.output_dir / "report.json"

    best_acc = -math.inf
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0
        loss_sum = 0.0
        align_sum = 0.0
        quality_sum = 0.0
        correct = 0
        for batch in train_loader:
            if isinstance(batch, tuple):
                batch, recs = batch
                idx = torch.as_tensor([int(r.index) for r in recs], dtype=torch.long)
                text_emb = text_cache[idx].to(device)
            else:
                text_emb = None
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, text_emb) if text_emb is not None else model(batch)
            loss, align_val, quality_val = compute_batch_loss(
                outputs,
                batch.y_cls,
                float(args.label_smoothing),
                float(args.lambda_align),
                float(args.lambda_quality),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = outputs.logits.argmax(dim=1)
            correct += int((preds == batch.y_cls).sum().item())
            total += int(batch.y_cls.numel())
            loss_sum += float(loss.item()) * int(batch.y_cls.numel())
            align_sum += align_val * int(batch.y_cls.numel())
            quality_sum += quality_val * int(batch.y_cls.numel())

        val_metrics = evaluate(model, val_loader, device, text_cache=text_cache if need_text else None, label_smoothing=float(args.label_smoothing), lambda_align=float(args.lambda_align))
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(1, total),
            "scenario_accuracy": correct / max(1, total),
            "align_loss": align_sum / max(1, total),
            "quality_loss": quality_sum / max(1, total),
            "val_loss": val_metrics["loss"],
            "val_scenario_accuracy": val_metrics["scenario_accuracy"],
            "val_align_loss": val_metrics["align_loss"],
            "val_quality_loss": val_metrics["quality_loss"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["scenario_accuracy"] > best_acc:
            best_acc = val_metrics["scenario_accuracy"]
            torch.save({"model": model.state_dict(), "config": vars(args)}, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    train_results = evaluate(model, train_loader, device, text_cache=text_cache if need_text else None, label_smoothing=float(args.label_smoothing), lambda_align=float(args.lambda_align), return_predictions=True)
    val_results = evaluate(model, val_loader, device, text_cache=text_cache if need_text else None, label_smoothing=float(args.label_smoothing), lambda_align=float(args.lambda_align), return_predictions=True)
    test_results = evaluate(model, test_loader, device, text_cache=text_cache if need_text else None, label_smoothing=float(args.label_smoothing), lambda_align=float(args.lambda_align), return_predictions=True)

    cm_output = {
        "train_confusion_matrix": confusion_matrix(train_results["targets"], train_results["predictions"]).tolist(),
        "val_confusion_matrix": confusion_matrix(val_results["targets"], val_results["predictions"]).tolist(),
        "test_confusion_matrix": confusion_matrix(test_results["targets"], test_results["predictions"]).tolist(),
        "labels": {str(i): str(name) for i, name in enumerate(arrays["class_names"].astype(str).tolist())},
        "strict_modalities": STRICT_DECOUPLED_COLUMNS,
        "feature_mode": feature_meta,
        "thermal_feature_policy": "raw_only",
    }
    (args.output_dir / "confusion_matrix.json").write_text(json.dumps(cm_output, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "stage": int(args.stage),
        "best_checkpoint": str(best_path),
        "best_val_scenario_accuracy": float(best_acc),
        "train_scenario_accuracy": float(train_results["scenario_accuracy"]),
        "val_scenario_accuracy": float(val_results["scenario_accuracy"]),
        "test_scenario_accuracy": float(test_results["scenario_accuracy"]),
        "train_loss": float(train_results["loss"]),
        "val_loss": float(val_results["loss"]),
        "test_loss": float(test_results["loss"]),
        "train_quality_loss": float(train_results["quality_loss"]),
        "val_quality_loss": float(val_results["quality_loss"]),
        "test_quality_loss": float(test_results["quality_loss"]),
        "strict_modalities": STRICT_DECOUPLED_COLUMNS,
        "feature_mode": feature_meta,
        "thermal_feature_policy": "raw_only",
        "quality_aware_fusion": {
            "enabled": bool(args.quality_aware_fusion),
            "quality_hidden_dim": int(args.quality_hidden_dim),
            "quality_drop_prob": float(args.quality_drop_prob),
            "quality_min_gate": float(args.quality_min_gate),
            "lambda_quality": float(args.lambda_quality),
        },
        "split": {
            "mode": "random_stratified",
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "test": int(test_idx.shape[0]),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(args.test_ratio),
        },
        "history": history,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
