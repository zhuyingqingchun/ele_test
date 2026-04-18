# file name: servo_llm_alignment/runtime.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from servo_diagnostic.multimodal_method import (
    ServoMultimodalDataset,
    apply_normalization,
    batch_input_dims,
    compute_normalization_stats,
    load_multimodal_arrays,
)

from .dataset import load_alignment_records, stratified_split
from .prototypes import Stage2PrototypeBank, build_stage2_prototype_bank
from .signal_encoder import SignalAlignmentEncoder
from .text_encoder import MockFrozenTextEncoder, QwenFrozenTextEncoder


@dataclass
class Stage1Runtime:
    model: SignalAlignmentEncoder
    text_encoder: object
    prototype_bank: Stage2PrototypeBank
    arrays: dict
    normalized_arrays: dict
    base_dataset: ServoMultimodalDataset
    records: list
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    device: torch.device
    checkpoint: dict


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def load_stage1_runtime(
    checkpoint_path: Path,
    dataset_path: Path,
    corpus_path: Path,
    device: str = "cuda",
    text_backbone: str = "qwen",
    qwen_path: Path | None = None,
    text_batch_size: int = 8,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    runtime_device = resolve_device(device)

    arrays = load_multimodal_arrays(dataset_path)
    records = load_alignment_records(corpus_path)

    dataset_len = int(arrays["y_cls"].shape[0])
    corpus_len = len(records)
    if corpus_len != dataset_len:
        if corpus_len < dataset_len:
            print(f"WARNING: Alignment corpus is smaller than dataset (corpus={corpus_len}, dataset={dataset_len}). "
                  "Using first {corpus_len} samples from dataset to match corpus.")
            # Truncate arrays to match corpus size
            truncated_arrays = {}
            for key, value in arrays.items():
                if hasattr(value, 'shape') and len(value.shape) > 0:
                    truncated_arrays[key] = value[:corpus_len]
                else:
                    truncated_arrays[key] = value
            arrays = truncated_arrays
            dataset_len = corpus_len
        else:
            raise ValueError(
                "Alignment corpus is stale or inconsistent with the multimodal dataset: "
                f"corpus size={corpus_len}, dataset size={dataset_len}. "
                "Rebuild the corpus before training/evaluating stage1 diagnostics:\n"
                "PYTHONPATH=. python src/build_stage1_alignment_corpus.py"
            )

    train_idx_np, val_idx_np = stratified_split(
        arrays["y_cls"].astype("int64"),
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        seed=int(cfg.get("seed", 7)),
    )
    stats = compute_normalization_stats(arrays, train_idx_np)
    normalized = apply_normalization(arrays, stats)
    base_dataset = ServoMultimodalDataset(normalized, torch.arange(arrays["y_cls"].shape[0]).numpy())

    scenario_names = [str(x) for x in ckpt["scenario_names"]]
    family_names = [str(x) for x in ckpt["family_names"]]
    location_names = [str(x) for x in ckpt["location_names"]]

    if text_backbone == "qwen":
        encoder_path = Path(qwen_path) if qwen_path is not None else Path(cfg.get("qwen_path", ""))
        text_encoder = QwenFrozenTextEncoder(str(encoder_path), device=runtime_device)
    else:
        text_encoder = MockFrozenTextEncoder(hidden_size=int(ckpt.get("resolved_align_dim", 768)))

    prototype_bank = build_stage2_prototype_bank(
        text_encoder=text_encoder,
        scenario_names=scenario_names,
        family_names=family_names,
        location_names=location_names,
        batch_size=text_batch_size,
    )

    sample_batch = base_dataset[0]
    model = SignalAlignmentEncoder(
        batch_input_dims(sample_batch),
        model_dim=int(cfg.get("model_dim", 128)),
        align_dim=int(ckpt.get("resolved_align_dim", cfg.get("align_dim", 768))),
    ).to(runtime_device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return Stage1Runtime(
        model=model,
        text_encoder=text_encoder,
        prototype_bank=prototype_bank,
        arrays=arrays,
        normalized_arrays=normalized,
        base_dataset=base_dataset,
        records=records,
        train_idx=torch.as_tensor(train_idx_np, dtype=torch.long),
        val_idx=torch.as_tensor(val_idx_np, dtype=torch.long),
        device=runtime_device,
        checkpoint=ckpt,
    )
