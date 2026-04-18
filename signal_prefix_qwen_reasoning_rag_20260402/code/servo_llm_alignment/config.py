from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_QWEN_PATH = Path("/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct")


@dataclass(frozen=True)
class Stage1AlignmentConfig:
    dataset_path: Path
    metadata_path: Path | None
    corpus_path: Path
    qwen_path: Path = DEFAULT_QWEN_PATH
    output_dir: Path = Path("trained_models/stage1_alignment")
    align_dim: int = 0
    model_dim: int = 128
    batch_size: int = 8
    epochs: int = 10
    lr: float = 2.0e-4
    weight_decay: float = 1.0e-4
    val_ratio: float = 0.15
    seed: int = 7
    device: str = "cuda"
