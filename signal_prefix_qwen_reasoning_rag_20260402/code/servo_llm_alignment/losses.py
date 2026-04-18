from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def info_nce(signal_emb: Tensor, text_emb: Tensor, temperature: float = 0.07) -> Tensor:
    logits = signal_emb @ text_emb.T / temperature
    targets = torch.arange(signal_emb.shape[0], device=signal_emb.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def prototype_classification_loss(signal_emb: Tensor, prototype_emb: Tensor, targets: Tensor, temperature: float = 0.07) -> Tensor:
    logits = signal_emb @ prototype_emb.T / temperature
    return F.cross_entropy(logits, targets)
