#file: retriever.py
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from .prototypes import Stage2PrototypeBank


@dataclass(frozen=True)
class RetrievalItem:
    rank: int
    label: str
    text: str
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    scenario: list[RetrievalItem]
    family: list[RetrievalItem]
    location: list[RetrievalItem]
    mechanism: list[RetrievalItem]


def _topk_items(embedding: Tensor, bank: Tensor, labels: list[str], texts: list[str], top_k: int) -> list[RetrievalItem]:
    scores = F.normalize(embedding, dim=-1) @ F.normalize(bank, dim=-1).T
    values, indices = torch.topk(scores, k=min(top_k, bank.shape[0]), dim=-1)
    results: list[RetrievalItem] = []
    for rank, (idx, value) in enumerate(zip(indices.tolist(), values.tolist()), start=1):
        results.append(RetrievalItem(rank=rank, label=labels[idx], text=texts[idx], score=float(value)))
    return results


def retrieve_topk(signal_embedding: Tensor, prototype_bank: Stage2PrototypeBank, top_k: int = 5) -> RetrievalResult:
    if signal_embedding.ndim != 1:
        raise ValueError("retrieve_topk expects a single embedding vector.")
    emb = signal_embedding.detach().cpu()
    return RetrievalResult(
        scenario=_topk_items(emb, prototype_bank.scenario_embeddings, prototype_bank.scenario_names, prototype_bank.scenario_texts, top_k),
        family=_topk_items(emb, prototype_bank.family_embeddings, prototype_bank.family_names, prototype_bank.family_texts, min(3, top_k)),
        location=_topk_items(emb, prototype_bank.location_embeddings, prototype_bank.location_names, prototype_bank.location_texts, min(3, top_k)),
        mechanism=_topk_items(emb, prototype_bank.mechanism_embeddings, prototype_bank.scenario_names, prototype_bank.mechanism_texts, top_k),
    )
