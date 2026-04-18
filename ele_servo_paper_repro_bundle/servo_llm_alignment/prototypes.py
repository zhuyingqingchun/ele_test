from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .text_templates import build_text_views


@dataclass(frozen=True)
class Stage2PrototypeBank:
    scenario_names: list[str]
    family_names: list[str]
    location_names: list[str]
    scenario_texts: list[str]
    family_texts: list[str]
    location_texts: list[str]
    mechanism_texts: list[str]
    evidence_texts: list[str]
    contrast_texts: list[str]
    scenario_embeddings: Tensor
    family_embeddings: Tensor
    location_embeddings: Tensor
    mechanism_embeddings: Tensor
    evidence_embeddings: Tensor
    contrast_embeddings: Tensor


def build_stage2_prototype_bank(text_encoder, scenario_names: list[str], family_names: list[str], location_names: list[str], batch_size: int = 8) -> Stage2PrototypeBank:
    scenario_views = [build_text_views(name) for name in scenario_names]
    scenario_texts = [v["combined_text"] for v in scenario_views]
    mechanism_texts = [v["mechanism_text"] for v in scenario_views]
    evidence_texts = [v["evidence_text"] for v in scenario_views]
    contrast_texts = [v["contrast_text"] for v in scenario_views]
    family_texts = [f"Fault family: {name}." for name in family_names]
    location_texts = [f"Fault location: {name}." for name in location_names]
    return Stage2PrototypeBank(
        scenario_names=scenario_names,
        family_names=family_names,
        location_names=location_names,
        scenario_texts=scenario_texts,
        family_texts=family_texts,
        location_texts=location_texts,
        mechanism_texts=mechanism_texts,
        evidence_texts=evidence_texts,
        contrast_texts=contrast_texts,
        scenario_embeddings=text_encoder.encode_texts(scenario_texts, batch_size=batch_size),
        family_embeddings=text_encoder.encode_texts(family_texts, batch_size=batch_size),
        location_embeddings=text_encoder.encode_texts(location_texts, batch_size=batch_size),
        mechanism_embeddings=text_encoder.encode_texts(mechanism_texts, batch_size=batch_size),
        evidence_embeddings=text_encoder.encode_texts(evidence_texts, batch_size=batch_size),
        contrast_embeddings=text_encoder.encode_texts(contrast_texts, batch_size=batch_size),
    )
