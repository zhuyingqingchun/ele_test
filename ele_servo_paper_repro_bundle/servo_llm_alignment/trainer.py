from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .losses import info_nce, prototype_classification_loss


@dataclass
class TextPrototypeBank:
    scenario_embeddings: Tensor
    family_embeddings: Tensor
    location_embeddings: Tensor
    scenario_text_embeddings: Tensor


def build_batch_text_embeddings(records, prototype_bank: TextPrototypeBank, scenario_index, family_index, location_index, device: torch.device):
    scenario_targets = torch.tensor([scenario_index[r.scenario] for r in records], dtype=torch.long, device=device)
    family_targets = torch.tensor([family_index[r.family] for r in records], dtype=torch.long, device=device)
    location_targets = torch.tensor([location_index[r.location] for r in records], dtype=torch.long, device=device)

    # Ensure indices are on the same device as the prototype embeddings to avoid
    # "indices should be either on cpu or on the same device as the indexed tensor" errors.
    scenario_targets_for_lookup = scenario_targets.to(prototype_bank.scenario_text_embeddings.device)
    paired_text = prototype_bank.scenario_text_embeddings[scenario_targets_for_lookup]
    return scenario_targets, family_targets, location_targets, paired_text


def train_stage1(model, loader, optimizer, prototype_bank: TextPrototypeBank, scenario_index, family_index, location_index, device: torch.device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch, records in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        scenario_targets, family_targets, location_targets, paired_text = build_batch_text_embeddings(
            records, prototype_bank, scenario_index, family_index, location_index, device
        )
        paired_text = paired_text.to(device)

        loss_signal_text = info_nce(outputs.embedding, paired_text)
        loss_family = prototype_classification_loss(outputs.embedding, prototype_bank.family_embeddings.to(device), family_targets)
        loss_location = prototype_classification_loss(outputs.embedding, prototype_bank.location_embeddings.to(device), location_targets)
        loss_scenario = prototype_classification_loss(outputs.embedding, prototype_bank.scenario_embeddings.to(device), scenario_targets)
        loss = loss_signal_text + 0.35 * loss_family + 0.25 * loss_location + 0.45 * loss_scenario + 0.05 * outputs.alignment_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = int(batch.y_cls.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_samples += batch_size
    return {"loss": total_loss / max(total_samples, 1)}


@torch.no_grad()
def evaluate_stage1(model, loader, prototype_bank: TextPrototypeBank, scenario_index, family_index, location_index, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    for batch, records in loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        scenario_targets, family_targets, location_targets, paired_text = build_batch_text_embeddings(
            records, prototype_bank, scenario_index, family_index, location_index, device
        )
        paired_text = paired_text.to(device)
        loss_signal_text = info_nce(outputs.embedding, paired_text)
        loss_family = prototype_classification_loss(outputs.embedding, prototype_bank.family_embeddings.to(device), family_targets)
        loss_location = prototype_classification_loss(outputs.embedding, prototype_bank.location_embeddings.to(device), location_targets)
        loss_scenario = prototype_classification_loss(outputs.embedding, prototype_bank.scenario_embeddings.to(device), scenario_targets)
        loss = loss_signal_text + 0.35 * loss_family + 0.25 * loss_location + 0.45 * loss_scenario + 0.05 * outputs.alignment_loss
        logits = outputs.embedding @ prototype_bank.scenario_embeddings.to(device).T
        pred = logits.argmax(dim=1)
        correct += int((pred == scenario_targets).sum().item())
        batch_size = int(batch.y_cls.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_samples += batch_size
    return {"loss": total_loss / max(total_samples, 1), "scenario_accuracy": correct / max(total_samples, 1)}


def move_batch_to_device(batch, device: torch.device):
    return batch.__class__(**{field: getattr(batch, field).to(device) for field in batch.__dataclass_fields__})
