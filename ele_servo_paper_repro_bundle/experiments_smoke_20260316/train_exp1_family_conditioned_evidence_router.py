from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from experiments_smoke_20260316.exp1_decoupled_models import Stage2DecoupledClassifier
from experiments_smoke_20260316.train_exp1_decoupled_stages import (
    DecoupledDataset,
    apply_feature_mode,
    apply_normalization,
    batch_input_dims,
    collate_decoupled_batch,
    compute_normalization_stats,
    load_decoupled_arrays,
    move_batch_to_device,
    stratified_split_three_way,
)

MODALITY_ORDER = ["pos", "electrical", "thermal", "vibration"]
TOKEN_COUNTS = {
    "pos": 20,
    "electrical": 24,
    "thermal": 8,
    "vibration": 12,
}

SCENARIO_MODALITY_EXPECTATIONS = {
    "load_disturbance_mild": ["pos", "electrical", "vibration"],
    "load_disturbance_severe": ["pos", "electrical", "vibration"],
    "friction_wear_mild": ["pos", "electrical", "vibration"],
    "friction_wear_severe": ["pos", "electrical", "vibration"],
    "jam_fault": ["pos", "vibration", "electrical"],
    "intermittent_jam_fault": ["vibration", "pos", "electrical"],
    "current_sensor_bias": ["electrical", "pos", "thermal"],
    "speed_sensor_scale": ["pos", "electrical", "vibration"],
    "position_sensor_bias": ["pos", "electrical", "vibration"],
    "winding_resistance_rise": ["electrical", "thermal", "pos"],
    "bus_voltage_sag_fault": ["electrical", "pos", "thermal"],
    "backlash_growth": ["pos", "vibration", "electrical"],
    "thermal_saturation": ["thermal", "electrical", "pos"],
    "motor_encoder_freeze": ["pos", "electrical", "vibration"],
    "partial_demagnetization": ["electrical", "pos", "thermal"],
    "inverter_voltage_loss": ["electrical", "vibration", "pos"],
    "bearing_defect": ["vibration", "electrical", "pos"],
}

RANK_TO_WEIGHT = [0.62, 0.23, 0.10, 0.05]


@dataclass
class RouterOutputs:
    scenario_logits: Tensor
    family_logits: Tensor
    routing_weights: Tensor
    quality_loss: Tensor | None


def _build_modality_slices() -> dict[str, tuple[int, int]]:
    start = 0
    out: dict[str, tuple[int, int]] = {}
    for name in MODALITY_ORDER:
        end = start + TOKEN_COUNTS[name]
        out[name] = (start, end)
        start = end
    return out


MODALITY_SLICES = _build_modality_slices()


def build_prior_matrix(class_names: list[str]) -> tuple[Tensor, Tensor]:
    priors = []
    mask = []
    for name in class_names:
        if name in SCENARIO_MODALITY_EXPECTATIONS:
            ranked = SCENARIO_MODALITY_EXPECTATIONS[name]
            weights = {m: RANK_TO_WEIGHT[-1] for m in MODALITY_ORDER}
            for idx, modality in enumerate(ranked[:3]):
                weights[modality] = RANK_TO_WEIGHT[idx]
            priors.append([float(weights[m]) for m in MODALITY_ORDER])
            mask.append(1.0)
        else:
            priors.append([0.25, 0.25, 0.25, 0.25])
            mask.append(0.0)
    return torch.tensor(priors, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


def build_family_mapping(records, class_names: list[str], family_names: list[str]) -> np.ndarray:
    scenario_to_family: dict[str, str] = {}
    for record in records:
        scenario_to_family[str(record.scenario)] = str(record.family)
    family_index = {name: idx for idx, name in enumerate(family_names)}
    return np.array([family_index[scenario_to_family[name]] for name in class_names], dtype=np.int64)


class FamilyConditionedEvidenceRouter(nn.Module):
    def __init__(self, base_model: Stage2DecoupledClassifier, num_classes: int, num_families: int, token_dim: int) -> None:
        super().__init__()
        self.base = base_model
        self.num_families = num_families
        self.token_dim = token_dim

        self.family_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(token_dim, num_families),
        )
        self.family_embeddings = nn.Embedding(num_families, token_dim)
        self.modality_embeddings = nn.Embedding(len(MODALITY_ORDER), token_dim)

        self.routing_head = nn.Sequential(
            nn.LayerNorm(token_dim * 3),
            nn.Linear(token_dim * 3, token_dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(token_dim, 1),
        )
        self.scenario_head = nn.Sequential(
            nn.LayerNorm(token_dim * 2),
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(token_dim, num_classes),
        )

    def forward(self, batch, family_ids: Tensor | None = None) -> RouterOutputs:
        tokens, _pooled, quality_loss = self.base.backbone(batch)
        cls = self.base.cls.expand(tokens.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = self.base.encoder(seq)
        seq = self.base.norm(seq)
        global_token = seq[:, 0, :]

        family_logits = self.family_head(global_token)
        if family_ids is not None:
            family_context = self.family_embeddings(family_ids)
        else:
            family_probs = torch.softmax(family_logits, dim=1)
            family_context = family_probs @ self.family_embeddings.weight

        modality_vectors = []
        for name in MODALITY_ORDER:
            start, end = MODALITY_SLICES[name]
            modality_tokens = seq[:, 1 + start : 1 + end, :]
            modality_vectors.append(modality_tokens.mean(dim=1))
        modality_stack = torch.stack(modality_vectors, dim=1)

        batch_size = modality_stack.shape[0]
        device = modality_stack.device
        modality_ids = torch.arange(len(MODALITY_ORDER), device=device)
        modality_context = self.modality_embeddings(modality_ids).unsqueeze(0).expand(batch_size, -1, -1)
        family_context_expanded = family_context.unsqueeze(1).expand(-1, len(MODALITY_ORDER), -1)
        global_context_expanded = global_token.unsqueeze(1).expand(-1, len(MODALITY_ORDER), -1)

        routing_inputs = torch.cat(
            [modality_stack + modality_context, family_context_expanded, global_context_expanded],
            dim=-1,
        )
        routing_logits = self.routing_head(routing_inputs).squeeze(-1)
        routing_weights = torch.softmax(routing_logits, dim=1)
        fused = torch.einsum("bm,bmd->bd", routing_weights, modality_stack)

        scenario_logits = self.scenario_head(torch.cat([fused, global_token], dim=-1))
        return RouterOutputs(
            scenario_logits=scenario_logits,
            family_logits=family_logits,
            routing_weights=routing_weights,
            quality_loss=quality_loss,
        )


def build_base_stage2(init_path: Path, input_dims: dict[str, int], num_classes: int) -> tuple[Stage2DecoupledClassifier, dict]:
    payload = torch.load(init_path, map_location="cpu", weights_only=False)
    cfg = dict(payload["config"])
    model = Stage2DecoupledClassifier(
        input_dims=input_dims,
        num_classes=num_classes,
        model_dim=int(cfg["model_dim"]),
        token_dim=int(cfg["token_dim"]),
        num_layers=int(cfg["fusion_layers"]),
        nhead=int(cfg["fusion_heads"]),
        dim_feedforward=int(cfg["fusion_ff"]),
        pool=str(cfg["pool"]),
        quality_aware_fusion=bool(cfg.get("quality_aware_fusion", False)),
        quality_hidden_dim=int(cfg.get("quality_hidden_dim", 128)),
        modality_drop_prob=float(cfg.get("quality_drop_prob", 0.0)),
        quality_min_gate=float(cfg.get("quality_min_gate", 0.10)),
    )
    model.load_state_dict(payload["model"], strict=False)
    return model, cfg


def modality_consistency_loss(
    routing_weights: Tensor,
    labels: Tensor,
    prior_matrix: Tensor,
    prior_mask: Tensor,
) -> Tensor:
    priors = prior_matrix[labels]
    mask = prior_mask[labels]
    if float(mask.sum().item()) < 0.5:
        return routing_weights.new_zeros(())
    per_sample = (priors * (torch.log(priors.clamp_min(1.0e-6)) - torch.log(routing_weights.clamp_min(1.0e-6)))).sum(dim=1)
    return (per_sample * mask).sum() / mask.sum().clamp_min(1.0)


def evaluate(
    model: FamilyConditionedEvidenceRouter,
    loader: DataLoader,
    device: torch.device,
    prior_matrix: Tensor,
    prior_mask: Tensor,
    class_to_family: Tensor,
    class_names: list[str],
) -> dict[str, object]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    rep_buckets: dict[str, list[np.ndarray]] = {name: [] for name in SCENARIO_MODALITY_EXPECTATIONS}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            family_ids = class_to_family[batch.y_cls]
            outputs = model(batch, family_ids=family_ids)

            cls_loss = F.cross_entropy(outputs.scenario_logits, batch.y_cls)
            fam_loss = F.cross_entropy(outputs.family_logits, family_ids)
            mc_loss = modality_consistency_loss(outputs.routing_weights, batch.y_cls, prior_matrix, prior_mask)
            quality_loss = outputs.quality_loss if outputs.quality_loss is not None else outputs.scenario_logits.new_zeros(())
            loss = cls_loss + 0.30 * fam_loss + 0.08 * mc_loss + 0.05 * quality_loss

            preds = outputs.scenario_logits.argmax(dim=1)
            total += int(batch.y_cls.shape[0])
            correct += int((preds == batch.y_cls).sum().item())
            loss_sum += float(loss.item()) * int(batch.y_cls.shape[0])

            weights_np = outputs.routing_weights.detach().cpu().numpy()
            labels_np = batch.y_cls.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            for w, y_true, y_pred in zip(weights_np, labels_np, preds_np):
                scenario_name = class_names[int(y_true)]
                if scenario_name in rep_buckets and int(y_true) == int(y_pred):
                    rep_buckets[scenario_name].append(w.astype(np.float32))

    details = []
    top1_hits = 0
    top2_hits = 0
    top3_hits = 0
    valid = 0
    for scenario_name, items in rep_buckets.items():
        if not items:
            continue
        valid += 1
        mean_weights = np.mean(np.stack(items, axis=0), axis=0)
        top_order = np.argsort(mean_weights)[::-1]
        expected_rank = SCENARIO_MODALITY_EXPECTATIONS[scenario_name]
        expected = expected_rank[0]
        observed_top1 = MODALITY_ORDER[int(top_order[0])]
        observed_top2 = [MODALITY_ORDER[int(i)] for i in top_order[:2].tolist()]
        observed_top3 = [MODALITY_ORDER[int(i)] for i in top_order[:3].tolist()]
        top1_ok = observed_top1 == expected
        top2_ok = expected in observed_top2
        top3_ok = expected in observed_top3
        top1_hits += int(top1_ok)
        top2_hits += int(top2_ok)
        top3_hits += int(top3_ok)
        details.append(
            {
                "scenario": scenario_name,
                "num_correct_samples": len(items),
                "expected_top1": expected_rank[0],
                "expected_top2": expected_rank[1],
                "expected_top3": expected_rank[2],
                "top1_modality": observed_top1,
                "top2_modalities": observed_top2,
                "top3_modalities": observed_top3,
                "top1_consistent": bool(top1_ok),
                "top2_consistent": bool(top2_ok),
                "top3_consistent": bool(top3_ok),
                "weights": {
                    name: float(mean_weights[idx]) for idx, name in enumerate(MODALITY_ORDER)
                },
            }
        )

    return {
        "category_accuracy": correct / max(1, total),
        "loss": loss_sum / max(1, total),
        "evidence_consistency_top1": top1_hits / max(1, valid),
        "evidence_consistency_top2": top2_hits / max(1, valid),
        "evidence_consistency_top3": top3_hits / max(1, valid),
        "all_fault_details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("derived_datasets/servo_multimodal_handoff_dataset.npz"))
    parser.add_argument("--corpus", type=Path, default=Path("derived_datasets/stage1_alignment_corpus.jsonl"))
    parser.add_argument("--base-stage2", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-mode", choices=["base", "modality_tf"], default="modality_tf")
    parser.add_argument("--feature-chunk-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    arrays, records = load_decoupled_arrays(args.dataset, args.corpus, 0)
    if records is None:
        raise SystemExit("records are required")
    arrays, feature_meta = apply_feature_mode(arrays, str(args.feature_mode), chunk_size=int(args.feature_chunk_size))
    labels = arrays["y_cls"].astype(np.int64)

    train_idx, val_idx, test_idx = stratified_split_three_way(labels, float(args.val_ratio), float(args.test_ratio), int(args.seed))
    stats = compute_normalization_stats(arrays, train_idx)
    normalized = apply_normalization(arrays, stats)
    full_indices = np.arange(labels.shape[0], dtype=np.int64)
    base_dataset = DecoupledDataset(normalized, full_indices)

    train_loader = DataLoader(DecoupledDataset(normalized, train_idx), batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_decoupled_batch)
    val_loader = DataLoader(DecoupledDataset(normalized, val_idx), batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_decoupled_batch)
    test_loader = DataLoader(DecoupledDataset(normalized, test_idx), batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_decoupled_batch)

    class_names = arrays["class_names"].astype(str).tolist()
    family_names = sorted({str(record.family) for record in records})
    class_to_family_np = build_family_mapping(records, class_names, family_names)
    class_to_family = torch.tensor(class_to_family_np, dtype=torch.long)
    prior_matrix, prior_mask = build_prior_matrix(class_names)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    base_stage2, cfg = build_base_stage2(args.base_stage2, batch_input_dims(base_dataset[0]), num_classes=len(class_names))
    model = FamilyConditionedEvidenceRouter(
        base_model=base_stage2,
        num_classes=len(class_names),
        num_families=len(family_names),
        token_dim=int(cfg["token_dim"]),
    ).to(device)

    prior_matrix = prior_matrix.to(device)
    prior_mask = prior_mask.to(device)
    class_to_family = class_to_family.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    report_path = args.output_dir / "report.json"
    report_md_path = args.output_dir / "report.md"

    best_val = -math.inf
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            family_ids = class_to_family[batch.y_cls]
            outputs = model(batch, family_ids=family_ids)
            cls_loss = F.cross_entropy(outputs.scenario_logits, batch.y_cls)
            fam_loss = F.cross_entropy(outputs.family_logits, family_ids)
            mc_loss = modality_consistency_loss(outputs.routing_weights, batch.y_cls, prior_matrix, prior_mask)
            quality_loss = outputs.quality_loss if outputs.quality_loss is not None else outputs.scenario_logits.new_zeros(())
            loss = cls_loss + 0.30 * fam_loss + 0.08 * mc_loss + 0.05 * quality_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()

            preds = outputs.scenario_logits.argmax(dim=1)
            total += int(batch.y_cls.shape[0])
            correct += int((preds == batch.y_cls).sum().item())
            loss_sum += float(loss.item()) * int(batch.y_cls.shape[0])

        train_acc = correct / max(1, total)
        val_metrics = evaluate(
            model, val_loader, device, prior_matrix, prior_mask, class_to_family, class_names
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, total),
            "train_category_accuracy": train_acc,
            "val_category_accuracy": float(val_metrics["category_accuracy"]),
            "val_evidence_consistency_top1": float(val_metrics["evidence_consistency_top1"]),
            "val_evidence_consistency_top2": float(val_metrics["evidence_consistency_top2"]),
            "val_evidence_consistency_top3": float(val_metrics["evidence_consistency_top3"]),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        score = (
            float(val_metrics["category_accuracy"])
            + 0.02 * float(val_metrics["evidence_consistency_top1"])
            + 0.01 * float(val_metrics["evidence_consistency_top2"])
        )
        if score > best_val:
            best_val = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(args),
                    "class_names": class_names,
                    "family_names": family_names,
                    "feature_mode": feature_meta,
                },
                best_path,
            )

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    metrics = evaluate(
        model, test_loader, device, prior_matrix, prior_mask, class_to_family, class_names
    )
    report = {
        "model": "family_conditioned_evidence_router",
        "category_accuracy": float(metrics["category_accuracy"]),
        "loss": float(metrics["loss"]),
        "evidence_consistency_top1": float(metrics["evidence_consistency_top1"]),
        "evidence_consistency_top2": float(metrics["evidence_consistency_top2"]),
        "evidence_consistency_top3": float(metrics["evidence_consistency_top3"]),
        "all_fault_details": metrics["all_fault_details"],
        "history": history,
        "feature_mode": feature_meta,
        "base_stage2": str(args.base_stage2),
        "best_checkpoint": str(best_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    report_csv_path = args.output_dir / "all_fault_evidence_consistency.csv"
    with report_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "num_correct_samples",
                "expected_top1",
                "expected_top2",
                "expected_top3",
                "top1_modality",
                "top2_modalities",
                "top3_modalities",
                "top1_consistent",
                "top2_consistent",
                "top3_consistent",
                "weight_pos",
                "weight_electrical",
                "weight_thermal",
                "weight_vibration",
            ],
        )
        writer.writeheader()
        for item in report["all_fault_details"]:
            writer.writerow(
                {
                    "scenario": item["scenario"],
                    "num_correct_samples": item["num_correct_samples"],
                    "expected_top1": item["expected_top1"],
                    "expected_top2": item["expected_top2"],
                    "expected_top3": item["expected_top3"],
                    "top1_modality": item["top1_modality"],
                    "top2_modalities": ", ".join(item["top2_modalities"]),
                    "top3_modalities": ", ".join(item["top3_modalities"]),
                    "top1_consistent": item["top1_consistent"],
                    "top2_consistent": item["top2_consistent"],
                    "top3_consistent": item["top3_consistent"],
                    "weight_pos": item["weights"]["pos"],
                    "weight_electrical": item["weights"]["electrical"],
                    "weight_thermal": item["weights"]["thermal"],
                    "weight_vibration": item["weights"]["vibration"],
                }
            )

    lines = [
        "# Family-Conditioned Evidence Router",
        "",
        f"- 类别准确率: {report['category_accuracy'] * 100:.2f}%",
        f"- 证据一致性 top-1: {report['evidence_consistency_top1'] * 100:.2f}%",
        f"- 证据一致性 top-2: {report['evidence_consistency_top2'] * 100:.2f}%",
        f"- 证据一致性 top-3: {report['evidence_consistency_top3'] * 100:.2f}%",
        "",
        "| 故障场景 | 期望 top-1 | 期望 top-2 | 期望 top-3 | 观察 top-1 | 观察 top-2 | 观察 top-3 | 一致性 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in report["all_fault_details"]:
        lines.append(
            f"| {item['scenario']} | {item['expected_top1']} | {item['expected_top2']} | {item['expected_top3']} | "
            f"{item['top1_modality']} | {', '.join(item['top2_modalities'])} | {', '.join(item['top3_modalities'])} | "
            f"top1={item['top1_consistent']}, top2={item['top2_consistent']}, top3={item['top3_consistent']} |"
        )
    report_md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
