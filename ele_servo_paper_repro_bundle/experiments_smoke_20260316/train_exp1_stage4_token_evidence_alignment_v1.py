from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from servo_llm_alignment.text_encoder import MockFrozenTextEncoder, QwenFrozenTextEncoder
from experiments_smoke_20260316.train_exp1_decoupled_stages import (
    DecoupledDataset,
    STRICT_DECOUPLED_COLUMNS,
    apply_feature_mode,
    apply_normalization,
    batch_input_dims,
    collate_decoupled_batch,
    compute_normalization_stats,
    load_decoupled_arrays,
    move_batch_to_device,
    stratified_split_three_way,
)
from experiments_smoke_20260316.exp1_stage4_token_evidence_alignment_v1_models import Stage4TokenEvidenceAlignmentV1
from experiments_smoke_20260316.modality_evidence_topk_v5 import (
    PRIMARY_MODALITY_BY_SCENARIO,
    SUPPORT_MODALITIES_BY_SCENARIO,
    SoftPrototypeBank,
    batch_primary_support_masks,
    canonical_scenario_name,
    contrast_separation_loss,
    hard_negative_text_loss,
    listwise_topk_loss,
    save_summary_json,
    summarize_topk_rows,
    topk_modalities,
    weighted_consistency_at3,
)
from experiments_smoke_20260316.exp1_topk_duallevel_align_v2_models import info_nce
from experiments_smoke_20260316.token_evidence_alignment_v1_utils import (
    aggregate_token_mass_by_modality,
    save_json,
    summarize_token_alignment_rows,
    token_primary_support_loss,
)

BRANCH_NAMES = ["position", "electrical", "thermal", "vibration"]


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


def collate_with_records(items):
    batches = [item[0] for item in items]
    recs = [item[1] for item in items]
    return collate_decoupled_batch(batches), recs


def _extract_view_text(record, view_name: str) -> str:
    texts = getattr(record, "texts", {}) or {}
    if view_name == "combined":
        return str(texts.get("combined_text", "") or getattr(record, "text", ""))
    if view_name == "evidence":
        return str(texts.get("evidence_text", "") or texts.get("combined_text", ""))
    if view_name == "mechanism":
        return str(texts.get("mechanism_text", "") or texts.get("combined_text", ""))
    if view_name == "contrast":
        return str(texts.get("contrast_text", "") or texts.get("combined_text", ""))
    raise KeyError(view_name)


def build_multiview_text_cache(output_dir: Path, records: list, text_backbone: str, qwen_path: Path, device: torch.device, text_batch_size: int):
    if text_backbone == "qwen":
        text_encoder = QwenFrozenTextEncoder(str(qwen_path), device=device)
        text_dim = int(text_encoder.hidden_size)
    else:
        text_encoder = MockFrozenTextEncoder(hidden_size=768)
        text_dim = int(text_encoder.hidden_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"text_token_alignment_multiview__{text_backbone}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False), text_dim
    cache = {}
    for view_name in ["combined", "evidence", "mechanism", "contrast"]:
        texts = [_extract_view_text(record, view_name) for record in records]
        cache[view_name] = text_encoder.encode_texts(texts, batch_size=text_batch_size)
    torch.save(cache, cache_path)
    return cache, text_dim


def build_text_views_from_records(text_cache: dict[str, torch.Tensor], recs: list, device: torch.device) -> dict[str, torch.Tensor]:
    idx = torch.as_tensor([int(r.index) for r in recs], dtype=torch.long)
    return {key: value[idx].to(device) for key, value in text_cache.items()}


def load_init(model, init_path: Path) -> None:
    payload = torch.load(init_path, map_location="cpu", weights_only=False)
    state = payload.get("model") or payload.get("model_state") or payload.get("state_dict") or payload
    model.load_state_dict(state, strict=False)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_total_loss(outputs, labels: torch.Tensor, recs: list, class_names: list[str], prototype_bank: SoftPrototypeBank | None, lambda_cfg: dict[str, float]):
    cls_loss = F.cross_entropy(outputs.logits, labels)
    align_combined = info_nce(outputs.signal_embedding, outputs.combined_text_embedding, temperature=0.07)
    align_evidence = info_nce(outputs.signal_embedding, outputs.evidence_text_embedding, temperature=0.07)
    align_mechanism = info_nce(outputs.signal_embedding, outputs.mechanism_text_embedding, temperature=0.07)
    contrast_loss = contrast_separation_loss(outputs.signal_embedding, outputs.evidence_text_embedding, outputs.contrast_text_embedding)

    scenarios = [canonical_scenario_name(getattr(r, "scenario", "")) for r in recs]
    primary_mask, support_mask = batch_primary_support_masks(scenarios, outputs.logits.device)

    listwise = listwise_topk_loss(outputs.modality_scores, primary_mask, support_mask)
    token_loss, token_entropy = token_primary_support_loss(outputs.token_plan, primary_mask, support_mask, lambda_support=0.5, lambda_sparse=float(lambda_cfg["token_sparse"]))
    hard_neg = hard_negative_text_loss(outputs.signal_embedding, outputs.evidence_text_embedding, labels, class_names)

    proto_loss = outputs.logits.new_zeros(())
    if prototype_bank is not None:
        proto_loss = prototype_bank.prototype_alignment_loss(
            labels,
            outputs.signal_embedding,
            outputs.evidence_text_embedding,
            outputs.mechanism_text_embedding,
        )

    quality_loss = outputs.quality_loss if outputs.quality_loss is not None else outputs.logits.new_zeros(())

    total = (
        cls_loss
        + lambda_cfg["combined"] * align_combined
        + lambda_cfg["evidence"] * align_evidence
        + lambda_cfg["mechanism"] * align_mechanism
        + lambda_cfg["contrast"] * contrast_loss
        + lambda_cfg["listwise"] * listwise
        + lambda_cfg["token"] * token_loss
        + lambda_cfg["hardneg"] * hard_neg
        + lambda_cfg["proto"] * proto_loss
        + lambda_cfg["quality"] * quality_loss
    )
    parts = {
        "cls": float(cls_loss.item()),
        "combined": float(align_combined.item()),
        "evidence": float(align_evidence.item()),
        "mechanism": float(align_mechanism.item()),
        "contrast": float(contrast_loss.item()),
        "listwise": float(listwise.item()),
        "token": float(token_loss.item()),
        "token_entropy": float(token_entropy.item()),
        "hardneg": float(hard_neg.item()),
        "proto": float(proto_loss.item()),
        "quality": float(quality_loss.item()),
    }
    return total, parts


def evaluate_model(model, loader, device: torch.device, *, text_cache: dict[str, torch.Tensor], class_names: list[str], return_rows: bool = False):
    model.eval()
    total = 0
    correct = 0
    rows = []
    with torch.no_grad():
        for batch, recs in loader:
            text_views = build_text_views_from_records(text_cache, recs, device)
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, text_views)
            preds = outputs.logits.argmax(dim=1)
            correct += int((preds == batch.y_cls).sum().item())
            total += int(batch.y_cls.numel())
            if return_rows:
                top2 = topk_modalities(outputs.modality_scores, k=2)
                top3 = topk_modalities(outputs.modality_scores, k=3)
                mass = aggregate_token_mass_by_modality(outputs.token_plan).detach().cpu().numpy()  # [B,4,2]
                for i, rec in enumerate(recs):
                    scenario = canonical_scenario_name(getattr(rec, "scenario", ""))
                    primary = PRIMARY_MODALITY_BY_SCENARIO.get(scenario, "balanced")
                    support = SUPPORT_MODALITIES_BY_SCENARIO.get(scenario, [])
                    t2, t3 = top2[i], top3[i]
                    rows.append({
                        "sample_index": int(rec.index),
                        "scenario": scenario,
                        "primary": primary,
                        "support": ";".join(support),
                        "pred_top2": ";".join(t2),
                        "pred_top3": ";".join(t3),
                        "evidence_primary_in_top2": primary in t2 if primary != "balanced" else True,
                        "evidence_primary_in_top3": primary in t3 if primary != "balanced" else True,
                        "evidence_primary_or_support_at3": ((primary in t3) or any(x in t3 for x in support)) if primary != "balanced" else True,
                        "evidence_weighted_consistency_at3": weighted_consistency_at3(t3, primary, support) if primary != "balanced" else 1.0,
                        "mechanism_primary_in_top2": primary in t2 if primary != "balanced" else True,
                        "mechanism_primary_in_top3": primary in t3 if primary != "balanced" else True,
                        "mechanism_primary_or_support_at3": ((primary in t3) or any(x in t3 for x in support)) if primary != "balanced" else True,
                        "mechanism_weighted_consistency_at3": weighted_consistency_at3(t3, primary, support) if primary != "balanced" else 1.0,
                        "evidence_mass_position": float(mass[i, 0, 0]),
                        "evidence_mass_electrical": float(mass[i, 1, 0]),
                        "evidence_mass_thermal": float(mass[i, 2, 0]),
                        "evidence_mass_vibration": float(mass[i, 3, 0]),
                        "mechanism_mass_position": float(mass[i, 0, 1]),
                        "mechanism_mass_electrical": float(mass[i, 1, 1]),
                        "mechanism_mass_thermal": float(mass[i, 2, 1]),
                        "mechanism_mass_vibration": float(mass[i, 3, 1]),
                    })
    result = {"scenario_accuracy": correct / max(1, total)}
    if return_rows:
        result["rows"] = rows
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--text-backbone", choices=["qwen", "mock"], default="mock")
    parser.add_argument("--qwen-path", type=Path, default=Path("/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--feature-mode", choices=["base", "modality_tf"], default="modality_tf")
    parser.add_argument("--feature-chunk-size", type=int, default=2048)
    parser.add_argument("--quality-hidden-dim", type=int, default=128)
    parser.add_argument("--quality-drop-prob", type=float, default=0.10)
    parser.add_argument("--quality-min-gate", type=float, default=0.10)
    parser.add_argument("--fault-gate-hidden-dim", type=int, default=128)
    parser.add_argument("--llm-layers", type=int, default=4)
    parser.add_argument("--llm-heads", type=int, default=8)
    parser.add_argument("--llm-ff", type=int, default=768)
    parser.add_argument("--pool", choices=["cls", "mean", "attn"], default="attn")
    parser.add_argument("--lambda-combined", type=float, default=0.20)
    parser.add_argument("--lambda-evidence", type=float, default=0.35)
    parser.add_argument("--lambda-mechanism", type=float, default=0.35)
    parser.add_argument("--lambda-contrast", type=float, default=0.15)
    parser.add_argument("--lambda-listwise", type=float, default=0.20)
    parser.add_argument("--lambda-token", type=float, default=0.25)
    parser.add_argument("--lambda-token-sparse", type=float, default=0.05)
    parser.add_argument("--lambda-hardneg", type=float, default=0.20)
    parser.add_argument("--lambda-proto", type=float, default=0.12)
    parser.add_argument("--lambda-quality", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    arrays, records = load_decoupled_arrays(args.dataset, args.corpus, int(args.max_samples))
    arrays, feature_meta = apply_feature_mode(arrays, str(args.feature_mode), chunk_size=int(args.feature_chunk_size))
    labels = arrays["y_cls"].astype(np.int64)
    train_idx, val_idx, test_idx = stratified_split_three_way(labels, float(args.val_ratio), float(args.test_ratio), int(args.seed))
    stats = compute_normalization_stats(arrays, train_idx)
    normalized = apply_normalization(arrays, stats)

    base_dataset = DecoupledDataset(normalized, np.arange(labels.shape[0], dtype=np.int64))
    train_loader = DataLoader(StageTextDataset(base_dataset, records, train_idx), batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_with_records)
    val_loader = DataLoader(StageTextDataset(base_dataset, records, val_idx), batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_with_records)
    test_loader = DataLoader(StageTextDataset(base_dataset, records, test_idx), batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_with_records)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    text_cache, text_dim = build_multiview_text_cache(args.output_dir, records, str(args.text_backbone), args.qwen_path, device, int(args.text_batch_size))
    class_names = arrays["class_names"].astype(str).tolist()
    input_dims = batch_input_dims(base_dataset[0])

    model = Stage4TokenEvidenceAlignmentV1(
        input_dims,
        len(class_names),
        int(text_dim),
        model_dim=int(args.model_dim),
        token_dim=int(args.token_dim),
        num_layers=int(args.llm_layers),
        nhead=int(args.llm_heads),
        dim_feedforward=int(args.llm_ff),
        pool=str(args.pool),
        quality_hidden_dim=int(args.quality_hidden_dim),
        modality_drop_prob=float(args.quality_drop_prob),
        quality_min_gate=float(args.quality_min_gate),
        fault_gate_hidden_dim=int(args.fault_gate_hidden_dim),
    ).to(device)

    load_init(model, args.init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    prototype_bank = SoftPrototypeBank(len(class_names), int(args.token_dim), device=device, momentum=0.97)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    report_path = args.output_dir / "report.json"
    best_val = -math.inf
    history = []

    lambda_cfg = {
        "combined": float(args.lambda_combined),
        "evidence": float(args.lambda_evidence),
        "mechanism": float(args.lambda_mechanism),
        "contrast": float(args.lambda_contrast),
        "listwise": float(args.lambda_listwise),
        "token": float(args.lambda_token),
        "token_sparse": float(args.lambda_token_sparse),
        "hardneg": float(args.lambda_hardneg),
        "proto": float(args.lambda_proto),
        "quality": float(args.lambda_quality),
    }

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        part_sums = {k: 0.0 for k in ["cls", "combined", "evidence", "mechanism", "contrast", "listwise", "token", "token_entropy", "hardneg", "proto", "quality"]}
        for batch, recs in train_loader:
            text_views = build_text_views_from_records(text_cache, recs, device)
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, text_views)
            loss, parts = compute_total_loss(outputs, batch.y_cls, recs, class_names, prototype_bank, lambda_cfg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                prototype_bank.update(batch.y_cls, outputs.signal_embedding.detach(), outputs.evidence_text_embedding.detach(), outputs.mechanism_text_embedding.detach())
            preds = outputs.logits.argmax(dim=1)
            correct += int((preds == batch.y_cls).sum().item())
            total += int(batch.y_cls.numel())
            loss_sum += float(loss.item()) * int(batch.y_cls.numel())
            for k, v in parts.items():
                part_sums[k] += v * int(batch.y_cls.numel())

        val_metrics = evaluate_model(model, val_loader, device, text_cache=text_cache, class_names=class_names, return_rows=False)
        row = {
            "epoch": epoch,
            "loss": loss_sum / max(1, total),
            "scenario_accuracy": correct / max(1, total),
            "val_scenario_accuracy": float(val_metrics["scenario_accuracy"]),
        }
        for k, v in part_sums.items():
            row[k] = v / max(1, total)
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if float(val_metrics["scenario_accuracy"]) > best_val:
            best_val = float(val_metrics["scenario_accuracy"])
            torch.save({"model": model.state_dict(), "config": vars(args)}, best_path)

    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"], strict=False)
    test_metrics = evaluate_model(model, test_loader, device, text_cache=text_cache, class_names=class_names, return_rows=True)
    rows = test_metrics["rows"]

    scores_csv = args.output_dir / "test_token_alignment_scores.csv"
    summary_json = args.output_dir / "test_token_alignment_summary.json"
    legacy_summary_json = args.output_dir / "test_topk_consistency_summary.json"

    write_rows_csv(scores_csv, rows)
    summary = summarize_token_alignment_rows(rows)
    save_json(summary_json, summary)
    save_summary_json(legacy_summary_json, summarize_topk_rows(rows))

    report = {
        "experiment": "stage4_token_evidence_alignment_v1",
        "best_checkpoint": str(best_path),
        "best_val_scenario_accuracy": float(best_val),
        "test_scenario_accuracy": float(test_metrics["scenario_accuracy"]),
        "feature_mode": feature_meta,
        "strict_modalities": STRICT_DECOUPLED_COLUMNS,
        "token_alignment_scores_csv": str(scores_csv),
        "token_alignment_summary_json": str(summary_json),
        "topk_summary_json": str(legacy_summary_json),
        "history": history,
        "lambda_cfg": lambda_cfg,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
