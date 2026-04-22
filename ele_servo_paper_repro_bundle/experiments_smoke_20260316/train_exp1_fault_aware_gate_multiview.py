from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from servo_llm_alignment.text_encoder import MockFrozenTextEncoder, QwenFrozenTextEncoder
from servo_diagnostic.multimodal_method import FAMILY_BY_SCENARIO

from experiments_smoke_20260316.modality_evidence_priors import BRANCH_NAMES, MECHANICAL_RELATED, primary_index, support_mask
from experiments_smoke_20260316.exp1_fault_aware_gate_multiview_models import Stage3FaultAwareMultiViewClassifier, Stage4FaultAwareMultiViewClassifier
from experiments_smoke_20260316.train_exp1_decoupled_stages import (
    STRICT_DECOUPLED_COLUMNS,
    DecoupledDataset,
    StageTextDataset,
    apply_feature_mode,
    apply_normalization,
    batch_input_dims,
    collate_with_records,
    compute_normalization_stats,
    load_decoupled_arrays,
    move_batch_to_device,
    stratified_split_three_way,
)


def load_init(model, init_path: Path) -> None:
    payload = torch.load(init_path, map_location="cpu", weights_only=False)
    state = payload.get("model") or payload.get("model_state") or payload.get("state_dict") or payload
    model.load_state_dict(state, strict=False)


def build_multiview_text_cache(output_dir: Path, records: list, text_backbone: str, qwen_path: Path, device: torch.device, text_batch_size: int):
    if text_backbone == "qwen":
        text_encoder = QwenFrozenTextEncoder(str(qwen_path), device=device)
        text_dim = int(text_encoder.hidden_size)
    else:
        text_encoder = MockFrozenTextEncoder(hidden_size=768)
        text_dim = int(text_encoder.hidden_size)
    cache_path = output_dir / f"multiview_text_embeddings__{text_backbone}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False), text_dim
    views = {"combined": [], "evidence": [], "mechanism": [], "contrast": []}
    for record in records:
        texts = getattr(record, "texts", {}) or {}
        combined = texts.get("combined_text", "") or getattr(record, "text", "")
        views["combined"].append(combined)
        views["evidence"].append(texts.get("evidence_text", "") or combined)
        views["mechanism"].append(texts.get("mechanism_text", "") or combined)
        views["contrast"].append(texts.get("contrast_text", "") or combined)
    cache = {k: text_encoder.encode_texts(v, batch_size=text_batch_size) for k, v in views.items()}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache, text_dim


def _batch_targets(records) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    primary = []
    support = []
    scenarios = []
    for rec in records:
        scenario = str(getattr(rec, "scenario", "normal"))
        primary.append(primary_index(scenario))
        support.append(support_mask(scenario))
        scenarios.append(scenario)
    return torch.tensor(primary, dtype=torch.long), torch.tensor(support, dtype=torch.float32), scenarios


def compute_total_loss(outputs, y_cls: torch.Tensor, label_smoothing: float, lambda_align: float, lambda_quality: float) -> tuple[torch.Tensor, dict[str, float]]:
    cls_loss = F.cross_entropy(outputs.logits, y_cls, label_smoothing=label_smoothing)
    total = cls_loss
    parts = {"cls_loss": float(cls_loss.item()), "align_loss": 0.0, "quality_loss": 0.0}
    if outputs.align_loss is not None:
        total = total + lambda_align * outputs.align_loss
        parts["align_loss"] = float(outputs.align_loss.item())
    if outputs.quality_loss is not None:
        total = total + lambda_quality * outputs.quality_loss
        parts["quality_loss"] = float(outputs.quality_loss.item())
    return total, parts


def evaluate(model, loader, device: torch.device, *, text_cache: dict[str, torch.Tensor], label_smoothing: float, lambda_align: float, lambda_quality: float, loss_weights: dict[str, float], return_predictions: bool = False):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    align_sum = 0.0
    quality_sum = 0.0
    all_preds, all_targets, all_sample_indices = [], [], []
    all_quality_gates, all_branch_energies, all_modality_gates = [], [], []
    all_evidence_scores, all_mechanism_scores, all_primary, all_support, all_scenarios = [], [], [], [], []
    with torch.no_grad():
        for batch, recs in loader:
            idx = torch.as_tensor([int(r.index) for r in recs], dtype=torch.long)
            text_views = {k: v[idx].to(device) for k, v in text_cache.items()}
            primary_t, support_t, scenarios = _batch_targets(recs)
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, text_views=text_views, labels=batch.y_cls, primary_targets=primary_t.to(device), support_targets=support_t.to(device), weights=loss_weights)
            loss, parts = compute_total_loss(outputs, batch.y_cls, label_smoothing, lambda_align, lambda_quality)
            preds = outputs.logits.argmax(dim=1)
            bs = int(batch.y_cls.numel())
            correct += int((preds == batch.y_cls).sum().item())
            total += bs
            loss_sum += float(loss.item()) * bs
            align_sum += parts["align_loss"] * bs
            quality_sum += parts["quality_loss"] * bs
            if return_predictions:
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(batch.y_cls.cpu().numpy().tolist())
                all_sample_indices.extend(batch.sample_index.cpu().numpy().tolist())
                all_primary.extend(primary_t.numpy().tolist())
                all_support.extend(support_t.numpy().tolist())
                all_scenarios.extend(scenarios)
                if outputs.quality_gates is not None:
                    all_quality_gates.extend(outputs.quality_gates.cpu().numpy().tolist())
                if outputs.branch_energies is not None:
                    all_branch_energies.extend(outputs.branch_energies.cpu().numpy().tolist())
                if outputs.modality_gates is not None:
                    all_modality_gates.extend(outputs.modality_gates.cpu().numpy().tolist())
                if outputs.evidence_scores is not None:
                    all_evidence_scores.extend(outputs.evidence_scores.cpu().numpy().tolist())
                if outputs.mechanism_scores is not None:
                    all_mechanism_scores.extend(outputs.mechanism_scores.cpu().numpy().tolist())
    result = {"loss": loss_sum / max(1, total), "scenario_accuracy": correct / max(1, total), "align_loss": align_sum / max(1, total), "quality_loss": quality_sum / max(1, total)}
    if return_predictions:
        result.update({
            "predictions": all_preds, "targets": all_targets, "sample_indices": all_sample_indices,
            "quality_gates": all_quality_gates, "branch_energies": all_branch_energies, "modality_gates": all_modality_gates,
            "evidence_scores": all_evidence_scores, "mechanism_scores": all_mechanism_scores,
            "primary_targets": all_primary, "support_targets": all_support, "scenarios": all_scenarios,
        })
    return result


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def export_modality_evidence_analysis(output_dir: Path, test_results: dict, class_names: np.ndarray) -> None:
    sample_indices = test_results.get("sample_indices", [])
    predictions = test_results.get("predictions", [])
    targets = test_results.get("targets", [])
    evidence_scores = test_results.get("evidence_scores", [])
    mechanism_scores = test_results.get("mechanism_scores", [])
    modality_gates = test_results.get("modality_gates", [])
    primary_targets = test_results.get("primary_targets", [])
    support_targets = test_results.get("support_targets", [])
    scenarios = test_results.get("scenarios", [])
    csv_path = output_dir / "test_modality_evidence_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["sample_index", "scenario", "true_class", "pred_class", "true_class_name", "pred_class_name", "primary_target"]
        header += [f"support_{name}" for name in BRANCH_NAMES]
        header += [f"fault_gate_{name}" for name in BRANCH_NAMES]
        header += [f"evidence_score_{name}" for name in BRANCH_NAMES]
        header += [f"mechanism_score_{name}" for name in BRANCH_NAMES]
        writer.writerow(header)
        for i, idx in enumerate(sample_indices):
            row = [idx, scenarios[i] if i < len(scenarios) else "", targets[i], predictions[i], str(class_names[targets[i]]), str(class_names[predictions[i]]), BRANCH_NAMES[int(primary_targets[i])] if i < len(primary_targets) else ""]
            row += [f"{v:.1f}" for v in (support_targets[i] if i < len(support_targets) else [0.0] * len(BRANCH_NAMES))]
            row += [f"{v:.6f}" for v in (modality_gates[i] if i < len(modality_gates) else [0.0] * len(BRANCH_NAMES))]
            row += [f"{v:.6f}" for v in (evidence_scores[i] if i < len(evidence_scores) else [0.0] * len(BRANCH_NAMES))]
            row += [f"{v:.6f}" for v in (mechanism_scores[i] if i < len(mechanism_scores) else [0.0] * len(BRANCH_NAMES))]
            writer.writerow(row)
    per_class = {}
    evidence_hits, mechanism_hits, mech_evidence_hits, mech_mechanism_hits = [], [], [], []
    for i, target in enumerate(targets):
        scenario = scenarios[i] if i < len(scenarios) else str(class_names[target])
        name = str(class_names[target])
        evidence_top = int(np.argmax(evidence_scores[i])) if i < len(evidence_scores) else -1
        mechanism_top = int(np.argmax(mechanism_scores[i])) if i < len(mechanism_scores) else -1
        primary = int(primary_targets[i]) if i < len(primary_targets) else -1
        support = np.array(support_targets[i]) if i < len(support_targets) else np.zeros(len(BRANCH_NAMES), dtype=np.float32)
        evidence_hit = int(evidence_top == primary)
        mechanism_hit = int(mechanism_top >= 0 and support[mechanism_top] > 0.5)
        evidence_hits.append(evidence_hit)
        mechanism_hits.append(mechanism_hit)
        if scenario in MECHANICAL_RELATED:
            mech_evidence_hits.append(evidence_hit)
            mech_mechanism_hits.append(mechanism_hit)
        bucket = per_class.setdefault(name, {"sample_count": 0, "primary": BRANCH_NAMES[primary] if primary >= 0 else "unknown", "evidence_hit_rate": 0.0, "mechanism_primary_or_support_hit_rate": 0.0, "mean_fault_gate": [0.0] * len(BRANCH_NAMES)})
        bucket["sample_count"] += 1
        bucket["evidence_hit_rate"] += evidence_hit
        bucket["mechanism_primary_or_support_hit_rate"] += mechanism_hit
        if i < len(modality_gates):
            for j in range(len(BRANCH_NAMES)):
                bucket["mean_fault_gate"][j] += modality_gates[i][j]
    for bucket in per_class.values():
        count = max(1, int(bucket["sample_count"]))
        bucket["evidence_hit_rate"] = float(bucket["evidence_hit_rate"] / count)
        bucket["mechanism_primary_or_support_hit_rate"] = float(bucket["mechanism_primary_or_support_hit_rate"] / count)
        bucket["mean_fault_gate"] = [float(x / count) for x in bucket["mean_fault_gate"]]
    summary = {
        "branch_names": BRANCH_NAMES,
        "evidence_primary_hit_rate": _mean(evidence_hits),
        "mechanism_primary_or_support_hit_rate": _mean(mechanism_hits),
        "mechanical_related": {"sample_count": len(mech_evidence_hits), "evidence_primary_hit_rate": _mean(mech_evidence_hits), "mechanism_primary_or_support_hit_rate": _mean(mech_mechanism_hits)},
        "per_class": per_class,
    }
    (output_dir / "test_modality_evidence_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=[3, 4], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--lambda-align", type=float, default=0.15)
    parser.add_argument("--lambda-quality", type=float, default=0.10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--text-backbone", choices=["qwen", "mock"], default="mock")
    parser.add_argument("--qwen-path", type=Path, default=Path("/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--text-batch-size", type=int, default=8)
    parser.add_argument("--feature-mode", choices=["base", "modality_tf"], default="modality_tf")
    parser.add_argument("--feature-chunk-size", type=int, default=2048)
    parser.add_argument("--quality-aware-fusion", action="store_true")
    parser.add_argument("--quality-hidden-dim", type=int, default=128)
    parser.add_argument("--quality-drop-prob", type=float, default=0.10)
    parser.add_argument("--quality-min-gate", type=float, default=0.10)
    parser.add_argument("--fault-gate-hidden-dim", type=int, default=128)
    parser.add_argument("--fault-gate-min", type=float, default=0.15)
    parser.add_argument("--fusion-layers", type=int, default=4)
    parser.add_argument("--fusion-heads", type=int, default=8)
    parser.add_argument("--fusion-ff", type=int, default=768)
    parser.add_argument("--pool", choices=["text", "cls"], default="text")
    parser.add_argument("--lambda-combined", type=float, default=1.0)
    parser.add_argument("--lambda-evidence", type=float, default=1.0)
    parser.add_argument("--lambda-mechanism", type=float, default=1.0)
    parser.add_argument("--lambda-contrast", type=float, default=0.35)
    parser.add_argument("--lambda-modality-align", type=float, default=0.25)
    parser.add_argument("--lambda-proto", type=float, default=0.10)
    parser.add_argument("--lambda-hard-negative", type=float, default=0.10)
    parser.add_argument("--hard-negative-margin", type=float, default=0.15)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    arrays, records = load_decoupled_arrays(args.dataset, args.corpus, int(args.max_samples))
    arrays, feature_meta = apply_feature_mode(arrays, str(args.feature_mode), chunk_size=int(args.feature_chunk_size))
    y_cls = arrays["y_cls"].astype(np.int64)
    train_idx, val_idx, test_idx = stratified_split_three_way(y_cls, float(args.val_ratio), float(args.test_ratio), int(args.seed))
    stats = compute_normalization_stats(arrays, train_idx)
    normalized = apply_normalization(arrays, stats)
    base_dataset = DecoupledDataset(normalized, np.arange(y_cls.shape[0], dtype=np.int64))
    train_loader = DataLoader(StageTextDataset(base_dataset, records, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate_with_records)
    val_loader = DataLoader(StageTextDataset(base_dataset, records, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_records)
    test_loader = DataLoader(StageTextDataset(base_dataset, records, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=collate_with_records)
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    text_cache, text_dim = build_multiview_text_cache(args.output_dir, records, str(args.text_backbone), args.qwen_path, device, int(args.text_batch_size))
    input_dims = batch_input_dims(base_dataset[0])
    num_classes = int(len(arrays["class_names"]))
    common = dict(input_dims=input_dims, num_classes=num_classes, text_dim=text_dim, model_dim=int(args.model_dim), token_dim=int(args.token_dim), quality_aware_fusion=bool(args.quality_aware_fusion), quality_hidden_dim=int(args.quality_hidden_dim), modality_drop_prob=float(args.quality_drop_prob), quality_min_gate=float(args.quality_min_gate), fault_gate_hidden_dim=int(args.fault_gate_hidden_dim), fault_gate_min=float(args.fault_gate_min))
    if int(args.stage) == 3:
        model = Stage3FaultAwareMultiViewClassifier(**common).to(device)
    else:
        model = Stage4FaultAwareMultiViewClassifier(**common, num_layers=int(args.fusion_layers), nhead=int(args.fusion_heads), dim_feedforward=int(args.fusion_ff), pool=str(args.pool)).to(device)
    if args.init is not None:
        load_init(model, args.init)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr), weight_decay=float(args.weight_decay))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    report_path = args.output_dir / "report.json"
    loss_weights = {"lambda_combined": float(args.lambda_combined), "lambda_evidence": float(args.lambda_evidence), "lambda_mechanism": float(args.lambda_mechanism), "lambda_contrast": float(args.lambda_contrast), "lambda_modality_align": float(args.lambda_modality_align), "lambda_proto": float(args.lambda_proto), "lambda_hard_negative": float(args.lambda_hard_negative), "hard_negative_margin": float(args.hard_negative_margin)}
    best_acc = -math.inf
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = correct = 0
        loss_sum = align_sum = quality_sum = 0.0
        for batch, recs in train_loader:
            idx = torch.as_tensor([int(r.index) for r in recs], dtype=torch.long)
            text_views = {k: v[idx].to(device) for k, v in text_cache.items()}
            primary_t, support_t, _ = _batch_targets(recs)
            batch = move_batch_to_device(batch, device)
            outputs = model(batch, text_views=text_views, labels=batch.y_cls, primary_targets=primary_t.to(device), support_targets=support_t.to(device), weights=loss_weights)
            loss, parts = compute_total_loss(outputs, batch.y_cls, float(args.label_smoothing), float(args.lambda_align), float(args.lambda_quality))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds = outputs.logits.argmax(dim=1)
            bs = int(batch.y_cls.numel())
            correct += int((preds == batch.y_cls).sum().item())
            total += bs
            loss_sum += float(loss.item()) * bs
            align_sum += parts["align_loss"] * bs
            quality_sum += parts["quality_loss"] * bs
        val_metrics = evaluate(model, val_loader, device, text_cache=text_cache, label_smoothing=float(args.label_smoothing), lambda_align=float(args.lambda_align), lambda_quality=float(args.lambda_quality), loss_weights=loss_weights)
        row = {"epoch": epoch, "loss": loss_sum / max(1, total), "scenario_accuracy": correct / max(1, total), "align_loss": align_sum / max(1, total), "quality_loss": quality_sum / max(1, total), "val_loss": val_metrics["loss"], "val_scenario_accuracy": val_metrics["scenario_accuracy"], "val_align_loss": val_metrics["align_loss"], "val_quality_loss": val_metrics["quality_loss"]}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if val_metrics["scenario_accuracy"] > best_acc:
            best_acc = val_metrics["scenario_accuracy"]
            torch.save({"model": model.state_dict(), "config": vars(args)}, best_path)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    train_results = evaluate(model, train_loader, device, text_cache=text_cache, label_smoothing=float(args.label_smoothing), lambda_align=float(args.lambda_align), lambda_quality=float(args.lambda_quality), loss_weights=loss_weights, return_predictions=True)
    val_results = evaluate(model, val_loader, device, text_cache=text_cache, label_smoothing=float(args.label_smoothing), lambda_align=float(args.lambda_align), lambda_quality=float(args.lambda_quality), loss_weights=loss_weights, return_predictions=True)
    test_results = evaluate(model, test_loader, device, text_cache=text_cache, label_smoothing=float(args.label_smoothing), lambda_align=float(args.lambda_align), lambda_quality=float(args.lambda_quality), loss_weights=loss_weights, return_predictions=True)
    cm_output = {"train_confusion_matrix": confusion_matrix(train_results["targets"], train_results["predictions"]).tolist(), "val_confusion_matrix": confusion_matrix(val_results["targets"], val_results["predictions"]).tolist(), "test_confusion_matrix": confusion_matrix(test_results["targets"], test_results["predictions"]).tolist(), "labels": {str(i): str(name) for i, name in enumerate(arrays["class_names"].astype(str).tolist())}, "strict_modalities": STRICT_DECOUPLED_COLUMNS, "feature_mode": feature_meta, "fault_aware_gate": True, "multiview_alignment": True}
    (args.output_dir / "confusion_matrix.json").write_text(json.dumps(cm_output, ensure_ascii=False, indent=2), encoding="utf-8")
    export_modality_evidence_analysis(args.output_dir, test_results, arrays["class_names"])
    report = {"stage": int(args.stage), "best_checkpoint": str(best_path), "best_val_scenario_accuracy": float(best_acc), "train_scenario_accuracy": float(train_results["scenario_accuracy"]), "val_scenario_accuracy": float(val_results["scenario_accuracy"]), "test_scenario_accuracy": float(test_results["scenario_accuracy"]), "train_loss": float(train_results["loss"]), "val_loss": float(val_results["loss"]), "test_loss": float(test_results["loss"]), "train_quality_loss": float(train_results["quality_loss"]), "val_quality_loss": float(val_results["quality_loss"]), "test_quality_loss": float(test_results["quality_loss"]), "strict_modalities": STRICT_DECOUPLED_COLUMNS, "feature_mode": feature_meta, "fault_aware_gate": {"enabled": True, "fault_gate_hidden_dim": int(args.fault_gate_hidden_dim), "fault_gate_min": float(args.fault_gate_min)}, "multiview_alignment": {"views": ["combined", "evidence", "mechanism", "contrast"], "weights": {k: float(v) for k, v in loss_weights.items() if k.startswith("lambda_")}, "hard_negative_margin": float(args.hard_negative_margin)}, "history": history}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")

if __name__ == "__main__":
    main()
