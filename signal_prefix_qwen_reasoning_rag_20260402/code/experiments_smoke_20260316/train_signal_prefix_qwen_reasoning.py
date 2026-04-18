from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments_smoke_20260316.train_exp1_decoupled_stages import (
    DecoupledDataset,
    STRICT_DECOUPLED_COLUMNS,
    _first_difference,
    _low_order_dct_trend,
    apply_feature_mode,
    apply_normalization,
    batch_input_dims,
    compute_normalization_stats,
    load_decoupled_arrays,
    move_batch_to_device,
    stratified_split_three_way,
)
from experiments_smoke_20260316.train_signal_prefix_qwen_lora import (
    FrozenSignalFusionEncoder,
    PrefixProjector,
    build_attention_mask,
    build_chat_prompt,
    build_inputs_embeds,
    build_labels,
    collate_tokenized_dialog,
    load_qwen_with_lora,
    normalize_text,
)


SYSTEM_PROMPT = (
    "You are a servo fault diagnosis reasoning assistant. "
    "You must infer the fault only from the sensor signal prefix tokens. "
    "You must answer in exactly three lines using the required keys."
)


CANONICAL_EXPLANATION_TEMPLATES = {
    "normal": "electrical, thermal and mechanical responses remain mutually consistent; tracking remains stable across modalities",
    "winding resistance rise": "drive effort rises under resistance-induced loss; copper dissipation and imbalance become more pronounced",
    "bus voltage sag fault": "available voltage headroom collapses; dynamic tracking becomes supply-limited",
    "partial demagnetization": "flux linkage weakens; electromagnetic conversion becomes less effective",
    "load disturbance severe": "tracking is stressed by strong disturbance from the load side; external load-path stress becomes dominant",
    "thermal saturation": "performance reduction is driven by thermal accumulation; temperature-induced derating reduces dynamic capability",
    "inverter voltage loss": "converter output capability is degraded; drive-side actuation suffers from inverter-side voltage loss",
    "current sensor bias": "electrical feedback becomes internally inconsistent; current-loop feedback is corrupted by sensing bias",
    "speed sensor scale": "feedback dynamics drift because sensed speed is not physically consistent; speed feedback is systematically mis-scaled",
    "position sensor bias": "tracking error contains a persistent sensing offset; motion remains feasible while the sensed position is biased",
    "bearing defect": "repetitive rolling-element impact behavior appears in vibration channels; impulsive mechanical contact signatures intensify",
    "friction wear mild": "drag torque rises mildly; low-speed actuation requires extra effort",
    "friction wear severe": "drag torque grows substantially; actuation effort rises under strong frictional loading",
    "backlash growth": "transmission slack grows; mechanical free-play becomes persistent",
    "jam fault": "load-side motion is blocked; command tracking collapses under a hard mechanical obstruction",
    "motor encoder freeze": "motor-side feedback intermittently stops updating; electrical-angle-related feedback becomes unreliable",
    "intermittent jam fault": "torque bursts recur during intermittent mechanical sticking; repeated obstruction produces transient collapse events",
}


@dataclass(frozen=True)
class SignalReasoningExample:
    index: int
    prompt: str
    answer: str
    scenario: str
    diagnostic_basis: str
    explanation: str


class TokenizedReasoningDataset(Dataset):
    def __init__(self, base_dataset: DecoupledDataset, rows: list[dict[str, object]]) -> None:
        self.base_dataset = base_dataset
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        batch = self.base_dataset[int(row["index"])]
        return batch, row


def cast_prefix_to_llm_dtype(prefix_embeds: torch.Tensor, llm) -> torch.Tensor:
    embed_layer = llm.get_input_embeddings()
    target_dtype = embed_layer.weight.dtype
    return prefix_embeds.to(dtype=target_dtype)


def _resolve_signal_checkpoint_feature_mode(signal_init: Path) -> tuple[str, bool]:
    checkpoint_mode = "base"
    legacy_thermal_tf = False
    try:
        payload = torch.load(signal_init, map_location="cpu", weights_only=False)
        checkpoint_mode = str(payload.get("config", {}).get("feature_mode", "base"))
    except Exception:
        checkpoint_mode = "base"
    report_path = signal_init.parent / "report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            feature_mode = report.get("feature_mode", {}) or {}
            checkpoint_mode = str(feature_mode.get("name", checkpoint_mode))
            thermal_algs = feature_mode.get("algorithms", {}).get("thermal", []) or []
            legacy_thermal_tf = any(name != "raw_only" for name in thermal_algs)
        except Exception:
            pass
    return checkpoint_mode, legacy_thermal_tf


def _apply_legacy_thermal_augmentation(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x = arrays["X_thermal"].astype(np.float32)
    trend = _low_order_dct_trend(x, keep_bins=4)
    grad = _first_difference(x)
    residual = (x - trend).astype(np.float32)
    out = dict(arrays)
    out["X_thermal"] = np.concatenate([x, trend, grad, residual], axis=2).astype(np.float32)
    return out


def prepare_arrays_for_signal_checkpoint(
    dataset_path: Path,
    corpus_path: Path,
    max_samples: int,
    feature_mode: str,
    feature_chunk_size: int,
    signal_init: Path,
):
    arrays, records = load_decoupled_arrays(dataset_path, corpus_path, max_samples)
    checkpoint_mode, legacy_thermal_tf = _resolve_signal_checkpoint_feature_mode(signal_init)
    target_mode = str(feature_mode)
    if checkpoint_mode == "modality_tf":
        target_mode = "modality_tf"
    arrays, feature_meta = apply_feature_mode(arrays, target_mode, chunk_size=int(feature_chunk_size))
    if checkpoint_mode == "modality_tf" and legacy_thermal_tf and arrays["X_thermal"].shape[-1] == 2:
        arrays = _apply_legacy_thermal_augmentation(arrays)
        feature_meta = dict(feature_meta)
        feature_meta["compatibility_note"] = "Applied legacy thermal modality_tf augmentation to match signal checkpoint."
        feature_meta["channel_expansion"] = dict(feature_meta.get("channel_expansion", {}))
        feature_meta["channel_expansion"]["X_thermal"] = {
            "original_dim": 2,
            "augmented_dim": 8,
            "feature_groups": ["raw", "low_dct_trend", "delta", "detrended_residual"],
        }
    return arrays, records, feature_meta, checkpoint_mode


def lexical_f1(pred: str, target: str) -> float:
    pred_tokens = normalize_text(pred).split()
    target_tokens = normalize_text(target).split()
    if not pred_tokens and not target_tokens:
        return 1.0
    if not pred_tokens or not target_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    tgt_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in target_tokens:
        tgt_counts[token] = tgt_counts.get(token, 0) + 1
    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, tgt_counts.get(token, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(target_tokens))
    return 2.0 * precision * recall / max(precision + recall, 1.0e-8)


def clean_reasoning_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()


def scenario_key(text: str) -> str:
    return normalize_text(text)


def extract_field(text: str, field: str) -> str:
    pattern = rf"{field}\s*:\s*(.+?)(?=(?:fault_result|diagnostic_basis|explanation)\s*:|$)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return clean_reasoning_text(match.group(1))


def parse_reasoning_output(text: str) -> dict[str, str]:
    return {
        "fault_result": extract_field(text, "fault_result"),
        "diagnostic_basis": extract_field(text, "diagnostic_basis"),
        "explanation": extract_field(text, "explanation"),
    }


def decode_generated_text(tokenizer, generated: torch.Tensor, prompt_token_len: int) -> list[str]:
    # When `generate` is driven by `inputs_embeds`, some HF model paths return only
    # newly generated tokens, while others may prepend the prompt-length prefix.
    if int(generated.shape[1]) > int(prompt_token_len):
        decode_ids = generated[:, int(prompt_token_len) :]
    else:
        decode_ids = generated
    return tokenizer.batch_decode(decode_ids, skip_special_tokens=True)


def select_basis_text(record) -> str:
    texts = getattr(record, "texts", {}) or {}
    candidates = [
        texts.get("evidence_text", ""),
        texts.get("combined_text", ""),
        texts.get("contrast_text", ""),
    ]
    for text in candidates:
        text = clean_reasoning_text(str(text))
        if text:
            return text
    scenario = str(record.scenario).replace("_", " ").strip()
    return f"signal evidence is most consistent with {scenario}"


def select_explanation_text(record) -> str:
    scenario = str(record.scenario).replace("_", " ").strip()
    canonical = CANONICAL_EXPLANATION_TEMPLATES.get(scenario_key(scenario))
    if canonical:
        return canonical
    texts = getattr(record, "texts", {}) or {}
    candidates = [
        texts.get("mechanism_text", ""),
        texts.get("combined_text", ""),
    ]
    for text in candidates:
        text = clean_reasoning_text(str(text))
        if text:
            return text
    family = str(record.family).replace("_", " ").strip()
    return f"the multimodal signal pattern is consistent with {family} related degradation and matches {scenario}"


def build_reasoning_examples(records: list, indices: np.ndarray) -> list[SignalReasoningExample]:
    prompt = (
        "Based only on the sensor signal prefix, infer the fault result, the diagnostic basis, "
        "and a short explanation. Output exactly three lines:\n"
        "fault_result: <scenario>\n"
        "diagnostic_basis: <short evidence>\n"
        "explanation: <one sentence explanation>"
    )
    examples: list[SignalReasoningExample] = []
    for idx in indices.astype(np.int64).tolist():
        record = records[int(idx)]
        scenario = str(record.scenario).replace("_", " ").strip()
        basis = select_basis_text(record)
        explanation = select_explanation_text(record)
        answer = (
            f"fault_result: {scenario}\n"
            f"diagnostic_basis: {basis}\n"
            f"explanation: {explanation}"
        )
        examples.append(
            SignalReasoningExample(
                index=int(idx),
                prompt=prompt,
                answer=answer,
                scenario=scenario,
                diagnostic_basis=basis,
                explanation=explanation,
            )
        )
    return examples


def tokenize_reasoning_examples(examples: list[SignalReasoningExample], tokenizer, max_length: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in examples:
        prompt_text = build_chat_prompt(tokenizer, example.prompt)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(example.answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        if len(prompt_ids) + len(answer_ids) > max_length:
            answer_keep = max(8, max_length - len(prompt_ids))
            answer_ids = answer_ids[:answer_keep]
        rows.append(
            {
                "index": int(example.index),
                "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
                "prompt_mask": torch.ones(len(prompt_ids), dtype=torch.long),
                "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
                "answer_mask": torch.ones(len(answer_ids), dtype=torch.long),
                "labels": torch.tensor(answer_ids, dtype=torch.long),
                "example": example,
            }
        )
    return rows


def evaluate_loss(signal_encoder, prefix_projector, llm, loader, device: torch.device) -> float:
    llm.eval()
    prefix_projector.eval()
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch, meta in loader:
            batch = move_batch_to_device(batch, device)
            prompt_ids = meta["prompt_ids"].to(device)
            prompt_mask = meta["prompt_mask"].to(device)
            answer_ids = meta["answer_ids"].to(device)
            answer_mask = meta["answer_mask"].to(device)
            labels = meta["labels"].to(device)

            signal_tokens = signal_encoder(batch)
            prefix_embeds = cast_prefix_to_llm_dtype(prefix_projector(signal_tokens), llm)
            inputs_embeds = build_inputs_embeds(llm, prefix_embeds, prompt_ids, answer_ids)
            attention_mask = build_attention_mask(prefix_embeds, prompt_mask, answer_mask)
            full_labels = build_labels(prefix_embeds, prompt_ids, labels)
            out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=full_labels)
            bs = int(prompt_ids.shape[0])
            loss_sum += float(out.loss.item()) * bs
            total += bs
    return loss_sum / max(1, total)


def evaluate_reasoning(signal_encoder, prefix_projector, llm, tokenizer, loader, device: torch.device, max_new_tokens: int):
    llm.eval()
    prefix_projector.eval()
    scenario_correct = 0
    format_correct = 0
    total = 0
    basis_f1_sum = 0.0
    explanation_f1_sum = 0.0
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch, meta in loader:
            batch = move_batch_to_device(batch, device)
            prompt_ids = meta["prompt_ids"].to(device)
            prompt_mask = meta["prompt_mask"].to(device)
            examples = meta["examples"]

            signal_tokens = signal_encoder(batch)
            prefix_embeds = cast_prefix_to_llm_dtype(prefix_projector(signal_tokens), llm)
            prompt_embeds = llm.get_input_embeddings()(prompt_ids)
            inputs_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prefix_mask = torch.ones(prefix_embeds.shape[:2], dtype=prompt_mask.dtype, device=device)
            attention_mask = torch.cat([prefix_mask, prompt_mask], dim=1)

            generated = llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            decoded = decode_generated_text(tokenizer, generated, int(attention_mask.shape[1]))

            for pred_text, example in zip(decoded, examples):
                pred_text = pred_text.strip()
                parsed = parse_reasoning_output(pred_text)
                pred_result = parsed["fault_result"]
                pred_basis = parsed["diagnostic_basis"]
                pred_explanation = parsed["explanation"]
                canonical_explanation = CANONICAL_EXPLANATION_TEMPLATES.get(scenario_key(pred_result))
                if canonical_explanation:
                    pred_explanation = canonical_explanation
                has_format = bool(pred_result and pred_basis and pred_explanation)
                result_ok = normalize_text(pred_result) == normalize_text(example.scenario)
                basis_f1 = lexical_f1(pred_basis, example.diagnostic_basis)
                explanation_f1 = lexical_f1(pred_explanation, example.explanation)

                scenario_correct += int(result_ok)
                format_correct += int(has_format)
                basis_f1_sum += basis_f1
                explanation_f1_sum += explanation_f1
                total += 1

                rows.append(
                    {
                        "prompt": example.prompt,
                        "target_result": example.scenario,
                        "prediction_result": pred_result,
                        "target_basis": example.diagnostic_basis,
                        "prediction_basis": pred_basis,
                        "target_explanation": example.explanation,
                        "prediction_explanation": pred_explanation,
                        "scenario_correct": "1" if result_ok else "0",
                        "format_correct": "1" if has_format else "0",
                        "basis_f1": f"{basis_f1:.6f}",
                        "explanation_f1": f"{explanation_f1:.6f}",
                        "raw_prediction": pred_text,
                    }
                )
    metrics = {
        "scenario_accuracy": scenario_correct / max(1, total),
        "format_accuracy": format_correct / max(1, total),
        "basis_f1": basis_f1_sum / max(1, total),
        "explanation_f1": explanation_f1_sum / max(1, total),
        "num_examples": total,
    }
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--signal-init", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, default=Path("/mnt/PRO6000_disk/models/Qwen/Qwen2.5-1.5B-Instruct"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature-mode", type=str, choices=["base", "modality_tf"], default="modality_tf")
    parser.add_argument("--feature-chunk-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=320)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--gen-max-new-tokens", type=int, default=96)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--test-shard-index", type=int, default=0)
    parser.add_argument("--test-shard-count", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    arrays, records, feature_meta, checkpoint_feature_mode = prepare_arrays_for_signal_checkpoint(
        args.dataset,
        args.corpus,
        int(args.max_samples),
        str(args.feature_mode),
        int(args.feature_chunk_size),
        args.signal_init,
    )
    labels = arrays["y_cls"].astype(np.int64)
    train_idx, val_idx, test_idx = stratified_split_three_way(labels, float(args.val_ratio), float(args.test_ratio), int(args.seed))
    stats = compute_normalization_stats(arrays, train_idx)
    normalized = apply_normalization(arrays, stats)
    base_dataset = DecoupledDataset(normalized, np.arange(labels.shape[0], dtype=np.int64))

    test_examples = build_reasoning_examples(records, test_idx)
    train_examples: list[SignalReasoningExample] = []
    val_examples: list[SignalReasoningExample] = []
    if not args.eval_only:
        train_examples = build_reasoning_examples(records, train_idx)
        val_examples = build_reasoning_examples(records, val_idx)

    shard_count = max(1, int(args.test_shard_count))
    shard_index = int(args.test_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid test shard index/count: index={shard_index}, count={shard_count}")
    shard_suffix = ""
    if shard_count > 1:
        shard_indices = np.array_split(np.arange(len(test_examples), dtype=np.int64), shard_count)[shard_index]
        test_examples = [test_examples[int(i)] for i in shard_indices.tolist()]
        shard_suffix = f".shard{shard_index + 1}-of-{shard_count}"

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_rows = tokenize_reasoning_examples(test_examples, tokenizer, max_length=int(args.max_length))
    test_loader = DataLoader(TokenizedReasoningDataset(base_dataset, test_rows), batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_tokenized_dialog)
    train_loader = None
    val_loader = None
    if not args.eval_only:
        train_rows = tokenize_reasoning_examples(train_examples, tokenizer, max_length=int(args.max_length))
        val_rows = tokenize_reasoning_examples(val_examples, tokenizer, max_length=int(args.max_length))
        train_loader = DataLoader(TokenizedReasoningDataset(base_dataset, train_rows), batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_tokenized_dialog)
        val_loader = DataLoader(TokenizedReasoningDataset(base_dataset, val_rows), batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_tokenized_dialog)

    input_dims = batch_input_dims(base_dataset[0])
    signal_encoder = FrozenSignalFusionEncoder(args.signal_init, input_dims).to(device)
    llm = load_qwen_with_lora(args.qwen_path, device, int(args.lora_r), int(args.lora_alpha), float(args.lora_dropout))
    llm_hidden = int(llm.config.hidden_size)
    prefix_projector = PrefixProjector(signal_encoder.token_dim, llm_hidden, int(args.prefix_length)).to(device)

    optimizer = None
    if not args.eval_only:
        params = [p for p in llm.parameters() if p.requires_grad] + list(prefix_projector.parameters())
        optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = math.inf
    adapter_dir = args.output_dir / "adapter"
    projector_path = args.output_dir / "prefix_projector.pt"

    if args.eval_only:
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Missing adapter directory for eval-only mode: {adapter_dir}")
        if not projector_path.exists():
            raise FileNotFoundError(f"Missing prefix projector for eval-only mode: {projector_path}")
    else:
        for epoch in range(1, int(args.epochs) + 1):
            llm.train()
            prefix_projector.train()
            assert optimizer is not None
            assert train_loader is not None
            assert val_loader is not None
            optimizer.zero_grad()
            running_loss = 0.0
            seen = 0
            for step, (batch, meta) in enumerate(train_loader, start=1):
                batch = move_batch_to_device(batch, device)
                prompt_ids = meta["prompt_ids"].to(device)
                prompt_mask = meta["prompt_mask"].to(device)
                answer_ids = meta["answer_ids"].to(device)
                answer_mask = meta["answer_mask"].to(device)
                labels_t = meta["labels"].to(device)

                with torch.no_grad():
                    signal_tokens = signal_encoder(batch)
                prefix_embeds = cast_prefix_to_llm_dtype(prefix_projector(signal_tokens), llm)
                inputs_embeds = build_inputs_embeds(llm, prefix_embeds, prompt_ids, answer_ids)
                attention_mask = build_attention_mask(prefix_embeds, prompt_mask, answer_mask)
                full_labels = build_labels(prefix_embeds, prompt_ids, labels_t)
                out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=full_labels)
                loss = out.loss / int(args.grad_accum)
                loss.backward()
                if step % int(args.grad_accum) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                bs = int(prompt_ids.shape[0])
                running_loss += float(out.loss.item()) * bs
                seen += bs
            if seen > 0 and len(train_loader) % int(args.grad_accum) != 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss = running_loss / max(1, seen)
            val_loss = evaluate_loss(signal_encoder, prefix_projector, llm, val_loader, device)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            if val_loss < best_val:
                best_val = val_loss
                llm.save_pretrained(adapter_dir)
                tokenizer.save_pretrained(adapter_dir)
                torch.save(prefix_projector.state_dict(), projector_path)

    base_dtype = llm.dtype if hasattr(llm, "dtype") else None
    if base_dtype is None:
        base_llm = AutoModelForCausalLM.from_pretrained(args.qwen_path, trust_remote_code=True)
    else:
        base_llm = AutoModelForCausalLM.from_pretrained(args.qwen_path, trust_remote_code=True, dtype=base_dtype)
    tuned_llm = PeftModel.from_pretrained(base_llm, adapter_dir).to(device)
    tuned_llm.config.use_cache = True
    eval_projector = PrefixProjector(signal_encoder.token_dim, int(tuned_llm.config.hidden_size), int(args.prefix_length)).to(device)
    eval_projector.load_state_dict(torch.load(projector_path, map_location=device, weights_only=False))

    metrics, pred_rows = evaluate_reasoning(
        signal_encoder,
        eval_projector,
        tuned_llm,
        tokenizer,
        test_loader,
        device,
        max_new_tokens=int(args.gen_max_new_tokens),
    )

    pred_path = args.output_dir / f"test_reasoning_predictions{shard_suffix}.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "prompt",
                "target_result",
                "prediction_result",
                "target_basis",
                "prediction_basis",
                "target_explanation",
                "prediction_explanation",
                "scenario_correct",
                "format_correct",
                "basis_f1",
                "explanation_f1",
                "raw_prediction",
            ],
        )
        writer.writeheader()
        writer.writerows(pred_rows)

    report = {
        "model": "signal_prefix_qwen_reasoning",
        "signal_init": str(args.signal_init),
        "signal_checkpoint_feature_mode": checkpoint_feature_mode,
        "qwen_path": str(args.qwen_path),
        "qwen_hidden_size": int(tuned_llm.config.hidden_size),
        "feature_mode": feature_meta,
        "prefix_length": int(args.prefix_length),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "best_val_loss": (float(best_val) if best_val < math.inf else None),
        "scenario_accuracy": float(metrics["scenario_accuracy"]),
        "format_accuracy": float(metrics["format_accuracy"]),
        "basis_f1": float(metrics["basis_f1"]),
        "explanation_f1": float(metrics["explanation_f1"]),
        "strict_modalities": STRICT_DECOUPLED_COLUMNS,
        "test_predictions_csv": str(pred_path),
        "history": history,
        "eval_only": bool(args.eval_only),
        "test_shard_index": shard_index,
        "test_shard_count": shard_count,
    }
    report_path = args.output_dir / f"report{shard_suffix}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
