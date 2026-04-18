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
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments_smoke_20260316.exp1_decoupled_models import Stage2DecoupledClassifier
from experiments_smoke_20260316.train_exp1_decoupled_stages import (
    DecoupledBatch,
    DecoupledDataset,
    STRICT_DECOUPLED_COLUMNS,
    apply_normalization,
    batch_input_dims,
    collate_decoupled_batch,
    compute_normalization_stats,
    load_decoupled_arrays,
    move_batch_to_device,
    stratified_split_three_way,
)
from servo_llm_alignment.dataset import load_alignment_records


SYSTEM_PROMPT = (
    "You are a servo fault diagnosis assistant. "
    "You must infer the fault from the provided sensor-signal prefix information."
)


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass(frozen=True)
class SignalDialogExample:
    index: int
    prompt: str
    answer: str
    scenario: str
    task: str


class SignalDialogDataset(Dataset):
    def __init__(self, base_dataset: DecoupledDataset, examples: list[SignalDialogExample]) -> None:
        self.base_dataset = base_dataset
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        item = self.examples[idx]
        return self.base_dataset[item.index], item


class TokenizedDialogDataset(Dataset):
    def __init__(self, base_dataset: DecoupledDataset, rows: list[dict[str, object]]) -> None:
        self.base_dataset = base_dataset
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        batch = self.base_dataset[int(row["index"])]
        return batch, row


def collate_tokenized_dialog(items):
    batches = [item[0] for item in items]
    rows = [item[1] for item in items]
    batch = collate_decoupled_batch(batches)

    def pad_tensor_list(key: str, pad_value: int) -> torch.Tensor:
        max_len = max(int(row[key].shape[0]) for row in rows)
        tensors = []
        for row in rows:
            tensor = row[key]
            pad = max_len - int(tensor.shape[0])
            tensors.append(F.pad(tensor, (0, pad), value=pad_value))
        return torch.stack(tensors, dim=0)

    meta = {
        "prompt_ids": pad_tensor_list("prompt_ids", 0),
        "prompt_mask": pad_tensor_list("prompt_mask", 0),
        "answer_ids": pad_tensor_list("answer_ids", 0),
        "answer_mask": pad_tensor_list("answer_mask", 0),
        "labels": pad_tensor_list("labels", -100),
        "examples": [row["example"] for row in rows],
    }
    return batch, meta


def build_examples(records: list, indices: np.ndarray, include_diagnosis: bool) -> list[SignalDialogExample]:
    examples: list[SignalDialogExample] = []
    for idx in indices.astype(np.int64).tolist():
        record = records[int(idx)]
        scenario = str(record.scenario).replace("_", " ").strip()
        family = str(record.family).replace("_", " ").strip()
        location = str(record.location).replace("_", " ").strip()
        boundary = str(record.boundary).replace("_", " ").strip()
        severity = float(record.severity)

        examples.append(
            SignalDialogExample(
                index=int(idx),
                prompt="Based only on the sensor signal prefix, what is the most likely fault scenario?",
                answer=scenario,
                scenario=scenario,
                task="scenario",
            )
        )
        if include_diagnosis:
            examples.append(
                SignalDialogExample(
                    index=int(idx),
                    prompt="Based only on the sensor signal prefix, give a short diagnosis including family, location, severity, and scenario.",
                    answer=(
                        f"fault family: {family}; location: {location}; "
                        f"boundary: {boundary}; severity: {severity:.2f}; scenario: {scenario}"
                    ),
                    scenario=scenario,
                    task="diagnosis",
                )
            )
    return examples


def build_chat_prompt(tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\nUser: {user_prompt}\nAssistant:"


def tokenize_examples(examples: list[SignalDialogExample], tokenizer, max_length: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for example in examples:
        prompt_text = build_chat_prompt(tokenizer, example.prompt)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(example.answer, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        if len(prompt_ids) + len(answer_ids) > max_length:
            answer_keep = max(1, max_length - len(prompt_ids))
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


class FrozenSignalFusionEncoder(nn.Module):
    def __init__(self, signal_init: Path, input_dims: dict[str, int]) -> None:
        super().__init__()
        payload = torch.load(signal_init, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        self.model = Stage2DecoupledClassifier(
            input_dims=input_dims,
            num_classes=17,
            model_dim=int(cfg["model_dim"]),
            token_dim=int(cfg["token_dim"]),
            num_layers=int(cfg["fusion_layers"]),
            nhead=int(cfg["fusion_heads"]),
            dim_feedforward=int(cfg["fusion_ff"]),
            pool=str(cfg["pool"]),
        )
        self.model.load_state_dict(payload["model"], strict=False)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.token_dim = int(cfg["token_dim"])

    @torch.no_grad()
    def forward(self, batch: DecoupledBatch) -> torch.Tensor:
        backbone_out = self.model.backbone(batch)
        if isinstance(backbone_out, tuple):
            tokens = backbone_out[0]
        else:
            tokens = backbone_out
        cls = self.model.cls.expand(tokens.shape[0], -1, -1)
        seq = torch.cat([cls, tokens], dim=1)
        seq = self.model.encoder(seq)
        seq = self.model.norm(seq)
        return seq[:, 1:, :]


class PrefixProjector(nn.Module):
    def __init__(self, signal_dim: int, llm_dim: int, prefix_length: int) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.proj = nn.Sequential(
            nn.LayerNorm(signal_dim),
            nn.Linear(signal_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        self.pool = nn.AdaptiveAvgPool1d(prefix_length)

    def forward(self, signal_tokens: torch.Tensor) -> torch.Tensor:
        x = self.proj(signal_tokens)
        x = self.pool(x.transpose(1, 2)).transpose(1, 2)
        return x


def build_inputs_embeds(model, prefix_embeds: torch.Tensor, prompt_ids: torch.Tensor, answer_ids: torch.Tensor):
    embed_tokens = model.get_input_embeddings()
    prompt_embeds = embed_tokens(prompt_ids)
    answer_embeds = embed_tokens(answer_ids)
    return torch.cat([prefix_embeds, prompt_embeds, answer_embeds], dim=1)


def build_attention_mask(prefix_embeds: torch.Tensor, prompt_mask: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    prefix_mask = torch.ones(prefix_embeds.shape[:2], dtype=prompt_mask.dtype, device=prompt_mask.device)
    return torch.cat([prefix_mask, prompt_mask, answer_mask], dim=1)


def build_labels(prefix_embeds: torch.Tensor, prompt_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    ignore = torch.full(
        (labels.shape[0], prefix_embeds.shape[1] + prompt_ids.shape[1]),
        -100,
        dtype=labels.dtype,
        device=labels.device,
    )
    return torch.cat([ignore, labels], dim=1)


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
            prefix_embeds = prefix_projector(signal_tokens)
            inputs_embeds = build_inputs_embeds(llm, prefix_embeds, prompt_ids, answer_ids)
            attention_mask = build_attention_mask(prefix_embeds, prompt_mask, answer_mask)
            full_labels = build_labels(prefix_embeds, prompt_ids, labels)
            out = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=full_labels)
            bs = int(prompt_ids.shape[0])
            loss_sum += float(out.loss.item()) * bs
            total += bs
    return loss_sum / max(1, total)


def evaluate_scenario_accuracy(signal_encoder, prefix_projector, llm, tokenizer, loader, device: torch.device, max_new_tokens: int):
    llm.eval()
    prefix_projector.eval()
    correct = 0
    total = 0
    rows: list[dict[str, str]] = []
    with torch.no_grad():
        for batch, meta in loader:
            batch = move_batch_to_device(batch, device)
            prompt_ids = meta["prompt_ids"].to(device)
            prompt_mask = meta["prompt_mask"].to(device)
            examples = meta["examples"]

            signal_tokens = signal_encoder(batch)
            prefix_embeds = prefix_projector(signal_tokens)
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
            gen_only = generated[:, attention_mask.shape[1] :]
            decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            for pred_text, example in zip(decoded, examples):
                if example.task != "scenario":
                    continue
                pred_text = pred_text.strip()
                pred_norm = normalize_text(pred_text)
                tgt_norm = normalize_text(example.answer)
                is_correct = pred_norm == tgt_norm or tgt_norm in pred_norm.split()
                correct += int(is_correct)
                total += 1
                rows.append(
                    {
                        "question": example.prompt,
                        "target": example.answer,
                        "prediction": pred_text,
                        "correct": "1" if is_correct else "0",
                    }
                )
    return correct / max(1, total), rows


def load_qwen_with_lora(model_path: Path, device: torch.device, lora_r: int, lora_alpha: int, lora_dropout: float):
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=dtype)
    model.config.use_cache = False
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--signal-init", type=Path, required=True)
    parser.add_argument("--qwen-path", type=Path, default=Path("/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--gen-max-new-tokens", type=int, default=16)
    parser.add_argument("--include-diagnosis", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    arrays, records = load_decoupled_arrays(args.dataset, args.corpus, int(args.max_samples))
    records = records or load_alignment_records(args.corpus)
    labels = arrays["y_cls"].astype(np.int64)
    train_idx, val_idx, test_idx = stratified_split_three_way(labels, float(args.val_ratio), float(args.test_ratio), int(args.seed))
    stats = compute_normalization_stats(arrays, train_idx)
    normalized = apply_normalization(arrays, stats)
    base_dataset = DecoupledDataset(normalized, np.arange(labels.shape[0], dtype=np.int64))

    train_examples = build_examples(records, train_idx, include_diagnosis=bool(args.include_diagnosis))
    val_examples = build_examples(records, val_idx, include_diagnosis=bool(args.include_diagnosis))
    test_examples = build_examples(records, test_idx, include_diagnosis=False)

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows = tokenize_examples(train_examples, tokenizer, max_length=int(args.max_length))
    val_rows = tokenize_examples(val_examples, tokenizer, max_length=int(args.max_length))
    test_rows = tokenize_examples(test_examples, tokenizer, max_length=int(args.max_length))

    train_loader = DataLoader(TokenizedDialogDataset(base_dataset, train_rows), batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_tokenized_dialog)
    val_loader = DataLoader(TokenizedDialogDataset(base_dataset, val_rows), batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_tokenized_dialog)
    test_loader = DataLoader(TokenizedDialogDataset(base_dataset, test_rows), batch_size=int(args.batch_size), shuffle=False, collate_fn=collate_tokenized_dialog)

    input_dims = batch_input_dims(base_dataset[0])
    signal_encoder = FrozenSignalFusionEncoder(args.signal_init, input_dims).to(device)
    llm = load_qwen_with_lora(args.qwen_path, device, int(args.lora_r), int(args.lora_alpha), float(args.lora_dropout))
    llm_hidden = int(llm.config.hidden_size)
    prefix_projector = PrefixProjector(signal_encoder.token_dim, llm_hidden, int(args.prefix_length)).to(device)

    params = [p for p in llm.parameters() if p.requires_grad] + list(prefix_projector.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = math.inf
    adapter_dir = args.output_dir / "adapter"
    projector_path = args.output_dir / "prefix_projector.pt"

    for epoch in range(1, int(args.epochs) + 1):
        llm.train()
        prefix_projector.train()
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
            prefix_embeds = prefix_projector(signal_tokens)
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

    test_acc, pred_rows = evaluate_scenario_accuracy(
        signal_encoder,
        eval_projector,
        tuned_llm,
        tokenizer,
        test_loader,
        device,
        max_new_tokens=int(args.gen_max_new_tokens),
    )

    pred_path = args.output_dir / "test_predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "target", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(pred_rows)

    report = {
        "model": "signal_prefix_qwen_lora",
        "signal_init": str(args.signal_init),
        "qwen_path": str(args.qwen_path),
        "prefix_length": int(args.prefix_length),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "test_scenario_questions": len(pred_rows),
        "best_val_loss": float(best_val),
        "test_scenario_accuracy": float(test_acc),
        "strict_modalities": STRICT_DECOUPLED_COLUMNS,
        "test_predictions_csv": str(pred_path),
        "history": history,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report={args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
