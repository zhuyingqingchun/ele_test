from __future__ import annotations

import json
import re

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class QwenJsonReasoner(nn.Module):
    def __init__(self, model_path: str, device: torch.device, max_length: int = 256, dtype: str = "auto") -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = None
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch_dtype)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.hidden_size = int(self.model.config.hidden_size)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 8) -> Tensor:
        outputs: list[Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            model_out = self.model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = model_out.hidden_states[-1]
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            outputs.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0)

    def reason(
        self,
        context_text: str,
        candidate_scenarios: list[str] | None = None,
        candidate_mechanisms: list[str] | None = None,
        default_diagnosis: str = "",
        default_family: str = "",
        default_location: str = "",
        default_mechanism: str = "",
        default_confidence: float = 0.0,
        max_new_tokens: int = 320,
    ) -> dict:
        candidate_scenarios = candidate_scenarios or []
        candidate_mechanisms = candidate_mechanisms or []
        candidate_line = ", ".join(candidate_scenarios) if candidate_scenarios else ""
        mechanism_line = " | ".join(candidate_mechanisms) if candidate_mechanisms else ""
        prompt = (
            "You are a servo fault diagnosis assistant.\n"
            "Use only the provided retrieved evidence and event summary.\n"
            "Return strict JSON only. Do not add commentary.\n"
            "diagnosis must be one scenario label from the retrieved scenarios.\n"
            "family must be one family label string.\n"
            "location must be one location label string.\n"
            "mechanism must be one short string.\n"
            "confidence must be a number between 0 and 1.\n"
            "evidence, maintenance, alternatives must be arrays of strings.\n"
            f"Allowed diagnosis labels: [{candidate_line}].\n"
            f"Allowed mechanism statements: [{mechanism_line}].\n"
            "Use this exact schema:\n"
            "{\"diagnosis\":\"\",\"family\":\"\",\"location\":\"\",\"confidence\":0.0,\"evidence\":[],\"mechanism\":\"\",\"maintenance\":[],\"alternatives\":[]}\n\n"
            f"{context_text}\n\nJSON:"
        )
        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(generated[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        parsed = self._parse_json(text)
        constrained = self._apply_constraints(
            parsed=parsed,
            raw_text=text,
            candidate_scenarios=candidate_scenarios,
            candidate_mechanisms=candidate_mechanisms,
            default_diagnosis=default_diagnosis,
            default_family=default_family,
            default_location=default_location,
            default_mechanism=default_mechanism,
            default_confidence=default_confidence,
        )
        return {"raw_text": text, "parsed": constrained}

    @staticmethod
    def _parse_json(text: str) -> dict:
        data = None
        try:
            data = json.loads(text)
        except Exception:
            match = re.search(r"\{", text, flags=re.S)
            if match:
                try:
                    decoder = json.JSONDecoder()
                    data, _ = decoder.raw_decode(text[match.start() :])
                except Exception:
                    pass
        if data is None:
            return {
                "diagnosis": "",
                "family": "",
                "location": "",
                "confidence": 0.0,
                "evidence": [],
                "mechanism": "",
                "maintenance": [],
                "alternatives": [],
                "parse_failed": True,
            }
        return QwenJsonReasoner._normalize_schema(data)

    @staticmethod
    def _normalize_schema(data: dict) -> dict:
        def _as_string(value) -> str:
            if isinstance(value, list):
                return str(value[0]) if value else ""
            if value is None:
                return ""
            return str(value)

        def _as_string_list(value) -> list[str]:
            if value is None:
                return []
            if isinstance(value, list):
                return [str(item) for item in value]
            return [str(value)] if str(value) else []

        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            "diagnosis": _as_string(data.get("diagnosis", "")),
            "family": _as_string(data.get("family", "")),
            "location": _as_string(data.get("location", "")),
            "confidence": confidence,
            "evidence": _as_string_list(data.get("evidence", [])),
            "mechanism": _as_string(data.get("mechanism", "")),
            "maintenance": _as_string_list(data.get("maintenance", [])),
            "alternatives": _as_string_list(data.get("alternatives", [])),
        }

    @staticmethod
    def _apply_constraints(
        parsed: dict,
        raw_text: str,
        candidate_scenarios: list[str],
        candidate_mechanisms: list[str],
        default_diagnosis: str,
        default_family: str,
        default_location: str,
        default_mechanism: str,
        default_confidence: float,
    ) -> dict:
        out = dict(parsed)
        raw_diagnosis = out.get("diagnosis", "")
        raw_mechanism = out.get("mechanism", "")
        if candidate_scenarios:
            if out.get("diagnosis", "") not in candidate_scenarios:
                lowered = raw_text.lower()
                chosen = ""
                best_pos = None
                for label in candidate_scenarios:
                    pos = lowered.find(label.lower())
                    if pos >= 0 and (best_pos is None or pos < best_pos):
                        best_pos = pos
                        chosen = label
                out["diagnosis"] = chosen or default_diagnosis or candidate_scenarios[0]
                if out.get("confidence", 0.0) <= 0.0:
                    out["confidence"] = default_confidence
            # Final diagnosis is retrieval-anchored; the LLM provides explanation and alternatives.
            if default_diagnosis:
                out["diagnosis_llm_raw"] = raw_diagnosis
                if raw_diagnosis and raw_diagnosis != default_diagnosis and raw_diagnosis not in out.get("alternatives", []):
                    out["alternatives"] = [raw_diagnosis] + list(out.get("alternatives", []))
                out["diagnosis"] = default_diagnosis
        if candidate_mechanisms:
            mechanism_allowed = any(str(raw_mechanism).strip() == mech for mech in candidate_mechanisms)
            if not mechanism_allowed:
                lowered = raw_text.lower()
                chosen_mech = ""
                best_pos = None
                for mech in candidate_mechanisms:
                    pos = lowered.find(mech.lower())
                    if pos >= 0 and (best_pos is None or pos < best_pos):
                        best_pos = pos
                        chosen_mech = mech
                out["mechanism_llm_raw"] = raw_mechanism
                if raw_mechanism and raw_mechanism not in out.get("alternatives", []):
                    out["alternatives"] = [raw_mechanism] + list(out.get("alternatives", []))
                out["mechanism"] = chosen_mech or default_mechanism or candidate_mechanisms[0]
        if not out.get("family"):
            out["family"] = default_family
        if not out.get("location"):
            out["location"] = default_location
        if not out.get("mechanism"):
            out["mechanism"] = default_mechanism
        if out.get("confidence", 0.0) <= 0.0:
            out["confidence"] = default_confidence
        return out
