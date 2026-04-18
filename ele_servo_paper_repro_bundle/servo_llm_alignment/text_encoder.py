from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class QwenFrozenTextEncoder(nn.Module):
    def __init__(self, model_path: str, device: torch.device, max_length: int = 192, dtype: str = "auto") -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        torch_dtype = None
        if dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "float32":
            torch_dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch_dtype)
        self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device
        self.max_length = max_length
        self.hidden_size = int(self.model.config.hidden_size)

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
            model_out = self.model(**encoded)
            hidden = model_out.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            outputs.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(outputs, dim=0)


class MockFrozenTextEncoder(nn.Module):
    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 8) -> Tensor:
        del batch_size
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little", signed=False)
            generator = torch.Generator().manual_seed(seed)
            rows.append(F.normalize(torch.randn(self.hidden_size, generator=generator), dim=0))
        return torch.stack(rows, dim=0)
