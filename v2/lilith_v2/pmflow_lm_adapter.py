"""Optional adapter for PMFlow language-model capabilities in Lilith v2.

This module is intentionally lightweight and fully optional:
- If PMFlow LM modules are unavailable, callers can keep language_model=None.
- If model assets are missing or malformed, construction fails gracefully.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class PMFlowLanguageAdapter:
    """Small runtime wrapper around PMFlowLanguageModel with a JSON vocab."""

    TOKEN_RE = re.compile(r"[\w']+|[^\w\s]", re.UNICODE)

    def __init__(
        self,
        model: Any,
        token_to_id: Dict[str, int],
        id_to_token: Dict[int, str],
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.device = device
        self.unk_id = token_to_id.get("<unk>", 0)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        vocab_path: str,
        device: str = "cpu",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "PMFlowLanguageAdapter":
        from pmflow.lm.pmflow_lm import PMFlowLanguageModel

        vocab_obj = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        if "token_to_id" in vocab_obj and isinstance(vocab_obj["token_to_id"], dict):
            token_to_id = {str(k): int(v) for k, v in vocab_obj["token_to_id"].items()}
        elif isinstance(vocab_obj, dict):
            token_to_id = {str(k): int(v) for k, v in vocab_obj.items()}
        else:
            raise ValueError("Unsupported vocab format for PMFlow LM adapter")

        id_to_token = {v: k for k, v in token_to_id.items()}
        kwargs = dict(model_kwargs or {})
        kwargs["vocab_size"] = len(token_to_id)

        model = PMFlowLanguageModel(**kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        logger.info(
            "PMFlowLanguageAdapter loaded model from %s with vocab size %d",
            checkpoint_path,
            len(token_to_id),
        )
        return cls(model=model, token_to_id=token_to_id, id_to_token=id_to_token, device=device)

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in self.TOKEN_RE.findall(text)]

    def _encode(self, text: str) -> List[int]:
        toks = self._tokenize(text)
        if not toks:
            return [self.unk_id]
        return [self.token_to_id.get(t, self.unk_id) for t in toks]

    def score_text(self, text: str) -> Optional[float]:
        """Return mean next-token log-probability (higher is better)."""
        token_ids = self._encode(text)
        if len(token_ids) < 2:
            return None

        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model.forward_sequence(x)
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            targets = x[:, 1:].unsqueeze(-1)
            selected = torch.gather(log_probs, dim=-1, index=targets).squeeze(-1)
            return float(selected.mean().item())

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 24,
        temperature: float = 0.9,
        top_k: Optional[int] = 30,
        top_p: Optional[float] = 0.95,
    ) -> str:
        prompt_ids = self._encode(prompt)
        with torch.no_grad():
            out_ids = self.model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        new_ids = out_ids[len(prompt_ids):]
        new_tokens = [self.id_to_token.get(i, "<unk>") for i in new_ids]

        text = " ".join(new_tokens).strip()
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text
