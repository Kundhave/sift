"""Cross-encoder with three heads (alignment, drift type, calibration/confidence)."""
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

DRIFT_CLASSES = ["collateral", "scope_expansion", "purpose_drift"]
DEFAULT_ENCODER = "answerdotai/ModernBERT-base"
STUDENT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class SiftModel(nn.Module):
    def __init__(self, encoder: str = DEFAULT_ENCODER, encoder_module: nn.Module | None = None, hidden: int | None = None,
                 dropout: float = 0.1, gradient_checkpointing: bool = False):
        super().__init__()
        self.encoder_name = encoder
        self.encoder = encoder_module if encoder_module is not None else AutoModel.from_pretrained(encoder)
        h = hidden or self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.align_head = nn.Linear(h, 1)
        self.drift_head = nn.Linear(h, len(DRIFT_CLASSES))
        self.cal_head = nn.Linear(h, 1)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def pooled(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "bert" in self.encoder.config.model_type and "modern" not in self.encoder.config.model_type:
            kw["token_type_ids"] = token_type_ids
        return self.encoder(**kw).last_hidden_state[:, 0]          # [CLS] representation

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        z = self.drop(self.pooled(input_ids, attention_mask, token_type_ids))
        return {"align_logit": self.align_head(z).squeeze(-1), "drift_logits": self.drift_head(z),
                "cal_logit": self.cal_head(z).squeeze(-1)}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def load_tokenizer(encoder: str):
    return AutoTokenizer.from_pretrained(encoder)
