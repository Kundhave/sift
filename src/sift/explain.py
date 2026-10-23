"""Token-level attribution for the cross-encoder: Integrated Gradients (captum) and Input x Gradient.

Attribution target: the misalignment probability  p = sigmoid(align_logit).
Baseline: embeddings of [PAD] at every non-special position (CLS/SEP kept), so the path moves from
"content removed" to the actual input.  Token score = sum over embedding dims of the attribution.
"""
from __future__ import annotations
import re
import numpy as np
import torch
from captum.attr import IntegratedGradients, InputXGradient


class Explainer:
    def __init__(self, model, tok, device="cuda", max_len=512):
        self.model, self.tok, self.device, self.max_len = model.eval(), tok, device, max_len
        self.emb = model.encoder.get_input_embeddings()

    def encode(self, intent, hunk):
        enc = self.tok(intent, hunk, truncation="longest_first", max_length=self.max_len, return_offsets_mapping=True,
                       return_tensors="pt")
        seq = enc.sequence_ids(0)
        return enc, seq

    def _fwd(self, embeds, mask):
        o = self.model.encoder(inputs_embeds=embeds, attention_mask=mask).last_hidden_state[:, 0]
        return torch.sigmoid(self.model.align_head(o).squeeze(-1).float())

    def prob(self, ids, mask):
        with torch.no_grad():
            return float(self._fwd(self.emb(ids), mask))

    def attribute(self, intent, hunk, method="ig", n_steps=24, internal_batch_size=8):
        enc, seq = self.encode(intent, hunk)
        ids, mask = enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        e = self.emb(ids).detach()
        special = torch.tensor([s is None for s in seq], device=self.device)
        pad = self.emb(torch.full_like(ids, self.tok.pad_token_id)).detach()
        base = torch.where(special[None, :, None], e, pad)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            if method == "ig":
                a = IntegratedGradients(self._fwd).attribute(e, baselines=base, additional_forward_args=(mask,),
                                                             n_steps=n_steps, internal_batch_size=internal_batch_size)
            elif method == "ixg":
                a = InputXGradient(self._fwd).attribute(e, additional_forward_args=(mask,))
            else:
                raise ValueError(method)
        tok_attr = a.detach().float().sum(-1)[0].cpu().numpy()
        toks = self.tok.convert_ids_to_tokens(ids[0].cpu())
        return {"ids": ids[0].cpu(), "mask": mask, "tokens": toks, "attr": tok_attr, "seq": seq,
                "offsets": enc["offset_mapping"][0].tolist(), "prob": self.prob(ids, mask)}

    # ----- aggregation -----------------------------------------------------------------
    @staticmethod
    def to_words(res, intent, hunk, segment):
        """Aggregate sub-token attributions to whitespace/identifier words for segment 0 (intent) or 1 (hunk)."""
        text = intent if segment == 0 else hunk
        words = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\w+|[^\w\s]", text)]
        score = np.zeros(len(words))
        for (s, e), sid, a in zip(res["offsets"], res["seq"], res["attr"]):
            if sid != segment or e <= s:
                continue
            for k, (ws, we, _) in enumerate(words):
                if s < we and e > ws:
                    score[k] += a
                    break
        return [(w[2], float(sc), w[0], w[1]) for w, sc in zip(words, score)]

    @staticmethod
    def to_lines(res, hunk):
        """Aggregate hunk-token attributions to diff lines (sum of positive attributions per line)."""
        starts, pos = [], 0
        for ln in hunk.split("\n"):
            starts.append((pos, pos + len(ln), ln)); pos += len(ln) + 1
        sc = np.zeros(len(starts))
        for (s, e), sid, a in zip(res["offsets"], res["seq"], res["attr"]):
            if sid != 1 or e <= s:
                continue
            for k, (ls, le, _) in enumerate(starts):
                if s < le + 1 and e > ls:
                    sc[k] += a
                    break
        return [(l[2], float(x)) for l, x in zip(starts, sc)]

    # ----- faithfulness ----------------------------------------------------------------
    def masked_prob(self, res, positions):
        ids = res["ids"].clone()
        ids[list(positions)] = self.tok.pad_token_id
        return self.prob(ids[None].to(self.device), res["mask"])

    def faithfulness(self, res, k, rng):
        """Deletion test: replace top-k attributed content tokens by [PAD]; compare with k random content tokens."""
        content = [i for i, s in enumerate(res["seq"]) if s is not None]
        k = min(k, len(content))
        order = sorted(content, key=lambda i: -res["attr"][i])
        top = order[:k]
        rnd = list(rng.choice(content, size=k, replace=False))
        p0 = res["prob"]
        return {"p0": p0, "p_top": self.masked_prob(res, top), "p_rand": self.masked_prob(res, rnd), "k": k}
