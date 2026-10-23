"""Single-diff inference: session intent turns + unified diff -> hunks ranked by misalignment probability.

Artifact directory layout:  model.pt (torch state_dict) *or* model.onnx,  meta.json
meta.json: {"encoder", "max_len", "temperature", "threshold", "backend": "torch"|"onnx"}
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
from .data import consolidate_intent
from .diffparse import parse_diff

DRIFT_CLASSES = ["collateral", "scope_expansion", "purpose_drift"]   # same order as sift.model.DRIFT_CLASSES (tested)


class _FastTok:
    """`tokenizers`-only pair tokenizer built from the artifact's tokenizer.json (no torch/transformers import).
    Mirrors AutoTokenizer(a, b, truncation='longest_first', max_length, padding=True, return_tensors='np')."""

    def __init__(self, path: Path, max_len: int):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(path))
        self.t.enable_truncation(max_length=max_len, strategy="longest_first")
        self.t.enable_padding()

    def __call__(self, a, b, truncation=None, max_length=None, padding=None, return_tensors=None):
        encs = self.t.encode_batch(list(zip(a, b)))
        return {"input_ids": np.array([e.ids for e in encs], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encs], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encs], dtype=np.int64)}


class SiftScorer:
    def __init__(self, artifact: str | Path, device: str | None = None):
        self.dir = Path(artifact)
        self.meta = json.load(open(self.dir / "meta.json"))
        self.max_len = int(self.meta.get("max_len", 512))
        self.backend = self.meta.get("backend", "torch")
        if self.backend == "onnx" and (self.dir / "tokenizer.json").exists():
            self.tok = _FastTok(self.dir / "tokenizer.json", self.max_len)
        else:
            from .model import load_tokenizer   # lazy: imports torch + transformers
            self.tok = load_tokenizer(self.meta["encoder"])
        self.T = float(self.meta.get("temperature", 1.0))
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.backend = self.meta.get("backend", "torch")
        if self.backend == "onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(self.meta.get("threads", 4))
            if os.environ.get("SIFT_LOW_MEM"):   # lite deployment: no pooled arena / memory-pattern planning (peak RSS)
                so.intra_op_num_threads = 1
                so.enable_cpu_mem_arena = False
                so.enable_mem_pattern = False
            self.sess = ort.InferenceSession(str(self.dir / self.meta.get("onnx_file", "model.onnx")), so,
                                             providers=["CPUExecutionProvider"])
            self.onnx_inputs = {i.name for i in self.sess.get_inputs()}
        else:
            import torch
            from .model import SiftModel
            self.torch = torch
            self.device = device or os.environ.get("SIFT_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model = SiftModel(self.meta["encoder"])
            self.model.load_state_dict(torch.load(self.dir / "model.pt", map_location="cpu"))
            self.model.to(self.device).eval()

    def _logits(self, intents: list[str], hunks: list[str], bs: int = 16):
        if os.environ.get("SIFT_LOW_MEM") and self.backend == "onnx":
            bs = int(os.environ.get("SIFT_LOW_MEM_BS", "2"))   # bounds peak RSS on long diffs (see DECISIONS.md)
        A, D, C = [], [], []
        for s in range(0, len(hunks), bs):
            enc = self.tok(intents[s:s + bs], hunks[s:s + bs], truncation="longest_first", max_length=self.max_len,
                           padding=True, return_tensors="np")
            if self.backend == "onnx":
                feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.onnx_inputs}
                a, d, c = self.sess.run(["align_logit", "drift_logits", "cal_logit"], feed)
            else:
                t = self.torch
                with t.no_grad():
                    o = self.model(*[t.tensor(enc[k]).to(self.device) for k in ("input_ids", "attention_mask")])
                a, d, c = (o["align_logit"].float().cpu().numpy(), o["drift_logits"].float().cpu().numpy(),
                           o["cal_logit"].float().cpu().numpy())
            A.append(np.atleast_1d(a)); D.append(d); C.append(np.atleast_1d(c))
        return np.concatenate(A), np.concatenate(D), np.concatenate(C)

    def embed(self, intents: list[str], hunks: list[str], bs: int = 16) -> np.ndarray:
        """Pooled [CLS] embeddings (n, hidden); PyTorch backend only."""
        if self.backend == "onnx":
            raise NotImplementedError("embeddings need the PyTorch backend")
        t, out = self.torch, []
        for s in range(0, len(hunks), bs):
            enc = self.tok(intents[s:s + bs], hunks[s:s + bs], truncation="longest_first", max_length=self.max_len,
                           padding=True, return_tensors="np")
            with t.no_grad():
                z = self.model.pooled(*[t.tensor(enc[k]).to(self.device) for k in ("input_ids", "attention_mask")])
            out.append(z.float().cpu().numpy())
        return np.concatenate(out)

    def score(self, intent_turns: list[str] | str, diff: str) -> dict:
        turns = [intent_turns] if isinstance(intent_turns, str) else list(intent_turns)
        intent = consolidate_intent(turns)
        hunks = parse_diff(diff)
        if not hunks:
            return {"intent": intent, "hunks": [], "n_hunks": 0}
        texts = [f"{h.file_path}\n{h.text}" for h in hunks]
        t0 = time.perf_counter()
        a, d, c = self._logits([intent] * len(texts), texts)
        lat = time.perf_counter() - t0
        p = 1 / (1 + np.exp(-a / self.T))
        dp = np.exp(d - d.max(1, keepdims=True)); dp /= dp.sum(1, keepdims=True)
        rows = [{"index": hunks[i].index, "file": hunks[i].file_path, "header": hunks[i].header, "p_misaligned": float(p[i]),
                 "verdict": "misaligned" if p[i] >= self.threshold else "aligned",
                 "drift_type": DRIFT_CLASSES[int(dp[i].argmax())] if p[i] >= self.threshold else None,
                 "drift_probs": dict(zip(DRIFT_CLASSES, map(float, dp[i]))),
                 "confidence_head": float(1 / (1 + np.exp(-c[i]))), "text": hunks[i].text} for i in range(len(hunks))]
        rows.sort(key=lambda r: -r["p_misaligned"])
        for rank, r in enumerate(rows, 1):
            r["rank"] = rank
        return {"intent": intent, "n_hunks": len(rows), "latency_s": lat, "hunks": rows}


def main():
    ap = argparse.ArgumentParser(description="Rank diff hunks by misalignment with the stated intent")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--intent", action="append", required=True, help="one intent turn; repeat for multi-turn")
    ap.add_argument("--diff", required=True, help="path to a unified diff, or - for stdin")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    import sys
    diff = sys.stdin.read() if a.diff == "-" else Path(a.diff).read_text()
    r = SiftScorer(a.artifact).score(a.intent, diff)
    if a.json:
        print(json.dumps(r, indent=2))
        return
    print(f"intent: {r['intent']}\n{r['n_hunks']} hunk(s)")
    for h in r["hunks"]:
        print(f"#{h['rank']}  {h['file']} {h['header']}  p={h['p_misaligned']:.3f}  {h['verdict']}"
              + (f" ({h['drift_type']})" if h["drift_type"] else ""))


if __name__ == "__main__":
    main()
