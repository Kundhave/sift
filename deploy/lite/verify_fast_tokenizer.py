"""Check the tokenizers-only path equals the transformers path (ids and probabilities) on N test hunks.
Run: PYTHONPATH=src .venv/bin/python deploy/lite/verify_fast_tokenizer.py <tokenizer.json> [N]"""
import sys, json, os
from pathlib import Path
import numpy as np
from sift import infer
from sift.data import load_split
from sift.model import load_tokenizer
tj, n = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 20
df = load_split("test").sample(n, random_state=473)
I, H = list(df.consolidated_intent), [f"{a}\n{b}" for a, b in zip(df.file_path, df.hunk)]
slow, fast = load_tokenizer("sentence-transformers/all-MiniLM-L6-v2"), infer._FastTok(tj, 512)
same_ids = 0
for i, h in zip(I, H):
    a = slow([i], [h], truncation="longest_first", max_length=512, padding=True, return_tensors="np")
    b = fast([i], [h])
    same_ids += int(all(np.array_equal(a[k], b[k]) for k in ("input_ids", "attention_mask", "token_type_ids")))
art = Path("artifacts/student_onnx_int8")
s_slow = infer.SiftScorer(art); s_fast = infer.SiftScorer(art); s_fast.tok = fast
A = s_slow._logits(I, H); B = s_fast._logits(I, H)
os.environ["SIFT_LOW_MEM"] = "1"
s_low = infer.SiftScorer(art); s_low.tok = fast; C = s_low._logits(I, H)
out = {"n": n, "token_ids_identical": f"{same_ids}/{n}",
       "logits_bitwise_identical_vs_transformers": all(np.array_equal(a, b) for a, b in zip(A, B)),
       "max_abs_logit_diff_lowmem_vs_default": float(max(np.abs(a - c).max() for a, c in zip(A, C)))}
print(json.dumps(out))
