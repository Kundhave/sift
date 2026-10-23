"""Build the three lite-only data files into OUT (default: deploy/lite/build):
  artifacts/baselines_lite.joblib   pre-fitted B4/B5 (same fit as serve.BaselineScorer, train split) + val thresholds
  artifacts/student_onnx_int8/tokenizer.json   fast-tokenizer file of the MiniLM student
  data/lite_samples.json            the curated test-set samples (serve._load_samples) frozen as JSON
Run from the repo root:  PYTHONPATH=src .venv/bin/python deploy/lite/build_lite_data.py OUT"""
import json, sys
from pathlib import Path
import joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sift import serve
from sift.baselines import LexicalFeatures
from sift.data import load_split
from sift.rank import best_threshold

out = Path(sys.argv[1] if len(sys.argv) > 1 else "deploy/lite/build")
(out / "artifacts" / "student_onnx_int8").mkdir(parents=True, exist_ok=True)
(out / "data").mkdir(parents=True, exist_ok=True)

tr, va = load_split("train"), load_split("val")
lf = LexicalFeatures(tr)
if hasattr(lf.vec, "stop_words_"):
    del lf.vec.stop_words_
lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(lf(tr)[0], (tr.label == "misaligned").values.astype(int))
yva = (va.label_clean == "misaligned").values.astype(int)
Xva, sim, _ = lf(va)
grid = np.linspace(0.05, 0.95, 91)
joblib.dump({"lf": lf, "lr": lr, "threshold_b4": float(best_threshold(yva, 1 - sim, grid)),
             "threshold_b5": float(best_threshold(yva, lr.predict_proba(Xva)[:, 1], grid))},
            out / "artifacts" / "baselines_lite.joblib", compress=3)

from sift.model import load_tokenizer
load_tokenizer("sentence-transformers/all-MiniLM-L6-v2").backend_tokenizer.save(
    str(out / "artifacts" / "student_onnx_int8" / "tokenizer.json"))
json.dump(serve._load_samples(), open(out / "data" / "lite_samples.json", "w"))
print("wrote", out)
