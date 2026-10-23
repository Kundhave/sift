"""Phase 5c: package full + student, export ONNX fp32 and dynamic-INT8, re-calibrate each on validation, evaluate on test."""
import json, shutil, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import ROOT, load_split
from sift.model import SiftModel
from sift.distill import export_onnx, quantize_int8, size_mb
from sift.infer import SiftScorer
from sift.calibrate import fit_temperature, calibration_report
from sift.rank import full_eval, best_threshold

ART = ROOT / "artifacts"
va, te = load_split("val"), load_split("test")
yv, yt = (va.label_clean == "misaligned").astype(int).values, (te.label_clean == "misaligned").astype(int).values
txt = lambda d: ([*d.consolidated_intent], [f"{r.file_path}\n{r.hunk}" for r in d.itertuples()])
out = {}

def evaluate(s, name, extra=None):
    iv, hv = txt(va); it, ht = txt(te)
    av, _, _ = s._logits(iv, hv); at, _, _ = s._logits(it, ht)
    T = fit_temperature(av, yv)
    pv, pt = 1 / (1 + np.exp(-av / T)), 1 / (1 + np.exp(-at / T))
    thr = best_threshold(yv, pv)
    s.T, s.threshold = T, thr
    out[name] = {"test": full_eval(pt, yt, list(te.session_id), list(te.mutation_difficulty), thr),
                 "calibration": calibration_report(av, yv, at, yt), **(extra or {})}
    return T, thr

# 1) torch artifacts
for run, a in (("full", "full"), ("student", "student")):
    import subprocess
    subprocess.run([sys.executable, "-m", "sift.make_artifact", "--run", run, "--out", f"artifacts/{a}"], check=True, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")})
s_full = SiftScorer(ART / "full"); s_stu = SiftScorer(ART / "student")
for s, n in ((s_full, "full"), (s_stu, "student_torch_fp32")):
    T, thr = evaluate(s, n)
    meta = json.load(open(s.dir / "meta.json")); meta.update(temperature=T, threshold=thr); json.dump(meta, open(s.dir / "meta.json", "w"), indent=2)
out["full"]["size_mb"] = size_mb(ART / "full/model.pt"); out["student_torch_fp32"]["size_mb"] = size_mb(ART / "student/model.pt")

# 2) ONNX fp32 + INT8 of the student
m = SiftModel("sentence-transformers/all-MiniLM-L6-v2"); m.load_state_dict(torch.load(ART / "student/model.pt", map_location="cpu"))
for tag in ("student_onnx_fp32", "student_onnx_int8"):
    (ART / tag).mkdir(exist_ok=True)
export_onnx(m, ART / "student_onnx_fp32/model.onnx")
quantize_int8(ART / "student_onnx_fp32/model.onnx", ART / "student_onnx_int8/model.onnx")
base = json.load(open(ART / "student/meta.json"))
for tag in ("student_onnx_fp32", "student_onnx_int8"):
    json.dump({**base, "backend": "onnx", "onnx_file": "model.onnx", "threads": 4}, open(ART / tag / "meta.json", "w"), indent=2)
    s = SiftScorer(ART / tag)
    T, thr = evaluate(s, tag, {"size_mb": size_mb(ART / tag / "model.onnx")})
    meta = json.load(open(ART / tag / "meta.json")); meta.update(temperature=T, threshold=thr); json.dump(meta, open(ART / tag / "meta.json", "w"), indent=2)
json.dump(out, open(ROOT / "experiments/results/variants.json", "w"), indent=2, default=float)
for k, v in out.items():
    print(k, round(v["test"]["macro_f1"], 4), round(v["test"]["ndcg@5"], 4), {t: round(x["auc"], 3) for t, x in v["test"]["pairwise_auc"].items()}, round(v["size_mb"], 1))
