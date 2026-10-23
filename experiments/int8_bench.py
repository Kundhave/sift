"""Phase 5b: verify + calibrate INT8 (and fp32) ONNX student on CPU, write meta.json, benchmark size and single-pair latency."""
import json, sys, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import ROOT, load_split
from sift.infer import SiftScorer
from sift.calibrate import fit_temperature, calibration_report
from sift.rank import full_eval, best_threshold
from sift.distill import size_mb, cpu_latency, cpu_info

ART = ROOT / "artifacts"
va, te = load_split("val"), load_split("test")
yv, yt = (va.label_clean == "misaligned").astype(int).values, (te.label_clean == "misaligned").astype(int).values
txt = lambda d: (list(d.consolidated_intent), [f"{r.file_path}\n{r.hunk}" for r in d.itertuples()])
out = {"cpu": cpu_info(), "threads": 4, "note": "CPU shared with a running GPU training job (its data loader / python thread); latencies may be pessimistic. Single pair, batch 1, max_len 512, tokenisation included."}
base = json.load(open(ART / "student/meta.json"))
iv, hv = txt(va); it, ht = txt(te)
sess = {}
for tag in ("student_onnx_fp32", "student_onnx_int8"):
    d = ART / tag
    json.dump({**base, "backend": "onnx", "onnx_file": "model.onnx", "threads": 4, "source_run": "student"}, open(d / "meta.json", "w"), indent=2)
    s = SiftScorer(d)
    av = s._logits(iv, hv)[0]; at = s._logits(it, ht)[0]
    T = fit_temperature(av, yv); pv, pt = 1 / (1 + np.exp(-av / T)), 1 / (1 + np.exp(-at / T))
    thr = best_threshold(yv, pv)
    out[tag] = {"size_mb": size_mb(d / "model.onnx"), "temperature": T, "threshold": float(thr),
                "test": full_eval(pt, yt, list(te.session_id), list(te.mutation_difficulty), thr),
                "calibration": calibration_report(av, yv, at, yt)}
    meta = json.load(open(d / "meta.json")); meta.update(temperature=T, threshold=float(thr),
        verified={"test_macro_f1": out[tag]["test"]["macro_f1"], "size_mb": out[tag]["size_mb"]}); json.dump(meta, open(d / "meta.json", "w"), indent=2)
    sample = te.sample(200, random_state=473)
    pairs = [(r.consolidated_intent, f"{r.file_path}\n{r.hunk}") for r in sample.itertuples()]
    out[tag]["latency_ms"] = cpu_latency(lambda x: s._logits([x[0]], [x[1]]), pairs, warmup=10)
    out[tag]["fp32_vs_int8_logit_corr"] = None
    sess[tag] = at
out["student_onnx_int8"]["fp32_vs_int8_logit_corr"] = float(np.corrcoef(sess["student_onnx_fp32"], sess["student_onnx_int8"])[0, 1])
out["targets"] = {"size_mb_lt": 30, "p95_ms_lt": 100}
out["targets_met"] = {"size": out["student_onnx_int8"]["size_mb"] < 30, "p95": out["student_onnx_int8"]["latency_ms"]["p95_ms"] < 100}
json.dump(out, open(ROOT / "results/latency_size_onnx.json", "w"), indent=2, default=float)
for t in ("student_onnx_fp32", "student_onnx_int8"):
    v = out[t]; print(t, round(v["size_mb"], 1), round(v["test"]["macro_f1"], 4), round(v["test"]["ndcg@5"], 4), v["latency_ms"])
print(out["targets_met"], out["student_onnx_int8"]["fp32_vs_int8_logit_corr"])
