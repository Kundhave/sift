"""Latency/size of the main ModernBERT model (artifacts/full), batch 1, ~median-length test pairs, L=512.
Usage: CUDA_VISIBLE_DEVICES='' python experiments/latency_full.py --device cpu ; python experiments/latency_full.py --device cuda
Merges into results/latency_full.json (top-level p50_ms/p95_ms/size_mb = CPU, read by make_tables.py)."""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--runs", type=int, default=100)
a = ap.parse_args()
os.environ["SIFT_DEVICE"] = a.device
import torch
from sift.data import load_split
from sift.infer import SiftScorer
from sift.distill import cpu_info
te = load_split("test")
pairs = [(r.consolidated_intent, f"{r.file_path}\n{r.hunk}") for r in te.itertuples()]
s = SiftScorer(ROOT / "artifacts/full", device=a.device)
lens = np.array([min(len(s.tok(i, h)["input_ids"]), s.max_len) for i, h in pairs])
med = float(np.median(lens)); idx = np.argsort(np.abs(lens - med))[: a.warmup + a.runs]   # closest to the median length
rng = np.random.RandomState(473); idx = rng.permutation(idx)
sel = [pairs[i] for i in idx]
def run(x): s._logits([x[0]], [x[1]], bs=1)
for x in sel[:a.warmup]: run(x)
ts = []
if a.device == "cuda": torch.cuda.reset_peak_memory_stats()
for x in sel[a.warmup:a.warmup + a.runs]:
    if a.device == "cuda": torch.cuda.synchronize()
    t = time.perf_counter(); run(x)
    if a.device == "cuda": torch.cuda.synchronize()
    ts.append((time.perf_counter() - t) * 1000)
res = {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "mean_ms": float(np.mean(ts)), "n": len(ts), "warmup": a.warmup,
       "median_pair_tokens": med, "selected_token_len_range": [int(lens[idx].min()), int(lens[idx].max())], "torch_threads": torch.get_num_threads(),
       "dtype": str(next(s.model.parameters()).dtype), "tokenisation_included": True}
if a.device == "cuda":
    res["gpu"] = torch.cuda.get_device_name(0); res["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
p = ROOT / "results/latency_full.json"
out = json.load(open(p)) if p.exists() else {}
out["model"] = "artifacts/full (ModernBERT-base, torch, L=512, batch 1)"
out["size_mb"] = os.path.getsize(ROOT / "artifacts/full/model.pt") / 1e6
out["cpu_info"] = cpu_info()
out["note"] = "Machine idle during measurement. Tokenisation included. Pairs chosen closest to the median token length of the test split."
out[a.device] = res
if a.device == "cpu": out.update(p50_ms=res["p50_ms"], p95_ms=res["p95_ms"], torch_threads=res["torch_threads"])
json.dump(out, open(p, "w"), indent=2)
print(a.device, res)
