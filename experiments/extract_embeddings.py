"""Pooled [CLS] embeddings of the full model (inference only, batch 8, bf16 autocast) for k-means."""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import ROOT, load_split
from sift.infer import SiftScorer

N_TRAIN = 4000
sc = SiftScorer(ROOT / "artifacts/full")
out = {}
for split in ("train", "val", "test"):
    df = load_split(split)
    if split == "train":
        df = df.sample(N_TRAIN, random_state=473).sort_index()
    ints = list(df.consolidated_intent); hs = [f"{r.file_path}\n{r.hunk}" for r in df.itertuples()]
    order = np.argsort([len(a) + len(b) for a, b in zip(ints, hs)])
    E = np.zeros((len(df), 768), np.float32); L = np.zeros(len(df), np.float32)
    for s in range(0, len(df), 8):
        ix = order[s:s + 8]
        enc = sc.tok([ints[i] for i in ix], [hs[i] for i in ix], truncation="longest_first", max_length=sc.max_len, padding=True, return_tensors="pt").to(sc.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z = sc.model.pooled(enc["input_ids"], enc["attention_mask"])
            lg = sc.model.align_head(z).squeeze(-1)
        E[ix] = z.float().cpu().numpy(); L[ix] = lg.float().cpu().numpy()
    out[f"{split}_emb"] = E; out[f"{split}_logit"] = L; out[f"{split}_ids"] = df.sample_id.values.astype(str)
    print(split, len(df), "peak GB", torch.cuda.max_memory_allocated() / 1e9, flush=True)
np.savez(ROOT / "experiments/results/pooled_embeddings.npz", **out)
