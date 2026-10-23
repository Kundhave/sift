"""Deliverable 4 (interpretation, see DECISIONS.md): adversarially trained alignment scorer on frozen MiniLM embeddings."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import load_split, hunk_input, ROOT
from sift.gan_baseline import embed, train_gan, Gen, Disc
from sift.rank import full_eval, best_threshold

dev = "cpu"
cache = ROOT / "experiments/results/gan_embeddings.npz"
dfs = {s: load_split(s) for s in ("train", "val", "test")}
if not cache.exists():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=dev)
    t0 = time.time(); Z = {}
    for s, d in dfs.items():
        Z[f"{s}_i"] = embed(list(d.consolidated_intent), st); Z[f"{s}_h"] = embed([hunk_input(r) for _, r in d.iterrows()], st)
        print(s, time.time() - t0, flush=True)
    np.savez(cache, **Z)
Z = np.load(cache)
y = {s: (d.label_clean == "misaligned").astype(int).values for s, d in dfs.items()}
y_obs = (dfs["train"].label == "misaligned").astype(int).values
res = {"encoder": "sentence-transformers/all-MiniLM-L6-v2 (frozen)", "variants": {}}

def evaluate(score_val, score_test):
    thr = best_threshold(y["val"], 1 / (1 + np.exp(-(score_val - np.median(score_val)))))  # monotone map to (0,1)
    sv, st_ = 1 / (1 + np.exp(-(score_val - np.median(score_val)))), 1 / (1 + np.exp(-(score_test - np.median(score_val))))
    thr = best_threshold(y["val"], sv)
    return full_eval(st_, y["test"], list(dfs["test"].session_id), list(dfs["test"].mutation_difficulty), thr)

# reference: cosine similarity of the frozen embeddings (no training)
cos = lambda s: -(Z[f"{s}_i"] * Z[f"{s}_h"]).sum(1)
res["variants"]["frozen_cosine_dissimilarity"] = {"test": evaluate(cos("val"), cos("test")), "trainable_params": 0}

for name, sup in (("gan_unsupervised", False), ("gan_hybrid_with_misaligned_negatives", True)):
    t0 = time.time()
    G, D, hist = train_gan(Z["train_i"], Z["train_h"], y_obs, Z["val_i"], Z["val_h"], y["val"], epochs=40, sup=sup, device=dev)
    D.eval()
    score = lambda s: (-D(torch.tensor(Z[f"{s}_i"]), torch.tensor(Z[f"{s}_h"]))).detach().numpy()
    import io
    buf = io.BytesIO(); torch.save(D.state_dict(), buf)
    # CPU latency of the discriminator alone on one pair (embedding cost measured separately below)
    ei, eh = torch.tensor(Z["test_i"][:1]), torch.tensor(Z["test_h"][:1])
    for _ in range(20): D(ei, eh)
    ts = []
    for k in range(300):
        a, b = torch.tensor(Z["test_i"][k:k+1]), torch.tensor(Z["test_h"][k:k+1]); t = time.perf_counter(); D(a, b); ts.append((time.perf_counter() - t) * 1000)
    res["variants"][name] = {"test": evaluate(score("val"), score("test")), "history": hist, "train_time_s": time.time() - t0,
                             "trainable_params_D": sum(p.numel() for p in D.parameters()), "trainable_params_G": sum(p.numel() for p in G.parameters()),
                             "disc_size_mb": len(buf.getvalue()) / 1e6, "disc_latency_ms_p50": float(np.percentile(ts, 50)), "disc_latency_ms_p95": float(np.percentile(ts, 95)),
                             "best_val_auc": max(h["val_auc"] for h in hist), "min_gen_diversity": min(h["gen_diversity_cos"] for h in hist),
                             "final_gen_diversity": hist[-1]["gen_diversity_cos"]}
    print(name, res["variants"][name]["test"]["macro_f1"], res["variants"][name]["test"]["pairwise_auc"], flush=True)
# embedding cost on CPU for one pair
from sentence_transformers import SentenceTransformer
st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
d = dfs["test"].head(60); lat = []
for _, r in d.iterrows():
    t = time.perf_counter(); embed([r.consolidated_intent], st); embed([hunk_input(r)], st); lat.append((time.perf_counter() - t) * 1000)
res["embedding_latency_ms_p50_per_pair"] = float(np.percentile(lat[5:], 50)); res["embedding_latency_ms_p95_per_pair"] = float(np.percentile(lat[5:], 95))
res["embedder_size_mb"] = sum(p.numel() for p in st.parameters()) * 4 / 1e6
json.dump(res, open(ROOT / "experiments/results/gan.json", "w"), indent=2, default=float)
