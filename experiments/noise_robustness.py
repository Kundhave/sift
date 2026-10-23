"""Noise-robustness sweep (reduced setting). ModernBERT cross-encoder trained on a train subset whose labels get symmetric random flips at
rate r in {0,10,20,30}%, with and without the denoising stage (rule filters + confident learning on out-of-fold TF-IDF/logistic-regression
probabilities, 5-fold grouped by session). Selection/threshold on validation (clean); test uses label_clean. Output: results/noise_robustness.json + png."""
import argparse, gc, json, math, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from sift import SEED
from sift.data import load_split
from sift.denoise import rule_filter_mask, confident_learning_issues, score_flag_set
from sift.losses import multitask_loss
from sift.model import SiftModel, load_tokenizer, DEFAULT_ENCODER
from sift.rank import best_threshold, full_eval
from sift.train import PairDataset, bucketed_batches, predict, set_seed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics.pairwise import cosine_similarity

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=3000); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--rates", default="0,0.1,0.2,0.3"); ap.add_argument("--max_steps", type=int, default=0); ap.add_argument("--out", default="noise_robustness")
a = ap.parse_args(); dev = "cuda" if torch.cuda.is_available() else "cpu"

tr = load_split("train").sample(a.n_train, random_state=SEED).reset_index(drop=True)
va, te = load_split("val"), load_split("test")
if a.max_steps: va, te = va.sample(300, random_state=SEED).reset_index(drop=True), te.sample(300, random_state=SEED).reset_index(drop=True)
ycl = (tr.label_clean == "misaligned").astype(int).values
tok = load_tokenizer(DEFAULT_ENCODER)
dva, dte = PairDataset(va, tok, a.max_len, labels="label_clean"), PairDataset(te, tok, a.max_len, labels="label_clean")

def oof_probs(df, y):
    """Cheap out-of-fold P(misaligned): LR on [tfidf cosine(intent,hunk), log sizes]. Grouped by session."""
    tf = TfidfVectorizer(token_pattern=r"[A-Za-z_]{3,}", max_features=20000, sublinear_tf=True).fit(list(df.consolidated_intent) + list(df.hunk))
    A, B = tf.transform(df.consolidated_intent), tf.transform(df.hunk)
    cos = np.asarray(A.multiply(B).sum(1)).ravel()
    X = np.c_[cos, np.log1p(df.hunk.str.len()), np.log1p(df.consolidated_intent.str.len())]
    p = np.zeros(len(df))
    for t, h in GroupKFold(5).split(X, y, df.session_id):
        p[h] = LogisticRegression(max_iter=500).fit(X[t], y[t]).predict_proba(X[h])[:, 1]
    return p

def train_eval(rows, y_noisy):
    d = PairDataset(rows, tok, a.max_len, labels="label"); d.y = y_noisy.astype(int)
    set_seed(SEED); mb, acc = 8, 2
    m = SiftModel(DEFAULT_ENCODER, gradient_checkpointing=True).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-5, weight_decay=0.01)
    total = math.ceil(len(d) / (mb * acc)) * a.epochs; warm = max(1, int(0.1 * total))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    rng = np.random.RandomState(SEED); n = 0
    for ep in range(a.epochs):
        m.train(); bs = bucketed_batches(d, mb, rng)
        for bi, idx in enumerate(bs):
            b = d.collate(idx)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = m(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["token_type_ids"].to(dev))
            loss, _ = multitask_loss({k: v.float() for k, v in o.items()}, b["y"].to(dev), b["d"].to(dev)); (loss / acc).backward(); n += 1
            if (bi + 1) % acc == 0 or bi == len(bs) - 1:
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            if a.max_steps and n >= a.max_steps: break
        if a.max_steps and n >= a.max_steps: break
    pv, pt = predict(m, dva, dev, bs=16)["prob"], predict(m, dte, dev, bs=16)["prob"]
    thr = best_threshold(dva.y, pv)
    r = full_eval(pt, dte.y, list(te.session_id), list(te.mutation_difficulty), thr)
    del m, opt; gc.collect(); torch.cuda.empty_cache()
    return r

out = {"setting": f"REDUCED: ModernBERT-base, train subset {a.n_train} rows (seed {SEED}), {a.epochs} epochs, max_len {a.max_len}, micro_bs 8 x accum 2, bf16, single seed; "
       "symmetric label flips injected at the stated rate on label_clean; denoising = rule filter + confident learning (prune_by_noise_rate) on 5-fold OOF "
       "TF-IDF/logistic-regression probabilities (not the ModernBERT OOF used in the main denoising study); eval on full val (threshold) and test (label_clean).",
       "debug_max_steps": a.max_steps, "results": []}
for r in [float(x) for x in a.rates.split(",")]:
    rs = np.random.RandomState(SEED + int(r * 100)); flip = rs.rand(len(tr)) < r
    yn = np.where(flip, 1 - ycl, ycl)
    for arm in ("no_denoise", "denoise"):
        t0 = time.time(); keep = np.ones(len(tr), bool); info = {}
        if arm == "denoise":
            rm = rule_filter_mask(tr); p = oof_probs(tr, yn)
            cl = confident_learning_issues(yn, p, filter_by="prune_by_noise_rate")
            keep = ~(rm | cl); info = {"rule_removed": int(rm.sum()), "cl_flagged": score_flag_set(cl & ~rm, flip), "kept": int(keep.sum())}
        res = train_eval(tr[keep].reset_index(drop=True), yn[keep])
        row = {"noise_rate": r, "arm": arm, "n_train_used": int(keep.sum()), "realised_flip_rate": float(flip.mean()), "denoise_info": info,
               "test": {k: res[k] for k in ("macro_f1", "precision", "recall", "specificity", "ndcg@5", "ece", "pairwise_auc") if k in res}, "seconds": time.time() - t0}
        out["results"].append(row); print(r, arm, round(res["macro_f1"], 4), flush=True)
        json.dump(out, open(ROOT / "results" / f"{a.out}.json", "w"), indent=2, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for j, (key, lab) in enumerate((("macro_f1", "test macro-F1"), ("ndcg@5", "test NDCG@5"))):
    for arm in ("no_denoise", "denoise"):
        rr = [x for x in out["results"] if x["arm"] == arm]
        ax[j].plot([x["noise_rate"] * 100 for x in rr], [x["test"][key] for x in rr], marker="o", label=arm.replace("_", " "))
    ax[j].set_xlabel("injected label-noise rate (%)"); ax[j].set_ylabel(lab); ax[j].legend(); ax[j].grid(alpha=.3)
fig.suptitle("Noise robustness (reduced setting, 1 seed)", fontsize=10); fig.tight_layout(); fig.savefig(ROOT / "results" / f"{a.out}.png", dpi=150)
