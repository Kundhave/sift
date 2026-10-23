"""Phase 5c: K-means on pooled embeddings: elbow/silhouette, purity/NMI, t-SNE, hybrid cluster features (with vs without)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.manifold import TSNE
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import ROOT, load_split
from sift.kmeans import fit_index, _norm
from sift.rank import full_eval, best_threshold, ece
from sift.calibrate import fit_temperature, apply_temperature

Z = np.load(ROOT / "experiments/results/pooled_embeddings.npz", allow_pickle=True)
df = {s: load_split(s).set_index("sample_id") for s in ("train", "val", "test")}
def lab(s):
    d = df[s].loc[Z[f"{s}_ids"]]
    y = (d.label_clean == "misaligned").astype(int).values
    dt = np.where(y == 1, d.drift_type_clean.values, "aligned")
    return d, y, dt
dtr, ytr, dttr = lab("train"); dva, yva, dtva = lab("val"); dte, yte, dtte = lab("test")
Xtr, Xva, Xte = (_norm(Z[f"{s}_emb"]) for s in ("train", "val", "test"))
res = {"embedding": "full model pooled [CLS] (768-d), L2-normalised; k-means fit on 4000 random train pairs (seed 473); val/test assigned by nearest centroid (cosine)",
       "n_fit": len(ytr)}
# ---- choose k
from sklearn.cluster import KMeans
ks = list(range(2, 21)); inertia, sil = [], []
for k in ks:
    km = KMeans(k, n_init=5, random_state=473).fit(Xtr)
    inertia.append(float(km.inertia_)); sil.append(float(silhouette_score(Xtr, km.labels_, sample_size=3000, random_state=473)))
k_best = ks[int(np.argmax(sil))]
res["k_scan"] = {"k": ks, "inertia": inertia, "silhouette": sil}; res["k_chosen"] = k_best
res["k_rule"] = "k maximising silhouette on the train fit set (elbow plotted for inspection; no sharp elbow is claimed)"
idx = fit_index(Xtr, ytr, Z["train_ids"], k_best, meta={"source": "artifacts/full pooled [CLS]", "n_fit": len(ytr), "seed": 473})
idx.save(ROOT / "artifacts/kmeans")
# ---- purity / NMI (train-fit clusters applied to test; also within-train)
def purity(cl, lbl):
    return float(sum(np.bincount(pd_).max() for pd_ in [np.unique(lbl[cl == c], return_inverse=True)[1] for c in np.unique(cl)]) / len(cl))
res["clustering_quality"] = {}
for name, (X, y, dt) in {"train_fit": (Xtr, ytr, dttr), "val": (Xva, yva, dtva), "test": (Xte, yte, dtte)}.items():
    cl = idx.assign(X)[0]
    res["clustering_quality"][name] = {"purity_vs_alignment": purity(cl, y), "purity_vs_drift4": purity(cl, dt), "nmi_vs_alignment": float(normalized_mutual_info_score(y, cl)),
                                       "nmi_vs_drift4": float(normalized_mutual_info_score(dt, cl)), "majority_class_rate_alignment": float(max(y.mean(), 1 - y.mean()))}
cl_te = idx.assign(Xte)[0]
res["test_cluster_table"] = [{"cluster": int(c), "n": int((cl_te == c).sum()), "misaligned_rate": float(yte[cl_te == c].mean()),
                              "train_prior": float(idx.prior[c]), "drift_counts": {t: int((dtte[cl_te == c] == t).sum()) for t in np.unique(dtte)}} for c in range(k_best)]
# ---- hybrid features: Platt on [logit] vs [logit + cluster prior + dist to nearest centroid + one-hot cluster]; fit on val (5-fold CV preds), test = model fit on all val
def feats(split, hybrid):
    lg = Z[f"{split}_logit"][:, None]
    if not hybrid:
        return lg
    X = {"val": Xva, "test": Xte}[split]
    cl, dmin, _ = idx.assign(X)
    return np.hstack([lg, idx.prior[cl][:, None], dmin[:, None]])
sid = {s: list(df[s].loc[Z[f"{s}_ids"]].session_id) for s in ("val", "test")}
tier = list(dte.mutation_difficulty)
res["hybrid"] = {}
for name, hyb in (("logit_only_platt", False), ("logit_plus_cluster_features", True)):
    Fv, Ft = feats("val", hyb), feats("test", hyb)
    lr = LogisticRegression(C=1.0, max_iter=1000)
    pv = cross_val_predict(lr, Fv, yva, cv=StratifiedKFold(5, shuffle=True, random_state=473), method="predict_proba")[:, 1]
    thr = best_threshold(yva, pv); lr.fit(Fv, yva); pt = lr.predict_proba(Ft)[:, 1]
    res["hybrid"][name] = {"test": full_eval(pt, yte, sid["test"], tier, thr), "coef": lr.coef_.tolist(), "features": ["logit"] + (["cluster_prior", "dist_to_centroid"] if hyb else [])}
res["hybrid"]["delta_cluster_minus_baseline"] = {m: res["hybrid"]["logit_plus_cluster_features"]["test"][m] - res["hybrid"]["logit_only_platt"]["test"][m] for m in ("macro_f1", "ndcg@5", "ece")}
for t in ("random", "hard"):
    res["hybrid"]["delta_cluster_minus_baseline"][f"auc_{t}"] = res["hybrid"]["logit_plus_cluster_features"]["test"]["pairwise_auc"][t]["auc"] - res["hybrid"]["logit_only_platt"]["test"]["pairwise_auc"][t]["auc"]
d = res["hybrid"]["delta_cluster_minus_baseline"]
res["honest_conclusion"] = ("Cluster features change test macro-F1 by %+.4f, NDCG@5 by %+.4f, hard-tier AUC by %+.4f, ECE by %+.4f (single seed, single k-means fit; no CIs)."
                            % (d["macro_f1"], d["ndcg@5"], d["auc_hard"], d["ece"]))
# ---- t-SNE on test (2700 points)
ts = TSNE(2, perplexity=35, init="pca", random_state=473).fit_transform(Xte)
res["tsne"] = {"perplexity": 35, "n": len(ts)}
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
cols = {"aligned": "#9aa5b1", "collateral": "#d1495b", "scope_expansion": "#edae49", "purpose_drift": "#00798c"}
for t, c in cols.items():
    m = dtte == t; ax[0].scatter(*ts[m].T, s=5, c=c, label=t, alpha=.7)
ax[0].legend(markerscale=2, fontsize=7); ax[0].set_title("t-SNE (test) by drift type", fontsize=9)
ax[1].scatter(*ts.T, s=5, c=cl_te, cmap="tab20"); ax[1].set_title(f"t-SNE (test) by k-means cluster (k={k_best})", fontsize=9)
ax[2].scatter(*ts.T, s=5, c=yte, cmap="coolwarm"); ax[2].set_title("t-SNE (test) by label (red = misaligned)", fontsize=9)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(ROOT / "results/kmeans_tsne.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(ks, inertia, "o-"); ax[0].set_title("Elbow (inertia)", fontsize=9); ax[0].set_xlabel("k")
ax[1].plot(ks, sil, "o-", color="#d1495b"); ax[1].axvline(k_best, ls="--", color="gray"); ax[1].set_title(f"Silhouette (best k={k_best})", fontsize=9); ax[1].set_xlabel("k")
fig.tight_layout(); fig.savefig(ROOT / "results/kmeans_elbow_silhouette.png", dpi=150); plt.close(fig)
json.dump(res, open(ROOT / "results/kmeans.json", "w"), indent=2, default=float)
print(k_best, res["clustering_quality"]["test"]); print(res["honest_conclusion"])
for n in ("logit_only_platt", "logit_plus_cluster_features"):
    t = res["hybrid"][n]["test"]; print(n, round(t["macro_f1"], 4), round(t["ndcg@5"], 4), round(t["ece"], 4), {k: round(v["auc"], 3) for k, v in t["pairwise_auc"].items()})
