"""K-means over the model's pooled [CLS] embeddings: saved index + cluster_id lookup used by the backend."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def _norm(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)


class KMeansIndex:
    """centroids (k,d) fitted on L2-normalised train embeddings, per-cluster misaligned prior, and the stored train points."""
    def __init__(self, centroids, prior, train_emb=None, train_ids=None, train_y=None, meta=None):
        self.c, self.prior, self.meta = _norm(centroids), np.asarray(prior, float), meta or {}
        self.train_emb = None if train_emb is None else _norm(train_emb)
        self.train_ids, self.train_y = train_ids, train_y
        self.train_cl = None if self.train_emb is None else self.assign(self.train_emb)[0]

    def assign(self, emb):
        """-> (cluster_id[n], distance_to_assigned_centroid[n], all_distances[n,k]) using cosine distance."""
        d = 1.0 - _norm(emb) @ self.c.T
        return d.argmin(1), d.min(1), d

    def cluster_id(self, emb) -> int:
        """Cluster of one pooled embedding (shape (d,))."""
        return int(self.assign(np.asarray(emb)[None])[0][0])

    def similar(self, emb, n=5, only_misaligned=True):
        """Nearest stored train hunks (same cluster first by cosine) -> [{sample_id, label, sim}]"""
        if self.train_emb is None:
            return []
        s = self.train_emb @ _norm(np.asarray(emb)[None])[0]
        idx = np.argsort(-s)
        out = []
        for i in idx:
            if only_misaligned and not self.train_y[i]:
                continue
            out.append({"sample_id": str(self.train_ids[i]), "label": "misaligned" if self.train_y[i] else "aligned", "similarity": float(s[i])})
            if len(out) >= n:
                break
        return out

    def save(self, d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "kmeans.npz", centroids=self.c, prior=self.prior,
                            train_emb=self.train_emb.astype(np.float16), train_ids=np.asarray(self.train_ids, str), train_y=np.asarray(self.train_y, int))
        json.dump({**self.meta, "k": int(len(self.c)), "dim": int(self.c.shape[1])}, open(d / "meta.json", "w"), indent=2)

    @classmethod
    def load(cls, d):
        d = Path(d); z = np.load(d / "kmeans.npz", allow_pickle=False)
        return cls(z["centroids"], z["prior"], z["train_emb"].astype(np.float32), z["train_ids"], z["train_y"], json.load(open(d / "meta.json")))


def fit_index(train_emb, train_y, train_ids, k, seed=473, meta=None):
    from sklearn.cluster import KMeans
    x = _norm(train_emb)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x)
    y = np.asarray(train_y, int)
    prior = np.array([(y[km.labels_ == c].sum() + 0.22 * 5) / ((km.labels_ == c).sum() + 5) for c in range(k)])  # smoothed rate
    return KMeansIndex(km.cluster_centers_, prior, x, train_ids, y, meta)
