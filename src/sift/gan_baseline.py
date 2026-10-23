"""INTERPRETATION TAKEN for Deliverable 4 (see DECISIONS.md): Sift has no GAN in its design, so this is an
adversarially trained alignment scorer on FROZEN embeddings of (intent, hunk).

  G(z, e_i)  -> synthetic aligned-hunk embedding  (z ~ N(0, I), e_i = intent embedding)
  D(e_i, e_h) -> P(pair is a real aligned pair)    (also used as the alignment scorer; misalignment score = 1 - D)

Training sees ONLY aligned pairs as 'real' (the standard GAN objective).  Variant `sup=True` additionally
treats real *misaligned* training pairs as negatives for D (a supervised hybrid), reported separately.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Gen(nn.Module):
    def __init__(self, d_emb=384, d_z=64, h=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_emb + d_z, h), nn.LeakyReLU(0.2), nn.Linear(h, h), nn.LeakyReLU(0.2), nn.Linear(h, d_emb))
        self.d_z = d_z

    def forward(self, e_i, z=None):
        z = torch.randn(e_i.shape[0], self.d_z, device=e_i.device) if z is None else z
        return F.normalize(self.net(torch.cat([e_i, z], -1)), dim=-1)


class Disc(nn.Module):
    def __init__(self, d_emb=384, h=512):
        super().__init__()
        # pair features: [e_i, e_h, e_i*e_h, |e_i-e_h|]
        self.net = nn.Sequential(nn.Linear(4 * d_emb, h), nn.LeakyReLU(0.2), nn.Dropout(0.2), nn.Linear(h, h // 2), nn.LeakyReLU(0.2),
                                 nn.Linear(h // 2, 1))

    def forward(self, e_i, e_h):
        return self.net(torch.cat([e_i, e_h, e_i * e_h, (e_i - e_h).abs()], -1)).squeeze(-1)


@torch.no_grad()
def embed(texts, st_model, bs=64, max_len=512):
    st_model.max_seq_length = max_len
    return st_model.encode(texts, batch_size=bs, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def diversity(x: torch.Tensor) -> float:
    """Mean pairwise cosine among a batch of generated embeddings (1.0 = complete mode collapse)."""
    x = F.normalize(x, dim=-1); s = x @ x.T
    n = x.shape[0]
    return float((s.sum() - n) / (n * (n - 1)))


def train_gan(Ei, Eh, y_obs, Ei_val, Eh_val, y_val, *, epochs=40, bs=128, lr=2e-4, sup=False, seed=473, device="cpu", d_z=64):
    """Ei/Eh: (N,d) frozen embeddings; y_obs: 1 = misaligned. Returns models, history, best-by-val-AUC D state."""
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed); np.random.seed(seed)
    d = Ei.shape[1]
    G, D = Gen(d, d_z).to(device), Disc(d).to(device)
    oG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999)); oD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    Ei_t, Eh_t, y_t = [torch.tensor(a, dtype=torch.float32, device=device) for a in (Ei, Eh, y_obs)]
    real_idx = np.where(np.asarray(y_obs) == 0)[0]
    mis_idx = np.where(np.asarray(y_obs) == 1)[0]
    hist, best, best_state = [], -1, None
    for ep in range(epochs):
        G.train(); D.train(); np.random.shuffle(real_idx); dl, gl, t0 = [], [], time.time()
        for s in range(0, len(real_idx), bs):
            ix = torch.tensor(real_idx[s:s + bs], device=device)
            ei, eh = Ei_t[ix], Eh_t[ix]
            fake = G(ei).detach()
            lreal = F.binary_cross_entropy_with_logits(D(ei, eh), torch.ones(len(ix), device=device))
            lfake = F.binary_cross_entropy_with_logits(D(ei, fake), torch.zeros(len(ix), device=device))
            ld = lreal + lfake
            if sup and len(mis_idx):
                mj = torch.tensor(np.random.choice(mis_idx, size=min(len(ix), len(mis_idx))), device=device)
                ld = ld + F.binary_cross_entropy_with_logits(D(Ei_t[mj], Eh_t[mj]), torch.zeros(len(mj), device=device))
            oD.zero_grad(); ld.backward(); oD.step()
            lg = F.binary_cross_entropy_with_logits(D(ei, G(ei)), torch.ones(len(ix), device=device))
            oG.zero_grad(); lg.backward(); oG.step()
            dl.append(float(ld)); gl.append(float(lg))
        D.eval(); G.eval()
        with torch.no_grad():
            sv = (-D(torch.tensor(Ei_val, device=device, dtype=torch.float32), torch.tensor(Eh_val, device=device, dtype=torch.float32))).cpu().numpy()
            div = diversity(G(Ei_t[real_idx[:256]]))
        auc = roc_auc_score(y_val, sv)
        hist.append({"epoch": ep + 1, "d_loss": float(np.mean(dl)), "g_loss": float(np.mean(gl)), "val_auc": float(auc),
                     "gen_diversity_cos": div, "epoch_s": time.time() - t0})
        if auc > best:
            best, best_state = auc, {k: v.clone() for k, v in D.state_dict().items()}
    D.load_state_dict(best_state)
    return G, D, hist
