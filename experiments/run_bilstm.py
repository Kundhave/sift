"""Siamese BiLSTM baseline: shared embedding + BiLSTM encodes intent and hunk separately (MiniLM wordpiece vocab,
embeddings trained from scratch); features [u, v, |u-v|, u*v] -> MLP -> alignment logit. Same splits, same denoised train rows,
threshold on validation, writes runs/<name>/{results.json,preds.npz,config.json,best.pt}."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from sift import SEED
from sift.data import load_split, hunk_input
from sift.model import load_tokenizer, STUDENT_ENCODER
from sift.rank import full_eval, best_threshold
from sift.train import select_train_rows, set_seed
from sklearn.metrics import f1_score


class Net(nn.Module):
    def __init__(self, V, E=128, H=128):
        super().__init__()
        self.emb = nn.Embedding(V, E, padding_idx=0); self.rnn = nn.LSTM(E, H, batch_first=True, bidirectional=True)
        self.mlp = nn.Sequential(nn.Linear(8 * H, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 1))
    def enc(self, x):
        m = (x != 0).unsqueeze(-1)
        h, _ = self.rnn(self.emb(x))
        return h.masked_fill(~m, -1e4).max(1).values
    def forward(self, a, b):
        u, v = self.enc(a), self.enc(b)
        return self.mlp(torch.cat([u, v, (u - v).abs(), u * v], -1)).squeeze(-1)


def tokenise(tok, texts, L):
    ids = tok(list(texts), truncation=True, max_length=L, add_special_tokens=False)["input_ids"]
    X = np.zeros((len(ids), L), np.int64)
    for i, s in enumerate(ids): X[i, :len(s)] = np.maximum(s, 1) if len(s) else 0
    return torch.tensor(X)


def prep(df, tok, la, lb):
    return tokenise(tok, df.consolidated_intent, la), tokenise(tok, [hunk_input(r) for _, r in df.iterrows()], lb), \
        torch.tensor((df.label_clean == "misaligned").values.astype(np.float32))


@torch.no_grad()
def logits(net, A, B, dev, bs=128):
    net.eval(); return np.concatenate([net(A[i:i+bs].to(dev), B[i:i+bs].to(dev)).float().cpu().numpy() for i in range(0, len(A), bs)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="bilstm"); p.add_argument("--epochs", type=int, default=15); p.add_argument("--patience", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--len_a", type=int, default=64); p.add_argument("--len_b", type=int, default=384)
    p.add_argument("--subset", type=int, default=0); p.add_argument("--max_steps", type=int, default=0); p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    set_seed(SEED); dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    out = ROOT / "experiments/runs" / a.name; out.mkdir(parents=True, exist_ok=True)
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    tr, info = select_train_rows(tr, True, str(ROOT / "experiments/results/cl_drop_ids.txt"))
    if a.subset:
        tr, va, te = [d.sample(min(a.subset, len(d)), random_state=SEED).reset_index(drop=True) for d in (tr, va, te)]
    tok = load_tokenizer(STUDENT_ENCODER)
    Atr, Btr, _ = prep(tr, tok, a.len_a, a.len_b); ytr = torch.tensor((tr.label == "misaligned").values.astype(np.float32))   # train on (noisy) label
    Ava, Bva, yva = prep(va, tok, a.len_a, a.len_b); Ate, Bte, yte = prep(te, tok, a.len_a, a.len_b)
    net = Net(tok.vocab_size).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0, device=dev))
    cfg = {**vars(a), **info, "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None, "params": sum(q.numel() for q in net.parameters()),
           "model": "siamese BiLSTM (emb128, hid128, max-pool), BCE pos_weight 3, intent source consolidated", "seed": SEED, "torch": torch.__version__}
    json.dump(cfg, open(out / "config.json", "w"), indent=2, default=str)
    rng = np.random.RandomState(SEED); best, bad, hist, t0, step = -1, 0, [], time.time(), 0
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(len(Atr)); ls = []
        for i in range(0, len(perm), a.bs):
            ix = perm[i:i + a.bs]
            loss = lossf(net(Atr[ix].to(dev), Btr[ix].to(dev)), ytr[ix].to(dev))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); ls.append(float(loss)); step += 1
            if a.max_steps and step >= a.max_steps: break
        lv = logits(net, Ava, Bva, dev); pv = 1 / (1 + np.exp(-lv)); thr = best_threshold(yva.numpy().astype(int), pv)
        f1 = f1_score(yva.numpy().astype(int), pv >= thr, average="macro")
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(ls)), "val_macro_f1": float(f1), "elapsed_s": time.time() - t0}); print(hist[-1], flush=True)
        if f1 > best: best, bad = f1, 0; torch.save(net.state_dict(), out / "best.pt")
        else: bad += 1
        if bad >= a.patience or (a.max_steps and step >= a.max_steps): break
    net.load_state_dict(torch.load(out / "best.pt", map_location=dev))
    lv, lt = logits(net, Ava, Bva, dev), logits(net, Ate, Bte, dev)
    pv, pt = 1 / (1 + np.exp(-lv)), 1 / (1 + np.exp(-lt)); yv, yt = yva.numpy().astype(int), yte.numpy().astype(int); thr = best_threshold(yv, pv)
    res = {"name": a.name, "config": cfg, "history": hist, "best_val_macro_f1": best, "train_time_s": time.time() - t0,
           "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None,
           "test": full_eval(pt, yt, list(te.session_id), list(te.mutation_difficulty), thr),
           "val": full_eval(pv, yv, list(va.session_id), list(va.mutation_difficulty), thr)}
    np.savez(out / "preds.npz", val_logit=lv, test_logit=lt, val_y=yv, test_y=yt, val_ids=va.sample_id.values, test_ids=te.sample_id.values)
    json.dump(res, open(out / "results.json", "w"), indent=2, default=float)
    print({k: res["test"][k] for k in ("macro_f1", "ndcg@5")})


if __name__ == "__main__":
    main()
