"""Training / prediction for Sift cross-encoders (full model, ablation variants, MiniLM student)."""
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import SEED
from .data import load_split, hunk_input, raw_intent, split_turns, consolidate_intent, ROOT
from .losses import multitask_loss, distill_loss
from .model import SiftModel, load_tokenizer, DRIFT_CLASSES, DEFAULT_ENCODER
from .rank import full_eval, best_threshold

DRIFT_IDX = {d: i for i, d in enumerate(DRIFT_CLASSES)}


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_texts(df, input_mode="full", intent_source="consolidated"):
    """Returns (text_a, text_b). input_mode: full | hunk_only | intent_only."""
    if intent_source == "consolidated":
        intents = list(df.consolidated_intent)
    elif intent_source == "raw":
        intents = [raw_intent(r) for _, r in df.iterrows()]
    elif intent_source == "rule":
        intents = [consolidate_intent(split_turns(t)) for t in df.intent_turns]
    else:
        raise ValueError(intent_source)
    hunks = [hunk_input(r) for _, r in df.iterrows()]
    if input_mode == "hunk_only":
        return hunks, None
    if input_mode == "intent_only":
        return intents, None
    return intents, hunks


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, df, tok, max_len, input_mode="full", intent_source="consolidated", labels="label"):
        a, b = build_texts(df, input_mode, intent_source)
        kw = dict(truncation="longest_first", max_length=max_len)
        enc = tok(a, b, **kw) if b is not None else tok(a, **kw)
        self.ids = enc["input_ids"]
        self.tt = enc.get("token_type_ids")
        self.y = (df[labels] == "misaligned").astype(int).values
        dcol = "drift_type" if labels == "label" else "drift_type_clean"
        self.d = np.array([DRIFT_IDX.get(x, -1) for x in df[dcol]])
        self.pad = tok.pad_token_id

    def __len__(self): return len(self.ids)

    def __getitem__(self, i): return i

    def collate(self, idx):
        L = max(len(self.ids[i]) for i in idx)
        ids = torch.full((len(idx), L), self.pad); am = torch.zeros((len(idx), L), dtype=torch.long)
        tt = torch.zeros((len(idx), L), dtype=torch.long)
        for k, i in enumerate(idx):
            n = len(self.ids[i]); ids[k, :n] = torch.tensor(self.ids[i]); am[k, :n] = 1
            if self.tt is not None: tt[k, :n] = torch.tensor(self.tt[i])
        return {"input_ids": ids, "attention_mask": am, "token_type_ids": tt,
                "y": torch.tensor(self.y[idx]), "d": torch.tensor(self.d[idx]), "idx": torch.tensor(idx)}


def bucketed_batches(ds, bs, rng):
    """Shuffle, then sort within chunks of 50 batches by length to cut padding."""
    order = rng.permutation(len(ds)).tolist()
    chunk = bs * 50
    batches = []
    for s in range(0, len(order), chunk):
        c = sorted(order[s:s + chunk], key=lambda i: len(ds.ids[i]))
        batches += [c[j:j + bs] for j in range(0, len(c), bs)]
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def predict(model, ds, device, bs=32, amp=True) -> dict:
    model.eval()
    order = sorted(range(len(ds)), key=lambda i: len(ds.ids[i]))
    P = {"align_logit": np.zeros(len(ds), np.float32), "drift_probs": np.zeros((len(ds), 3), np.float32),
         "cal_prob": np.zeros(len(ds), np.float32)}
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        b = ds.collate(idx)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            o = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["token_type_ids"].to(device))
        P["align_logit"][idx] = o["align_logit"].float().cpu().numpy()
        P["drift_probs"][idx] = torch.softmax(o["drift_logits"].float(), -1).cpu().numpy()
        P["cal_prob"][idx] = torch.sigmoid(o["cal_logit"].float()).cpu().numpy()
    P["prob"] = 1 / (1 + np.exp(-P["align_logit"]))
    return P


def select_train_rows(tr, rule_filter: bool, drop_file: str | None):
    n0 = len(tr); info = {"n_train_raw": n0}
    keep = np.ones(n0, bool)
    if rule_filter:
        ns = (tr.non_semantic_reason != "none").values
        dup = tr.duplicated("hunk_sha1", keep="first").values
        keep &= ~(ns | dup)
        info.update(rule_removed_non_semantic=int(ns.sum()), rule_removed_duplicates=int((dup & ~ns).sum()))
    if drop_file:
        ids = set(Path(drop_file).read_text().split())
        m = tr.sample_id.isin(ids).values
        keep &= ~m
        info["cl_removed"] = int(m.sum())
    info["n_train_used"] = int(keep.sum())
    return tr[keep].reset_index(drop=True), info


def run(args) -> dict:
    set_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    out_dir = ROOT / "experiments" / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    tr, sel_info = select_train_rows(tr, args.rule_filter, args.drop_file)
    held = None
    if args.nfolds:
        from sklearn.model_selection import GroupKFold
        folds = list(GroupKFold(n_splits=args.nfolds).split(tr, groups=tr.session_id))
        tr_i, ho_i = folds[args.fold]
        held, tr = tr.iloc[ho_i].reset_index(drop=True), tr.iloc[tr_i].reset_index(drop=True)
    if args.subset:
        tr = tr.sample(min(args.subset, len(tr)), random_state=SEED).reset_index(drop=True)
        va = va.sample(min(args.subset, len(va)), random_state=SEED).reset_index(drop=True)
        te = te.sample(min(args.subset, len(te)), random_state=SEED).reset_index(drop=True)
    if args.train_subset:
        tr = tr.sample(min(args.train_subset, len(tr)), random_state=SEED).reset_index(drop=True)
    tok = load_tokenizer(args.encoder)
    mk = lambda d, lab: PairDataset(d, tok, args.max_len, args.input_mode, args.intent_source, lab)
    dtr, dva, dte = mk(tr, "label"), mk(va, "label_clean"), mk(te, "label_clean")
    model = SiftModel(args.encoder, gradient_checkpointing=args.grad_ckpt).to(dev)
    use_drift, use_cal = (not args.no_multitask), (not args.no_multitask)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(dtr) / (args.micro_bs * args.accum))
    total = steps_per_epoch * args.epochs; warm = int(0.1 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    teacher = None
    if args.teacher_logits:
        z = np.load(args.teacher_logits, allow_pickle=True)
        lut = dict(zip(z["sample_id"], z["train_logit"]))
        teacher = torch.tensor(np.array([lut[i] for i in tr.sample_id], dtype=np.float32))
    rng = np.random.RandomState(args.seed)
    cfg = {**vars(args), **sel_info, "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "torch": torch.__version__, "python": __import__("platform").python_version(),
           "transformers": __import__("transformers").__version__, "numpy": np.__version__,
           "cuda": torch.version.cuda, "dtype": "bf16 autocast" if dev == "cuda" else "fp32", "params": model.n_params(), "effective_batch": args.micro_bs * args.accum,
           "n_val": len(va), "n_test": len(te)}
    json.dump(cfg, open(out_dir / "config.json", "w"), indent=2, default=str)
    best, bad, history, t0 = -1.0, 0, [], time.time()
    yva = dva.y
    ck_dir = ROOT / "checkpoints" / args.name; ck_dir.mkdir(parents=True, exist_ok=True)
    start_ep = 0
    if args.resume and (ck_dir / "last.pt").exists():      # resume after an interruption (epoch granularity)
        ck = torch.load(ck_dir / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        rng.set_state(ck["rng"]); best, bad, history, start_ep = ck["best"], ck["bad"], ck["history"], ck["epoch"]
        t0 -= history[-1]["elapsed_s"] if history else 0
        print(f"resumed from epoch {start_ep}", flush=True)
    for ep in range(start_ep, args.epochs):
        model.train(); losses = []
        batches = bucketed_batches(dtr, args.micro_bs, rng)
        for bi, idx in enumerate(batches):
            b = dtr.collate(idx)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                o = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["token_type_ids"].to(dev))
            o = {k: v.float() for k, v in o.items()}
            y, d = b["y"].to(dev), b["d"].to(dev)
            if teacher is not None:
                loss = distill_loss(o["align_logit"], teacher[b["idx"]].to(dev), y, args.tau, args.alpha)
                m = d >= 0
                if m.any():   # keep the drift-type head useful in the student
                    loss = loss + 0.3 * torch.nn.functional.cross_entropy(o["drift_logits"][m], d[m])
                parts = {"total": float(loss)}
            else:
                loss, parts = multitask_loss(o, y, d, use_focal=not args.ce, use_drift=use_drift, use_cal=use_cal)
            (loss / args.accum).backward()
            losses.append(parts["total"])
            if (bi + 1) % args.accum == 0 or bi == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if args.log_every and bi % args.log_every == 0:
                print(f"ep{ep} {bi}/{len(batches)} loss {np.mean(losses[-args.log_every:]):.4f} {time.time()-t0:.0f}s", flush=True)
        Pva = predict(model, dva, dev)
        thr = best_threshold(yva, Pva["prob"])
        from sklearn.metrics import f1_score
        f1 = f1_score(yva, Pva["prob"] >= thr, average="macro")
        history.append({"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val_macro_f1": float(f1), "val_threshold": thr,
                        "elapsed_s": time.time() - t0})
        print(history[-1], flush=True)
        if f1 > best:
            best, bad = f1, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
        tmp = ck_dir / "last.pt.tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "rng": rng.get_state(),
                    "best": best, "bad": bad, "history": history, "epoch": ep + 1}, tmp)
        os.replace(tmp, ck_dir / "last.pt")
        if bad >= args.patience: break
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=dev))
    Pva, Pte = predict(model, dva, dev), predict(model, dte, dev)
    thr = best_threshold(yva, Pva["prob"])
    res = {"name": args.name, "config": cfg, "history": history, "best_val_macro_f1": best, "train_time_s": time.time() - t0,
           "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if dev == "cuda" else None,
           "test": full_eval(Pte["prob"], dte.y, list(te.session_id), list(te.mutation_difficulty), thr),
           "val": full_eval(Pva["prob"], dva.y, list(va.session_id), list(va.mutation_difficulty), thr)}
    np.savez(out_dir / "preds.npz", val_logit=Pva["align_logit"], test_logit=Pte["align_logit"],
             val_drift=Pva["drift_probs"], test_drift=Pte["drift_probs"], val_cal=Pva["cal_prob"], test_cal=Pte["cal_prob"],
             val_y=dva.y, test_y=dte.y, val_ids=va.sample_id.values, test_ids=te.sample_id.values)
    if held is not None:
        Pho = predict(model, mk(held, "label"), dev)
        np.savez(out_dir / "oof.npz", sample_id=held.sample_id.values, prob=Pho["prob"], label=(held.label == "misaligned").astype(int).values,
                 flip=held.label_noise.values)
    if args.dump_train_logits:
        dall = mk(load_split("train"), "label")
        full_tr = load_split("train")
        np.savez(out_dir / "train_logits.npz", sample_id=full_tr.sample_id.values, train_logit=predict(model, dall, dev)["align_logit"])
    json.dump(res, open(out_dir / "results.json", "w"), indent=2, default=float)
    return res


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--micro_bs", type=int, default=8)
    p.add_argument("--accum", type=int, default=2)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--input_mode", default="full", choices=["full", "hunk_only", "intent_only"])
    p.add_argument("--intent_source", default="consolidated", choices=["consolidated", "raw", "rule"])
    p.add_argument("--ce", action="store_true", help="plain cross-entropy instead of focal loss")
    p.add_argument("--no_multitask", action="store_true")
    p.add_argument("--rule_filter", action="store_true", help="drop non-semantic and duplicate train hunks")
    p.add_argument("--drop_file", default=None, help="file of sample_ids to drop (confident-learning output)")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--train_subset", type=int, default=0, help="subsample train rows only (val/test stay full and in CSV order)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--resume", action="store_true", help="resume from checkpoints/<name>/last.pt")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--teacher_logits", default=None)
    p.add_argument("--tau", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--nfolds", type=int, default=0)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--dump_train_logits", action="store_true")
    return p


if __name__ == "__main__":
    r = run(parser().parse_args())
    print(json.dumps({k: r["test"][k] for k in ("macro_f1", "ndcg@5", "pairwise_auc")}, indent=1, default=float))
