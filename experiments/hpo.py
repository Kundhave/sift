"""Small Optuna (TPE) study for the ModernBERT cross-encoder: lr, loss weights, max length.
Scaled down on purpose: train subset (default 2400 rows, denoised), validation subset (default 1200), 2 epochs, ~8 trials.
Objective = validation macro-F1 (threshold tuned on the validation subset). The test split is never loaded here.
Resumable (sqlite storage). Output: results/hpo.json."""
import argparse, gc, json, math, sys, time
from pathlib import Path
import numpy as np, optuna, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from sift import SEED
from sift.data import load_split
from sift.losses import multitask_loss
from sift.model import SiftModel, load_tokenizer, DEFAULT_ENCODER
from sift.rank import best_threshold
from sift.train import PairDataset, bucketed_batches, predict, select_train_rows, set_seed
from sklearn.metrics import f1_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=8); p.add_argument("--n_train", type=int, default=2400); p.add_argument("--n_val", type=int, default=1200)
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--cpu", action="store_true"); p.add_argument("--name", default="hpo")
    p.add_argument("--max_steps", type=int, default=0, help="debug: stop each trial after N micro-batches")
    p.add_argument("--encoder", default=DEFAULT_ENCODER); p.add_argument("--lens", default="256,384,512", help="candidate max lengths")
    a = p.parse_args()
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    out = ROOT / "experiments/runs" / a.name; out.mkdir(parents=True, exist_ok=True)
    tr, _ = select_train_rows(load_split("train"), True, str(ROOT / "experiments/results/cl_drop_ids.txt"))
    tr = tr.sample(min(a.n_train, len(tr)), random_state=SEED).reset_index(drop=True)
    va = load_split("val").sample(a.n_val, random_state=SEED).reset_index(drop=True) if a.n_val < 2700 else load_split("val")
    tok = load_tokenizer(a.encoder)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
        L = trial.suggest_categorical("max_len", [int(x) for x in a.lens.split(",")])
        w_a = trial.suggest_float("w_align", 0.4, 0.8); w_d = trial.suggest_float("w_drift", 0.1, 0.4); w_c = trial.suggest_float("w_cal", 0.02, 0.2)
        set_seed(SEED)
        dtr, dva = PairDataset(tr, tok, L, labels="label"), PairDataset(va, tok, L, labels="label_clean")
        model = SiftModel(a.encoder, gradient_checkpointing=True).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        mb, acc = 8, 2; total = math.ceil(len(dtr) / (mb * acc)) * a.epochs; warm = max(1, int(0.1 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
        rng = np.random.RandomState(SEED); n = 0; t0 = time.time()
        try:
            for ep in range(a.epochs):
                model.train(); bs = bucketed_batches(dtr, mb, rng)
                for bi, idx in enumerate(bs):
                    b = dtr.collate(idx)
                    with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                        o = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["token_type_ids"].to(dev))
                    loss, _ = multitask_loss({k: v.float() for k, v in o.items()}, b["y"].to(dev), b["d"].to(dev), weights=(w_a, w_d, w_c))
                    (loss / acc).backward(); n += 1
                    if (bi + 1) % acc == 0 or bi == len(bs) - 1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                    if a.max_steps and n >= a.max_steps: break
                if a.max_steps and n >= a.max_steps: break
            pv = predict(model, dva, dev, bs=8)["prob"]
            thr = best_threshold(dva.y, pv); f1 = float(f1_score(dva.y, pv >= thr, average="macro"))
        except torch.cuda.OutOfMemoryError:
            raise optuna.TrialPruned()
        finally:
            del model, opt; gc.collect()
            if dev == "cuda": torch.cuda.empty_cache()
        trial.set_user_attr("val_threshold", thr); trial.set_user_attr("seconds", time.time() - t0)
        return f1

    study = optuna.create_study(study_name=a.name, direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                storage=f"sqlite:///{out}/study.db", load_if_exists=True)
    todo = a.trials - len([t for t in study.trials if t.state.is_finished()])
    if todo > 0: study.optimize(objective, n_trials=todo)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    res = {"objective": "validation macro-F1 on a val subset (test never used)", "sampler": "TPE", "seed": SEED, "planned_trials_brief": 60, "trials_run": len(study.trials),
           "n_train": len(tr), "n_val": len(va), "epochs": a.epochs, "encoder": a.encoder, "device": dev, "debug_max_steps": a.max_steps,
           "best": ({"value": study.best_value, "params": study.best_params} if done else None),
           "trials": [{"number": t.number, "state": t.state.name, "value": t.value, "params": t.params, "seconds": t.user_attrs.get("seconds")} for t in study.trials]}
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(res, open(ROOT / "results" / (a.name + ".json"), "w"), indent=2, default=float)
    print(json.dumps(res["best"]))

if __name__ == "__main__":
    main()
