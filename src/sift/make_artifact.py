"""Package a trained run into a deployable artifact (state_dict + meta with temperature and threshold)."""
import argparse, json, shutil
from pathlib import Path
import numpy as np
from .calibrate import fit_temperature
from .data import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run, out = ROOT / "experiments/runs" / a.run, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(run / "config.json")); z = np.load(run / "preds.npz", allow_pickle=True)
    T = fit_temperature(z["val_logit"], z["val_y"])
    from .rank import best_threshold
    thr = best_threshold(z["val_y"], 1 / (1 + np.exp(-z["val_logit"] / T)))
    shutil.copy(run / "best.pt", out / "model.pt")
    json.dump({"encoder": cfg["encoder"], "max_len": cfg["max_len"], "temperature": T, "threshold": thr, "backend": "torch",
               "source_run": a.run}, open(out / "meta.json", "w"), indent=2)
    print(out, T, thr)


if __name__ == "__main__":
    main()
