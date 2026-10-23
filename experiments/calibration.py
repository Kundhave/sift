"""Phase 5a: temperature scaling of the full model on validation; ECE before/after on test; reliability diagram."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import ROOT
from sift.calibrate import fit_temperature, apply_temperature, reliability_bins, calibration_report
from sift.rank import ece

d = np.load(ROOT / "experiments/runs/full/preds.npz", allow_pickle=True)
vl, vy, tl, ty = d["val_logit"], d["val_y"], d["test_logit"], d["test_y"]
rep = calibration_report(vl, vy, tl, ty)
T = rep["temperature"]
rep["ece_val_before"] = ece(vy, apply_temperature(vl, 1.0)); rep["ece_val_after"] = ece(vy, apply_temperature(vl, T))
rep["reliability_before"] = reliability_bins(ty, apply_temperature(tl, 1.0))
rep["model"] = "full (ModernBERT-base)"; rep["fit_on"] = "validation (clean labels)"; rep["eval_on"] = "test"
rep["ece_bins"] = 15; rep["n_val"] = int(len(vy)); rep["n_test"] = int(len(ty))
json.dump(rep, open(ROOT / "results/calibration.json", "w"), indent=2)
fig, ax = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
for a, key, title in ((ax[0], "reliability_before", f"Before (T=1)  ECE={rep['ece_test_before']:.3f}"),
                      (ax[1], "reliability_after", f"After (T={T:.2f})  ECE={rep['ece_test_after']:.3f}")):
    b = [x for x in rep[key] if x["n"] > 0]
    a.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    a.plot([x["mean_conf"] for x in b], [x["frac_pos"] for x in b], "o-", color="#2a6f97")
    a.set_title(title, fontsize=9); a.set_xlabel("mean predicted P(misaligned)"); a.grid(alpha=.3)
ax[0].set_ylabel("observed fraction misaligned")
fig.suptitle("Reliability diagram, test split, full model", fontsize=10); fig.tight_layout()
fig.savefig(ROOT / "results/calibration_reliability.png", dpi=160)
print({k: rep[k] for k in ("temperature", "ece_test_before", "ece_test_after", "ece_val_before", "ece_val_after", "nll_test_before", "nll_test_after")})
