"""Phase 5a: rule filters + confident learning (cleanlab) on out-of-fold MiniLM probabilities, scored on the 569 flips."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.data import load_split, ROOT
from sift.denoise import rule_filter_mask, confident_learning_issues, score_flag_set

tr = load_split("train")
parts = [np.load(ROOT / f"experiments/runs/cl_oof_fold{f}/oof.npz", allow_pickle=True) for f in range(3)]
sid = np.concatenate([p["sample_id"] for p in parts]); prob = np.concatenate([p["prob"] for p in parts])
lab = np.concatenate([p["label"] for p in parts])
oof = dict(zip(sid, zip(prob, lab)))
prob = np.array([oof[s][0] for s in tr.sample_id]); lab = np.array([oof[s][1] for s in tr.sample_id])
assert (lab == (tr.label == "misaligned").astype(int).values).all()
flip = tr.label_noise.values
res = {"n_train": len(tr), "oof_auc_vs_observed": None}
from sklearn.metrics import roc_auc_score
res["oof_auc_vs_observed"] = float(roc_auc_score(lab, prob))
res["oof_auc_vs_clean"] = float(roc_auc_score((tr.label_clean == "misaligned").astype(int), prob))
rm = rule_filter_mask(tr)
res["rules"] = {"removed": int(rm.sum()), "non_semantic": int((tr.non_semantic_reason != "none").sum()),
                "duplicates_after_first": int(tr.duplicated("hunk_sha1", keep="first").sum()),
                "removed_that_are_label_flips": int((rm & flip).sum())}
res["confident_learning"] = {}
best = None
for mode in ("prune_by_noise_rate", "prune_by_class", "both", "confident_learning"):
    m = confident_learning_issues(lab, prob, filter_by=mode)
    sc = score_flag_set(m, flip); res["confident_learning"][mode] = sc
    if mode == "prune_by_noise_rate": best = m
# co-teaching is described/implemented in denoise.py but NOT run here (second option; not executed)
res["note"] = "Co-teaching helper implemented (sift.denoise) but not executed in this study."
# default list used for the final model: prune_by_noise_rate, restricted to rows not already removed by rules
drop = best & ~rm
Path(ROOT / "experiments/results/cl_drop_ids.txt").write_text("\n".join(tr.sample_id[drop]))
res["drop_list_size"] = int(drop.sum()); res["drop_list_vs_flips"] = score_flag_set(drop, flip)
json.dump(res, open(ROOT / "experiments/results/denoise.json", "w"), indent=2)
print(json.dumps(res, indent=1))
