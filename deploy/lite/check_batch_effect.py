"""INT8 dynamic quantisation makes a hunk's logit depend slightly on its batch mates. Compare lite (batch 2) vs default (batch 16)
over all curated samples. Run: PYTHONPATH=src .venv/bin/python deploy/lite/check_batch_effect.py"""
import os, json, numpy as np
from sift import serve, infer
sc = infer.SiftScorer("artifacts/student_onnx_int8")
d, f, n = [], 0, 0
for s in serve._load_samples().values():
    os.environ.pop("SIFT_LOW_MEM", None); a = sc.score(s["turns"], s["diff"])
    os.environ["SIFT_LOW_MEM"] = "1"; b = sc.score(s["turns"], s["diff"])
    pa = {h["index"]: h for h in a["hunks"]}
    for h in b["hunks"]:
        x = pa[h["index"]]; d.append(abs(x["p_misaligned"] - h["p_misaligned"])); n += 1; f += x["verdict"] != h["verdict"]
print(json.dumps({"hunks": n, "max_abs_dp": float(max(d)), "mean_abs_dp": float(np.mean(d)), "verdict_flips": int(f)}))
