"""ONNX export, dynamic INT8 quantisation and CPU latency benchmarking."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import torch


class _Export(torch.nn.Module):
    def __init__(self, m, use_tt):
        super().__init__(); self.m, self.use_tt = m, use_tt

    def forward(self, input_ids, attention_mask):
        o = self.m(input_ids, attention_mask)
        return o["align_logit"], o["drift_logits"], o["cal_logit"]


def export_onnx(model, out_path: Path, example_len=64, opset=17):
    model = model.cpu().eval()
    ids = torch.randint(5, 1000, (2, example_len)); am = torch.ones_like(ids)
    torch.onnx.export(_Export(model, False), (ids, am), str(out_path),
                      input_names=["input_ids", "attention_mask"], output_names=["align_logit", "drift_logits", "cal_logit"],
                      dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                    "align_logit": {0: "b"}, "drift_logits": {0: "b"}, "cal_logit": {0: "b"}},
                      opset_version=opset, dynamo=False)
    return out_path


def quantize_int8(fp32_path: Path, int8_path: Path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    return int8_path


def size_mb(p: Path) -> float:
    return os.path.getsize(p) / 1e6


def cpu_latency(session_fn, inputs_list, warmup=5, runs=None):
    """Per-request latency (ms) for single-pair inference; returns p50/p95/mean."""
    for x in inputs_list[:warmup]:
        session_fn(x)
    ts = []
    for x in (inputs_list if runs is None else inputs_list[:runs]):
        t = time.perf_counter(); session_fn(x); ts.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)), "mean_ms": float(np.mean(ts)), "n": len(ts)}


def cpu_info() -> dict:
    name = "unknown"
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                name = l.split(":", 1)[1].strip(); break
    except OSError:
        pass
    return {"cpu": name, "logical_cores": os.cpu_count()}
