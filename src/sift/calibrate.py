"""Temperature scaling (Guo et al., 2017) fitted on validation logits, and reliability statistics."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from .rank import ece


def _nll(logit, y, T):
    z = np.asarray(logit, float) / T
    return float(np.mean(np.logaddexp(0, z) - y * z))


def fit_temperature(val_logit, val_y) -> float:
    y = np.asarray(val_y, float)
    r = minimize_scalar(lambda lt: _nll(val_logit, y, np.exp(lt)), bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def apply_temperature(logit, T):
    return 1 / (1 + np.exp(-np.asarray(logit, float) / T))


def reliability_bins(y, p, n_bins=10):
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    return [{"bin": b, "n": int((idx == b).sum()), "mean_conf": float(p[idx == b].mean()) if (idx == b).any() else None,
             "frac_pos": float(y[idx == b].mean()) if (idx == b).any() else None} for b in range(n_bins)]


def calibration_report(val_logit, val_y, test_logit, test_y):
    T = fit_temperature(val_logit, val_y)
    pb, pa = apply_temperature(test_logit, 1.0), apply_temperature(test_logit, T)
    return {"temperature": T, "ece_test_before": ece(test_y, pb), "ece_test_after": ece(test_y, pa),
            "nll_test_before": _nll(test_logit, np.asarray(test_y, float), 1.0),
            "nll_test_after": _nll(test_logit, np.asarray(test_y, float), T),
            "reliability_after": reliability_bins(test_y, pa)}
