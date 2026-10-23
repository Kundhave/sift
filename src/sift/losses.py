"""Focal loss and the weighted multi-task objective  L = 0.6 L_align + 0.3 L_drift + 0.1 L_cal."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float | None = None,
               reduction: str = "mean") -> torch.Tensor:
    """Binary focal loss on logits. gamma=0 and alpha=None reduces to plain BCE-with-logits."""
    t = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p_t = torch.exp(-bce)
    loss = (1 - p_t) ** gamma * bce
    if alpha is not None:
        loss = loss * (alpha * t + (1 - alpha) * (1 - t))
    return loss.mean() if reduction == "mean" else loss.sum() if reduction == "sum" else loss


def multitask_loss(out: dict, align_y: torch.Tensor, drift_y: torch.Tensor, *, use_focal=True, gamma=2.0,
                   use_drift=True, use_cal=True, weights=(0.6, 0.3, 0.1)) -> tuple[torch.Tensor, dict]:
    """out: {'align_logit':(B,), 'drift_logits':(B,3), 'cal_logit':(B,)}.
    drift_y is -1 for rows that are not misaligned (excluded from the drift term).
    Calibration target: whether the alignment decision (threshold 0.5) is correct, detached from the align head."""
    w_a, w_d, w_c = weights
    la = focal_loss(out["align_logit"], align_y, gamma) if use_focal else \
        F.binary_cross_entropy_with_logits(out["align_logit"], align_y.float())
    total, parts = w_a * la, {"align": la.detach().item()}
    if use_drift:
        m = drift_y >= 0
        ld = F.cross_entropy(out["drift_logits"][m], drift_y[m]) if m.any() else out["align_logit"].sum() * 0
        total = total + w_d * ld; parts["drift"] = float(ld.detach())
    if use_cal:
        correct = ((out["align_logit"].detach() > 0).long() == align_y).float()
        lc = F.binary_cross_entropy_with_logits(out["cal_logit"], correct)
        total = total + w_c * lc; parts["cal"] = float(lc.detach())
    parts["total"] = float(total.detach())
    return total, parts


def distill_loss(student_logit, teacher_logit, align_y, tau=4.0, alpha=0.5):
    """Binary distillation: alpha * CE(hard) + (1-alpha) * tau^2 * KL(teacher_soft || student_soft)."""
    hard = F.binary_cross_entropy_with_logits(student_logit, align_y.float())
    ps = torch.sigmoid(student_logit / tau).clamp(1e-6, 1 - 1e-6)
    pt = torch.sigmoid(teacher_logit / tau).clamp(1e-6, 1 - 1e-6)
    kl = pt * (pt / ps).log() + (1 - pt) * ((1 - pt) / (1 - ps)).log()
    return alpha * hard + (1 - alpha) * tau ** 2 * kl.mean()
