"""Back-to-back (differential) testing: full model vs distilled student vs INT8 ONNX student (CDAO 'Testing Methods')."""
import numpy as np
import pytest
from conftest import ART, record
from sift.infer import SiftScorer

pytestmark = pytest.mark.needs_student


def _load(name):
    if not (ART / name / "meta.json").exists():
        pytest.skip(f"artifact {name} not present")
    return SiftScorer(ART / name)


def _probs(s, df):
    i, h = list(df.consolidated_intent), [f"{p}\n{x}" for p, x in zip(df.file_path, df.hunk)]
    a, _, _ = s._logits(i, h)
    return 1 / (1 + np.exp(-a / s.T))


@pytest.fixture(scope="module")
def sample(data):
    return data["test"].sample(300, random_state=473)


def test_student_fp32_onnx_matches_student_torch(sample):
    st, so = _load("student"), _load("student_onnx_fp32")
    d = np.abs(_probs(st, sample) - _probs(so, sample))
    record("b2b_student_torch_vs_onnx_fp32", {"n": len(d), "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean())})
    assert d.max() < 0.02


def test_int8_agrees_with_fp32_student_verdicts(sample):
    so, s8 = _load("student_onnx_fp32"), _load("student_onnx_int8")
    p0, p1 = _probs(so, sample), _probs(s8, sample)
    agree = float(np.mean((p0 >= so.threshold) == (p1 >= so.threshold)))
    record("b2b_int8_vs_fp32_student", {"n": len(p0), "verdict_agreement": agree, "mean_abs_dp": float(np.mean(np.abs(p0 - p1))),
                                       "max_abs_dp": float(np.max(np.abs(p0 - p1)))})
    assert agree >= 0.90


def test_student_agrees_with_teacher_verdicts(sample):
    t, s = _load("full"), _load("student")
    p0, p1 = _probs(t, sample), _probs(s, sample)
    agree = float(np.mean((p0 >= t.threshold) == (p1 >= s.threshold)))
    corr = float(np.corrcoef(p0, p1)[0, 1])
    record("b2b_teacher_vs_student", {"n": len(p0), "verdict_agreement": agree, "prob_correlation": corr})
    assert agree >= 0.80


def test_rank_correlation_teacher_student_within_sessions(sample):
    from scipy.stats import spearmanr
    t, s = _load("full"), _load("student")
    rho = spearmanr(_probs(t, sample), _probs(s, sample)).statistic
    record("b2b_teacher_student_spearman", float(rho))
    assert rho > 0.5
