import json, os, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SIFT_DEVICE", "cpu")   # tests run on CPU (deployment target; leaves the GPU to training jobs)
sys.path.insert(0, str(ROOT / "src"))
ART = ROOT / "artifacts"
MEAS = ROOT / "experiments" / "results" / "testing_measurements.json"
_meas: dict = {}


def record(key, value):
    """Store a measured quantity (e.g. a metamorphic violation rate) for the report builder."""
    _meas[key] = value


def pytest_sessionfinish(session, exitstatus):
    if _meas:
        old = {}
        if MEAS.exists():
            try: old = json.load(open(MEAS))
            except Exception: old = {}
        old.update(_meas)
        MEAS.parent.mkdir(parents=True, exist_ok=True)
        json.dump(old, open(MEAS, "w"), indent=2, default=float)


@pytest.fixture(scope="session")
def tiny_model():
    """Random-init 2-layer BERT cross-encoder (structure tests only; no downloads)."""
    import torch
    from transformers import BertConfig, BertModel
    from sift.model import SiftModel
    torch.manual_seed(0)
    enc = BertModel(BertConfig(vocab_size=1000, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=64, max_position_embeddings=128))
    return SiftModel("tiny", encoder_module=enc).eval()


@pytest.fixture(scope="session")
def data():
    from sift.data import load_all
    return load_all()


@pytest.fixture(scope="session")
def scorer():
    p = ART / "full"
    if not (p / "meta.json").exists():
        pytest.skip("trained artifact artifacts/full not present")
    from sift.infer import SiftScorer
    return SiftScorer(p)


@pytest.fixture(scope="session")
def test_samples(data):
    """Deterministic evaluation subset of the test split: 60 misaligned + 60 aligned."""
    t = data["test"]
    return t[t.label_clean == "misaligned"].sample(60, random_state=473).append(t[t.label_clean == "aligned"].sample(60, random_state=473)) \
        if hasattr(t, "append") else __import__("pandas").concat([t[t.label_clean == "misaligned"].sample(60, random_state=473),
                                                                  t[t.label_clean == "aligned"].sample(60, random_state=473)])
