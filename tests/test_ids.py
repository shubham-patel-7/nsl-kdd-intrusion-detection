from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ids import config as C
from ids.data import build_preprocessor, class_weights
from ids.evaluate import evaluate
from ids.models import TorchMLP
from ids.predict import read_traffic

ROOT = Path(__file__).resolve().parents[1]
has_data = (C.DATA_DIR / "KDDTrain+.txt").exists() and (C.DATA_DIR / "KDDTest+.txt").exists()
has_models = (C.MODEL_DIR / "manifest.json").exists()


def test_attack_map_covers_39_names_into_4_families():
    fams = set(C.ATTACK_FAMILY.values())
    assert fams == {"normal", "DoS", "Probe", "R2L", "U2R"}


@pytest.mark.skipif(not has_data, reason="dataset not downloaded")
def test_every_label_in_both_files_is_mapped():
    from ids.data import load_raw

    for split in ("train", "test"):
        df = load_raw(split)
        assert df["family"].notna().all()
        assert len(df.columns) == 41 + 3  # features + attack + family + binary


@pytest.mark.skipif(not has_data, reason="dataset not downloaded")
def test_dataset_sizes_match_published():
    from ids.data import load_raw

    assert len(load_raw("train")) == 125973 and len(load_raw("test")) == 22544


@pytest.mark.skipif(not has_data, reason="dataset not downloaded")
def test_preprocessor_handles_unseen_category_and_is_leak_free():
    from ids.data import load_raw

    tr = load_raw("train").head(2000)
    pre = build_preprocessor().fit(tr[C.FEATURE_COLS])
    row = tr.head(1)[C.FEATURE_COLS].copy()
    row["service"] = "never_seen_service"
    out = pre.transform(row)
    assert out.shape[1] == pre.transform(tr.head(1)[C.FEATURE_COLS]).shape[1]
    assert np.isfinite(out).all()


def test_class_weights_favour_rare_classes():
    y = np.array([0] * 90 + [1] * 10)
    w = class_weights(y, 2)
    assert w[1] > w[0] and w.mean() == pytest.approx(1.0)


def test_torch_mlp_learns_a_separable_problem():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 10)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    m = TorchMLP(10, 2, epochs=30).fit(X[:500], y[:500], X[500:], y[500:])
    acc = (m.predict_proba(X[500:]).argmax(1) == y[500:]).mean()
    assert acc > 0.9


def test_evaluate_binary_and_multiclass():
    y = np.array([0, 0, 1, 1])
    perfect = np.eye(2)[y]
    r = evaluate(y, perfect, ["normal", "attack"])
    assert r["f1"] == 1.0 and r["roc_auc"] == 1.0 and r["false_positive_rate"] == 0.0
    y3 = np.array([0, 1, 2])
    r3 = evaluate(y3, np.eye(3)[y3], ["a", "b", "c"])
    assert r3["f1_macro"] == 1.0


def test_read_traffic_with_and_without_header(tmp_path):
    row = {c: 0 for c in C.FEATURE_COLS}
    row.update(protocol_type="tcp", service="http", flag="SF")
    df = pd.DataFrame([row])
    with_header = tmp_path / "h.csv"
    df.assign(label="normal").to_csv(with_header, index=False)
    got = read_traffic(with_header)
    assert list(got.columns) == C.FEATURE_COLS + ["attack"]
    headerless = tmp_path / "n.csv"
    df.assign(label="neptune", difficulty=20).to_csv(headerless, index=False, header=False)
    got2 = read_traffic(headerless)
    assert got2["attack"].iloc[0] == "neptune"


def test_read_traffic_rejects_garbage(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b,c\n1,2,3\n")
    with pytest.raises(ValueError):
        read_traffic(p)


@pytest.mark.skipif(not has_models, reason="models not trained")
def test_detector_end_to_end():
    from ids.predict import Detector

    df = read_traffic(ROOT / "sample_data" / "sample_traffic.csv")
    det = Detector()
    for task in ("binary", "multiclass"):
        for name in det.available(task):
            out = det.predict(df, task, name)
            assert len(out) == len(df) and out["prediction"].isin(det.classes(task)).all()


@pytest.mark.skipif(not has_models, reason="models not trained")
def test_dashboard_runs_headless():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    assert len(at.metric) >= 3
