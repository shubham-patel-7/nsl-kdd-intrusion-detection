"""Inference helpers: read a traffic CSV and classify it with a trained model."""
from __future__ import annotations

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .models import TorchMLP


def read_traffic(source) -> pd.DataFrame:
    """Read NSL-KDD style records from a path / bytes / file-like object.

    Accepts a CSV with a header row, or a raw NSL-KDD .txt/.csv with no header (41, 42 or 43 columns:
    41 features [+ label] [+ difficulty]). Returns a frame containing the 41 feature columns, plus
    `attack` if a label column was present.
    """
    if hasattr(source, "read"):
        raw = source.read()
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        raw = Path(source).read_bytes()
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    if not set(C.FEATURE_COLS) <= set(df.columns):
        df = pd.read_csv(io.BytesIO(raw), header=None, low_memory=False)
        if df.shape[1] not in (41, 42, 43):
            raise ValueError(
                f"Expected a header row with the 41 NSL-KDD feature names, or a headerless file with 41-43 "
                f"columns; got {df.shape[1]} columns."
            )
        df.columns = C.ALL_COLS[: df.shape[1]]
    if "label" in df.columns:
        df = df.rename(columns={"label": "attack"})
    missing = [c for c in C.FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    for c in C.NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[C.NUMERIC].isna().any().any():
        bad = df[C.NUMERIC].columns[df[C.NUMERIC].isna().any()].tolist()
        raise ValueError(f"Non-numeric or missing values in: {bad[:5]}")
    keep = C.FEATURE_COLS + (["attack"] if "attack" in df.columns else [])
    return df[keep].reset_index(drop=True)


class Detector:
    """Loads the preprocessor and any trained model listed in models/manifest.json."""

    def __init__(self, model_dir: Path = C.MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.manifest = json.loads((self.model_dir / "manifest.json").read_text())
        self.pre = joblib.load(self.model_dir / self.manifest["preprocessor"])
        self._cache: dict = {}

    def available(self, task: str) -> list:
        return [n for n, f in self.manifest["tasks"][task]["models"].items() if (self.model_dir / f).exists()]

    def classes(self, task: str) -> list:
        return self.manifest["tasks"][task]["classes"]

    def _model(self, task: str, name: str):
        key = (task, name)
        if key not in self._cache:
            path = self.model_dir / self.manifest["tasks"][task]["models"][name]
            self._cache[key] = TorchMLP.load(path) if path.suffix == ".pt" else joblib.load(path)
        return self._cache[key]

    def predict(self, df: pd.DataFrame, task: str = "binary", model: str = "XGBoost") -> pd.DataFrame:
        X = self.pre.transform(df[C.FEATURE_COLS]).astype(np.float32)
        proba = self._model(task, model).predict_proba(X)
        classes = self.classes(task)
        out = pd.DataFrame({
            "prediction": [classes[i] for i in proba.argmax(1)],
            "confidence": proba.max(1),
        })
        if task == "binary":
            out["p_attack"] = proba[:, 1]
        return out
