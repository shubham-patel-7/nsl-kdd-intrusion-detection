"""Download, load, label and preprocess NSL-KDD."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config as C


def download(data_dir: Path = C.DATA_DIR) -> None:
    """Fetch KDDTrain+ and KDDTest+ if missing."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in C.DATA_URLS.items():
        dest = data_dir / name
        if dest.exists():
            continue
        print(f"downloading {name} ...")
        with urllib.request.urlopen(url, timeout=120) as r:
            dest.write_bytes(r.read())


def load_raw(split: str, data_dir: Path = C.DATA_DIR) -> pd.DataFrame:
    """Load 'train' (KDDTrain+) or 'test' (KDDTest+) with attack names and labels attached."""
    name = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}[split]
    path = data_dir / name
    if not path.exists():
        download(data_dir)
    df = pd.read_csv(path, header=None, names=C.ALL_COLS)
    unknown = set(df["label"]) - set(C.ATTACK_FAMILY)
    if unknown:
        raise ValueError(f"Unmapped attack names: {sorted(unknown)}")
    df = df.rename(columns={"label": "attack"}).drop(columns="difficulty")
    df["family"] = df["attack"].map(C.ATTACK_FAMILY)
    df["binary"] = np.where(df["attack"] == "normal", "normal", "attack")
    return df


def _log1p(x):
    return np.log1p(np.clip(x, 0, None))


def build_preprocessor() -> ColumnTransformer:
    """log1p the heavy-tailed columns, standardise numerics, one-hot the categoricals.

    `handle_unknown='ignore'` matters: KDDTest+ contains `service` values never seen in training.
    """
    plain_numeric = [c for c in C.NUMERIC if c not in C.LOG_COLS]
    return ColumnTransformer(
        [
            ("log", Pipeline([("log1p", FunctionTransformer(_log1p, feature_names_out="one-to-one")), ("scale", StandardScaler())]), C.LOG_COLS),
            ("num", StandardScaler(), plain_numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Square-root inverse-frequency weights: help rare classes without wrecking precision."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = (len(y) / (n_classes * np.maximum(counts, 1))) ** 0.5
    return w / w.mean()


def make_splits(seed: int = 0):
    """Official split: KDDTrain+ is cut 80/20 (stratified) into train/val; KDDTest+ is untouched."""
    train_all = load_raw("train")
    test = load_raw("test")
    tr, va = train_test_split(train_all, test_size=C.VAL_FRACTION, stratify=train_all["attack"],
                              random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True), test


if __name__ == "__main__":
    download()
    print("NSL-KDD ready in", C.DATA_DIR)
