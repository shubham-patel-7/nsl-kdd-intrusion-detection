"""Metrics, confusion matrices and timing."""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score)


def evaluate(y_true: np.ndarray, proba: np.ndarray, classes: list) -> dict:
    """Metrics for binary (positive class = index 1 = 'attack') or multiclass problems."""
    y_pred = proba.argmax(1)
    k = len(classes)
    out: dict = {"accuracy": float(accuracy_score(y_true, y_pred))}
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))
    out["confusion_matrix"] = cm.tolist()
    if k == 2:
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
        out.update(precision=float(p[0]), recall=float(r[0]), f1=float(f[0]),
                   roc_auc=float(roc_auc_score(y_true, proba[:, 1])))
        tn, fp = cm[0, 0], cm[0, 1]
        out["false_positive_rate"] = float(fp / max(tn + fp, 1))
    else:
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(k)), zero_division=0)
        out.update(precision_macro=float(p.mean()), recall_macro=float(r.mean()), f1_macro=float(f.mean()),
                   f1_weighted=float(precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]))
        out["per_class"] = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                            for i, c in enumerate(classes)}
    return out


def ms_per_1000(model, X, repeats: int = 5) -> float:
    """Median wall-clock inference time per 1,000 samples (model only, features already built)."""
    reps = max(1, int(np.ceil(5000 / len(X))))
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(reps):
            model.predict_proba(X)
        times.append((time.perf_counter() - t0) / (reps * len(X)) * 1000 * 1000)
    return float(np.median(times))


def mean_std(values: list) -> dict:
    a = np.asarray(values, float)
    return {"mean": float(a.mean()), "std": float(a.std())}
