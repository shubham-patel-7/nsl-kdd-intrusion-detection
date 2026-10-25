"""Train and evaluate all models on NSL-KDD (binary + 5-class).

Usage:  python -m ids.train            # full run (~10-15 min on a laptop CPU)
        python -m ids.train --quick    # 10 % subsample smoke run
"""
from __future__ import annotations

import argparse
import json
import re
import time

import joblib
import numpy as np
import torch

from . import config as C
from .data import build_preprocessor, class_weights, make_splits
from .evaluate import evaluate, mean_std, ms_per_1000
from .models import MODEL_NAMES, STOCHASTIC, TorchMLP, fit_model

BINARY_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate"]
MULTI_KEYS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_model(name: str, task: str, model) -> str:
    base = C.MODEL_DIR / f"{task}_{slug(name)}"
    if isinstance(model, TorchMLP):
        path = base.with_suffix(".pt")
        model.save(path)
    else:
        path = base.with_suffix(".joblib")
        joblib.dump(model, path, compress=3)
    return path.name


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=C.SEEDS)
    args = ap.parse_args(argv)
    torch.set_num_threads(2)
    C.MODEL_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)

    tr, va, te = make_splits(seed=0)
    if args.quick:
        tr, va, te = (d.sample(frac=0.1, random_state=0).reset_index(drop=True) for d in (tr, va, te))
        args.seeds = args.seeds[:1]
    pre = build_preprocessor().fit(tr[C.FEATURE_COLS])
    Xtr, Xva, Xte = (pre.transform(d[C.FEATURE_COLS]).astype(np.float32) for d in (tr, va, te))
    joblib.dump(pre, C.MODEL_DIR / "preprocessor.joblib", compress=3)
    names = list(pre.get_feature_names_out())
    print(f"features: {Xtr.shape[1]}  train {len(tr)}  val {len(va)}  test {len(te)}")

    known = set(tr["attack"]) | set(va["attack"])
    is_attack = (te["attack"] != "normal").to_numpy()
    known_mask = is_attack & te["attack"].isin(known).to_numpy()
    novel_mask = is_attack & ~te["attack"].isin(known).to_numpy()
    results = {
        "data": {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "n_features": int(Xtr.shape[1]),
                 "test_known_attacks": int(known_mask.sum()), "test_novel_attacks": int(novel_mask.sum()),
                 "novel_attack_types": sorted(set(te["attack"][novel_mask])),
                 "family_counts": {s: d["family"].value_counts().to_dict() for s, d in
                                   [("train", tr), ("val", va), ("test", te)]}},
        "tasks": {},
    }
    manifest = {"preprocessor": "preprocessor.joblib", "tasks": {}}

    for task, col, classes in [("binary", "binary", C.CLASSES_BINARY), ("multiclass", "family", C.CLASSES_MULTI)]:
        idx = {c: i for i, c in enumerate(classes)}
        ytr, yva, yte = (d[col].map(idx).to_numpy() for d in (tr, va, te))
        weights = class_weights(ytr, len(classes))
        keys = BINARY_KEYS if task == "binary" else MULTI_KEYS
        task_res, probas = {}, {"y_test": yte}
        manifest["tasks"][task] = {"classes": classes, "models": {}}
        for name in MODEL_NAMES:
            seeds = args.seeds if name in STOCHASTIC else [0]
            runs = []
            for s in seeds:
                t0 = time.time()
                model = fit_model(name, Xtr, ytr, Xva, yva, len(classes), s, weights)
                fit_s = time.time() - t0
                p_va, p_te = model.predict_proba(Xva), model.predict_proba(Xte)
                run = {"seed": s, "fit_seconds": fit_s, "val": evaluate(yva, p_va, classes),
                       "test": evaluate(yte, p_te, classes)}
                if s == seeds[0]:  # seed 0 is the representative model we persist and plot
                    run["infer_ms_per_1000"] = ms_per_1000(model, Xte[:5000])
                    probas[slug(name)] = p_te.astype(np.float32)
                    manifest["tasks"][task]["models"][name] = save_model(name, task, model)
                    size = (C.MODEL_DIR / manifest["tasks"][task]["models"][name]).stat().st_size / 1e6
                    run["model_size_mb"] = size
                    if task == "binary":
                        pred_attack = p_te.argmax(1) == 1
                        run["detection_known"] = float(pred_attack[known_mask].mean()) if known_mask.any() else None
                        run["detection_novel"] = float(pred_attack[novel_mask].mean()) if novel_mask.any() else None
                    if isinstance(model, TorchMLP):
                        (C.RESULTS_DIR / f"mlp_history_{task}.json").write_text(json.dumps(model.history))
                    if name == "XGBoost":
                        imp = sorted(zip(names, model.feature_importances_.tolist()), key=lambda t: -t[1])[:20]
                        (C.RESULTS_DIR / f"xgb_importance_{task}.json").write_text(json.dumps(imp))
                runs.append(run)
                print(f"[{task}] {name:20s} seed {s}  test {keys[3]}={run['test'][keys[3]]:.4f}  "
                      f"val {keys[3]}={run['val'][keys[3]]:.4f}  fit {fit_s:.0f}s")
            task_res[name] = {
                "n_seeds": len(runs),
                "val": {k: mean_std([r["val"][k] for r in runs]) for k in keys},
                "test": {k: mean_std([r["test"][k] for r in runs]) for k in keys},
                "fit_seconds": mean_std([r["fit_seconds"] for r in runs]),
                "representative": runs[0],  # seed 0: confusion matrix, per-class, timing, size
            }
        results["tasks"][task] = task_res
        np.savez_compressed(C.RESULTS_DIR / f"test_proba_{task}.npz", **probas)

    (C.MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print("\nSaved models/ and results/metrics.json")


if __name__ == "__main__":
    main()
