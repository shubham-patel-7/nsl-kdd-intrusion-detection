"""Figures and markdown tables from saved results.   Usage: python -m ids.report"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from sklearn.metrics import roc_curve

from . import config as C

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# one fixed colour per model, used in every figure (colour follows the entity)
MCOLOR = {"XGBoost": "#2a78d6", "Random Forest": "#eb6834", "MLP (PyTorch)": "#1baf7a", "Logistic Regression": "#eda100"}
ORDER = ["XGBoost", "Random Forest", "MLP (PyTorch)", "Logistic Regression"]
SHORT = {"XGBoost": "XGBoost", "Random Forest": "Random\nForest", "MLP (PyTorch)": "MLP\n(PyTorch)", "Logistic Regression": "Logistic\nReg."}
GREY, DARK = "#b9b8b0", "#2a2a28"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 10.5,
        "axes.titlelocation": "left", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 9.5, "legend.frameon": False, "figure.dpi": 150, "lines.linewidth": 2,
    })


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def _bars(ax, vals, errs, colors, fmt="{:.3f}", ylim=1.0):
    bars = ax.bar(range(len(vals)), vals, yerr=errs, color=colors, width=0.62, ecolor=INK2, capsize=3,
                  edgecolor=SURFACE, linewidth=1.5)
    for i, v in enumerate(vals):
        ax.text(i, v + (errs[i] or 0) + ylim * 0.015, fmt.format(v), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(vals)))
    ax.grid(axis="x", visible=False)
    return bars


def fig_binary_metrics(R):
    t = R["tasks"]["binary"]
    keys = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2), sharey=True)
    for ax, (k, title) in zip(axes, keys):
        vals = [t[m]["test"][k]["mean"] for m in ORDER]
        errs = [t[m]["test"][k]["std"] for m in ORDER]
        _bars(ax, vals, errs, [MCOLOR[m] for m in ORDER])
        ax.set_xticklabels([SHORT[m] for m in ORDER], fontsize=8)
        ax.set_title(title)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("KDDTest+ (attack = positive)")
    fig.tight_layout()
    _save(fig, "binary_metrics.png")


def _heat(ax, cm, classes, title):
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:,}\n{norm[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_title(title)
    ax.grid(False)


def fig_confusion_binary(R):
    t = R["tasks"]["binary"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, m in zip(axes, ORDER):
        _heat(ax, t[m]["representative"]["test"]["confusion_matrix"], C.CLASSES_BINARY, m)
    axes[0].set_ylabel("Actual")
    fig.tight_layout()
    _save(fig, "confusion_binary.png")


def fig_confusion_multi(R):
    t = R["tasks"]["multiclass"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, m in zip(axes, ["XGBoost", "MLP (PyTorch)"]):
        _heat(ax, t[m]["representative"]["test"]["confusion_matrix"], C.CLASSES_MULTI, m)
    axes[0].set_ylabel("Actual (cell = count and share of the actual class)")
    fig.tight_layout()
    _save(fig, "confusion_multiclass.png")


def fig_roc(R):
    P = np.load(C.RESULTS_DIR / "test_proba_binary.npz")
    y = P["y_test"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    for m in ORDER:
        s = m.lower().replace(" ", "_").replace("(", "").replace(")", "")
        key = "mlp_pytorch" if "mlp" in s else s
        fpr, tpr, _ = roc_curve(y, P[key][:, 1])
        auc = R["tasks"]["binary"][m]["representative"]["test"]["roc_auc"]
        ax.plot(fpr, tpr, color=MCOLOR[m], label=f"{m} (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on KDDTest+ (seed 0)")
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "roc_curves.png")


def fig_generalization(R):
    t = R["tasks"]["binary"]
    x = np.arange(len(ORDER))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    v = [t[m]["val"]["f1"]["mean"] for m in ORDER]
    s = [t[m]["test"]["f1"]["mean"] for m in ORDER]
    ax.bar(x - 0.2, v, 0.38, color=GREY, label="Validation (same distribution as training)", edgecolor=SURFACE, linewidth=1.5)
    ax.bar(x + 0.2, s, 0.38, color=DARK, label="KDDTest+ (contains unseen attack types)", edgecolor=SURFACE, linewidth=1.5)
    for i in range(len(ORDER)):
        ax.text(x[i] - 0.2, v[i] + 0.01, f"{v[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, s[i] + 0.01, f"{s[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in ORDER])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Binary F1")
    ax.set_title("Same models, very different scores: distribution shift")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "generalization_gap.png")


def fig_novel(R):
    t = R["tasks"]["binary"]
    x = np.arange(len(ORDER))
    kn = [t[m]["representative"]["detection_known"] for m in ORDER]
    nv = [t[m]["representative"]["detection_novel"] for m in ORDER]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - 0.2, kn, 0.38, color=GREY, label="Attack types seen in training", edgecolor=SURFACE, linewidth=1.5)
    ax.bar(x + 0.2, nv, 0.38, color=DARK, label="Attack types never seen in training", edgecolor=SURFACE, linewidth=1.5)
    for i in range(len(ORDER)):
        ax.text(x[i] - 0.2, kn[i] + 0.01, f"{kn[i]:.0%}", ha="center", fontsize=8.5)
        ax.text(x[i] + 0.2, nv[i] + 0.01, f"{nv[i]:.0%}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[m] for m in ORDER])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Share of attacks detected")
    ax.set_title("Detection rate: known vs novel attacks (KDDTest+)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "novel_attacks.png")


def fig_speed(R):
    t = R["tasks"]["binary"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for m in ORDER:
        r = t[m]["representative"]
        f1 = t[m]["test"]["f1"]["mean"]
        ax.scatter(r["infer_ms_per_1000"], f1, s=90, color=MCOLOR[m], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(m, (r["infer_ms_per_1000"], f1), textcoords="offset points", xytext=(8, 5), fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Inference time per 1,000 records (ms, log scale)")
    ax.set_ylabel("Test F1 (binary)")
    ax.set_title("Accuracy vs. speed")
    fig.tight_layout()
    _save(fig, "speed_vs_accuracy.png")


def fig_per_class(R):
    t = R["tasks"]["multiclass"]
    x = np.arange(len(C.CLASSES_MULTI))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for i, m in enumerate(ORDER):
        f = [t[m]["representative"]["test"]["per_class"][c]["f1"] for c in C.CLASSES_MULTI]
        ax.bar(x + (i - 1.5) * w, f, w * 0.92, color=MCOLOR[m], label=m, edgecolor=SURFACE, linewidth=1.2)
    ax.set_xticks(x)
    sup = t["XGBoost"]["representative"]["test"]["per_class"]
    ax.set_xticklabels([f"{c}\n(n={sup[c]['support']:,})" for c in C.CLASSES_MULTI])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-class F1 on KDDTest+")
    ax.set_title("Rare attack families (R2L, U2R) are the hard part")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8.5)
    fig.tight_layout()
    _save(fig, "per_class_f1.png")


def fig_importance():
    imp = json.loads((C.RESULTS_DIR / "xgb_importance_binary.json").read_text())[:15][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.barh([n for n, _ in imp], [v for _, v in imp], color=MCOLOR["XGBoost"], height=0.62, edgecolor=SURFACE)
    ax.set_xlabel("XGBoost feature importance (gain share)")
    ax.set_title("What the best tree model looks at")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "xgb_importance.png")


def tables(R) -> str:
    out = []
    t = R["tasks"]["binary"]
    out += ["### Binary (normal vs attack) - KDDTest+, mean ± std over seeds", "",
            "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | ms / 1k rec | Size (MB) |", "|---|---|---|---|---|---|---|---|---|"]
    for m in ORDER:
        r = t[m]
        f = lambda k: f"{r['test'][k]['mean']:.3f} ± {r['test'][k]['std']:.3f}" if r["n_seeds"] > 1 else f"{r['test'][k]['mean']:.3f}"
        out.append(f"| {m} | {f('accuracy')} | {f('precision')} | {f('recall')} | {f('f1')} | {f('roc_auc')} | {f('false_positive_rate')} | "
                   f"{r['representative']['infer_ms_per_1000']:.1f} | {r['representative']['model_size_mb']:.2f} |")
    t = R["tasks"]["multiclass"]
    out += ["", "### 5-class (normal / DoS / Probe / R2L / U2R) - KDDTest+, mean ± std over seeds", "",
            "| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |", "|---|---|---|---|---|---|"]
    for m in ORDER:
        r = t[m]
        f = lambda k: f"{r['test'][k]['mean']:.3f} ± {r['test'][k]['std']:.3f}" if r["n_seeds"] > 1 else f"{r['test'][k]['mean']:.3f}"
        out.append(f"| {m} | {f('accuracy')} | {f('precision_macro')} | {f('recall_macro')} | {f('f1_macro')} | {f('f1_weighted')} |")
    b = R["tasks"]["binary"]
    out += ["", "### Generalisation (binary F1, seed 0 / mean) and novel-attack detection", "",
            "| Model | Val F1 | Test F1 | Detection: known attacks | Detection: novel attacks |", "|---|---|---|---|---|"]
    for m in ORDER:
        r = b[m]
        out.append(f"| {m} | {r['val']['f1']['mean']:.3f} | {r['test']['f1']['mean']:.3f} | "
                   f"{r['representative']['detection_known']:.1%} | {r['representative']['detection_novel']:.1%} |")
    return "\n".join(out)


def main():
    _style()
    R = json.loads((C.RESULTS_DIR / "metrics.json").read_text())
    for fn in (fig_binary_metrics, fig_confusion_binary, fig_confusion_multi, fig_roc, fig_generalization,
               fig_novel, fig_speed, fig_per_class):
        fn(R)
    fig_importance()
    md = tables(R)
    (C.RESULTS_DIR / "results_tables.md").write_text(md + "\n")
    print(md)
    print(f"\nFigures written to {C.FIG_DIR}")


if __name__ == "__main__":
    main()
