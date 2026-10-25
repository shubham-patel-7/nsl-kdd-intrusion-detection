"""Three demo runs of the 'upload a CSV' flow, using the saved models.

Builds three upload files from KDDTest+ (never used for training) and classifies each exactly as the
Streamlit app does (read_traffic -> Detector.predict):

  1. mixed          - 1,000 random records (realistic mix of normal and attack traffic)
  2. known_attacks  - 500 attack records of types that DO appear in the training data
  3. novel_attacks  - 500 attack records of types that do NOT appear in the training data

    python scripts/demo_runs.py        # needs data/KDDTest+.txt (python -m ids.data), writes sample_data/demo_*.csv
                                       # and results/demo_runs.json / demo_runs.md
"""
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids import config as C  # noqa: E402
from ids.data import make_splits  # noqa: E402
from ids.predict import Detector, read_traffic  # noqa: E402

MODELS = ["XGBoost", "MLP (PyTorch)"]


def main():
    tr, va, te = make_splits(seed=0)
    known = set(tr["attack"]) | set(va["attack"])
    atk = te[te["attack"] != "normal"]
    batches = {
        "1. Mixed traffic (1,000 random records)": te.sample(1000, random_state=1),
        "2. Known attack types only (500)": atk[atk["attack"].isin(known)].sample(500, random_state=2),
        "3. Novel attack types only (500)": atk[~atk["attack"].isin(known)].sample(500, random_state=3),
    }
    det = Detector()
    rows, details = [], {}
    for (title, df), fname in zip(batches.items(), ["demo_mixed.csv", "demo_known_attacks.csv", "demo_novel_attacks.csv"]):
        path = ROOT / "sample_data" / fname
        df[C.FEATURE_COLS].to_csv(path, index=False)          # the file a user would upload (no labels)
        up = read_traffic(path)
        truth_attack = (df["attack"] != "normal").to_numpy()
        for m in MODELS:
            t0 = time.perf_counter()
            out = det.predict(up, "binary", m)
            ms = (time.perf_counter() - t0) * 1000
            flagged = (out["prediction"] == "attack").to_numpy()
            row = {"batch": title, "model": m, "n": len(df), "attacks_in_batch": int(truth_attack.sum()),
                   "flagged": int(flagged.sum()), "flagged_share": float(flagged.mean()),
                   "mean_confidence": float(out["confidence"].mean()), "ms": ms}
            if truth_attack.any():
                row["detection_rate"] = float(flagged[truth_attack].mean())
            if (~truth_attack).any():
                row["false_alarm_rate"] = float(flagged[~truth_attack].mean())
            rows.append(row)
            if "Mixed" in title and m == "XGBoost":     # which attack families slip through?
                fam = df["family"].to_numpy()
                details["mixed_detection_by_family_xgboost"] = {
                    f: {"n": int((fam == f).sum()), "detected": float(flagged[fam == f].mean())}
                    for f in ["DoS", "Probe", "R2L", "U2R"] if (fam == f).any()}
            if "Novel" in title and m == "XGBoost":
                names = df["attack"].to_numpy()
                details["novel_detection_by_attack_xgboost"] = {
                    a: {"n": int((names == a).sum()), "detected": float(flagged[names == a].mean())}
                    for a in sorted(set(names)) if (names == a).sum() >= 15}

    (C.RESULTS_DIR / "demo_runs.json").write_text(json.dumps({"rows": rows, **details}, indent=2))
    L = ["| Demo run | Model | Records | Flagged as attack | Detection rate (of real attacks) | False-alarm rate (of normal) | Mean confidence |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        d = f"{r['detection_rate']:.1%}" if "detection_rate" in r else "-"
        fa = f"{r['false_alarm_rate']:.1%}" if "false_alarm_rate" in r else "-"
        L.append(f"| {r['batch']} | {r['model']} | {r['n']:,} | {r['flagged']:,} ({r['flagged_share']:.1%}) | {d} | {fa} | {r['mean_confidence']:.2f} |")
    md = "\n".join(L)
    (C.RESULTS_DIR / "demo_runs.md").write_text(md + "\n")
    print(md)
    print(json.dumps(details, indent=1))


if __name__ == "__main__":
    main()
