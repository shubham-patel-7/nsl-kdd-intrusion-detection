"""Upload network-traffic records (CSV) and classify them as normal / attack.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ids import config as C
from ids.predict import Detector, read_traffic

st.set_page_config(page_title="Intrusion detection", page_icon="🛡️", layout="wide")
BLUE, ORANGE, INK2, GRID = "#2a78d6", "#eb6834", "#52514e", "#e6e5e1"
SAMPLE = ROOT / "sample_data" / "sample_traffic.csv"
BENCH = ROOT / "app" / "bench.html"


@st.cache_resource
def get_detector():
    return Detector()


def bench_tab():
    """Standalone in-browser page: pick a KDDTest+ record and watch five classifiers label it."""
    st.caption("Pick a real KDDTest+ connection record and compare five classifiers side by side. "
               "This page runs in your browser and uses its own models, separate from the Detector tab.")
    if not BENCH.exists():
        st.warning("The bench page (app/bench.html) is missing from this copy of the repo.")
        return
    components.html(BENCH.read_text(encoding="utf-8"), height=1500, scrolling=True)


def main():
    st.title("🛡️ Network intrusion detector")
    st.caption("Classifies NSL-KDD connection records with models trained in this repo (classic ML vs deep learning).")
    tab_detector, tab_bench = st.tabs(["Detector", "Bench"])
    with tab_bench:
        bench_tab()
    with tab_detector:
        detector_tab()


def detector_tab():
    try:
        det = get_detector()
    except FileNotFoundError:
        st.error("No trained models found. Run `python -m ids.train` first.")
        st.stop()

    st.sidebar.header("Settings")
    task_label = st.sidebar.radio("Task", ["Normal vs attack", "Attack family (5 classes)"])
    task = "binary" if task_label.startswith("Normal") else "multiclass"
    model = st.sidebar.selectbox("Model", det.available(task), index=min(2, len(det.available(task)) - 1))
    thr = None
    if task == "binary":
        thr = st.sidebar.slider("Alert threshold on P(attack)", 0.05, 0.95, 0.50, 0.05,
                                help="Lower = catch more attacks but raise more false alarms.")

    up = st.file_uploader("Upload a CSV of connection records", type=["csv", "txt"])
    use_sample = st.checkbox("…or use the bundled sample (500 KDDTest+ records)", value=up is None)
    source = up if up is not None else (SAMPLE if use_sample and SAMPLE.exists() else None)
    if source is None:
        st.info("Upload a file to begin. Expected: the 41 NSL-KDD feature columns (header optional); a label column is optional.")
        st.stop()

    try:
        df = read_traffic(source)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    res = det.predict(df, task, model)
    if task == "binary":
        res["prediction"] = np.where(res["p_attack"] >= thr, "attack", "normal")
    out = pd.concat([df.reset_index(drop=True), res], axis=1)

    n = len(out)
    flagged = int((out["prediction"] != "normal").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Records", f"{n:,}")
    c2.metric("Flagged as attack", f"{flagged:,}  ({flagged / n:.1%})")
    c3.metric("Model", model)

    left, right = st.columns(2)
    with left:
        st.subheader("Predictions")
        counts = out["prediction"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(counts.index, counts.values, color=[BLUE if i == "normal" else ORANGE for i in counts.index], width=0.55)
        for i, v in enumerate(counts.values):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Records")
        ax.grid(axis="y", color=GRID)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    with right:
        if "attack" in out.columns:
            st.subheader("Check against the labels you supplied")
            truth = out["attack"].map(C.ATTACK_FAMILY)
            if task == "binary":
                truth = np.where(truth == "normal", "normal", "attack")
            truth = pd.Series(truth)
            acc = float((truth == out["prediction"]).mean())
            st.metric("Accuracy on this file", f"{acc:.1%}")
            st.dataframe(pd.crosstab(truth.rename("actual"), out["prediction"].rename("predicted")))
            if truth.isna().any():
                st.caption("Some labels were not recognised NSL-KDD attack names and were ignored in the table.")
        else:
            st.subheader("No label column")
            st.caption("Add a `label` column (e.g. `neptune`, `normal`) to see accuracy and a confusion table.")

    st.subheader("Most suspicious records")
    sort_col = "p_attack" if task == "binary" else "confidence"
    show_cols = ["prediction", sort_col, "protocol_type", "service", "flag", "src_bytes", "dst_bytes"] + (
        ["attack"] if "attack" in out.columns else [])
    top = out[out["prediction"] != "normal"].sort_values(sort_col, ascending=False)
    st.dataframe(top[show_cols].head(200).round(3), hide_index=True, use_container_width=True, height=300)
    st.download_button("Download all predictions (CSV)", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

    with st.expander("About this tool"):
        st.markdown(
            """
Trained on **NSL-KDD** (KDDTrain+). Real attacks in the held-out KDDTest+ set include types never seen in training,
so expect lower recall on novel attacks than on known ones. See the README for measured numbers.
This is a research/teaching model and is **not** a production IDS.
"""
        )


main()
