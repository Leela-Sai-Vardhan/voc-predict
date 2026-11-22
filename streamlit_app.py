"""Streamlit lab debugger for VOC-Predict signal tuning.

Runs on port 8501. Purposes:
- Upload raw ZnO sensor data (CSV with time/resistance, or Excel with X/Y1/Y2/Y3).
- Tune Savitzky-Golay window, polynomial order, onset threshold.
- Visualize raw R(t), normalized S(t), detected onset, peak.
- Inspect extracted feature values.
- "Export" button writes a minimal debug record; does NOT auto-commit to training.

Usage: streamlit run streamlit_app.py --server.port 8501
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import pipeline

st.set_page_config(layout="wide", page_title="VOC-Predict Lab Debugger")

st.sidebar.title("VOC-Predict — Lab Debugger")
st.sidebar.caption("Port 8501 — read-only signal tuning; does not commit training data.")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(uploaded_file) -> pd.DataFrame:
    """Read uploaded CSV or Excel into a time/resistance DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        # Expect columns time, resistance (or auto-rename)
        if "time" in df.columns and "resistance" in df.columns:
            return df[["time", "resistance"]]
        # fallback: first two columns
        return df.iloc[:, :2].rename(columns={df.columns[0]: "time", df.columns[1]: "resistance"})
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
        # If X/Y1/Y2/Y3 present, map Y channels
        if all(c in df.columns for c in ["X", "Y1", "Y2", "Y3"]):
            st.sidebar.success("Excel with channels detected — Y1=ethanol, Y2=acetone, Y3=isopropanol")
            channel = st.sidebar.selectbox("Select channel", ["Y1 (ethanol)", "Y2 (acetone)", "Y3 (isopropanol)"])
            col = channel.split("(")[1].rstrip(")")
            signal = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            time = pd.to_numeric(df["X"], errors="coerce").fillna(0.0)
            return pd.DataFrame({"time": time.values, "resistance": signal.values})
        # fallback: first two columns
        return df.iloc[:, :2].rename(columns={df.columns[0]: "time", df.columns[1]: "resistance"})
    else:
        st.error("Unsupported file type. Upload .csv or .xlsx.")
        return pd.DataFrame({"time": [], "resistance": []})


def _run_pipeline(df: pd.DataFrame, window: int, poly: int, threshold_k: float):
    """Run pipeline.extract_features with given params; return features + plots."""
    s = pipeline.smooth_sg(df["resistance"].abs(), window, poly)
    start, _ = pipeline.detect_exposure(s, df, threshold_k=threshold_k)
    feats = pipeline.extract_features(df, window_s=5.0, sg_window=window, sg_poly=poly, threshold_k=threshold_k)

    # Build plots
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # Raw R(t)
    axes[0].plot(df["time"], df["resistance"], color="#e8eef7", lw=1, label="Raw R(t)")
    if start is not None:
        axes[0].axvline(df["time"].iloc[start], color="k", ls=":", lw=1, label="Onset")
    axes[0].set_ylabel("Resistance (Ω)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Raw resistance response R(t)")

    # Normalized S(t) = (R-R0)/R0  (pipeline does this internally; we compute for display)
    r0 = pipeline.baseline_r0(df, window_s=5.0)
    s_raw = (df["resistance"] - r0) / r0
    s_smooth = pipeline.smooth_sg(s_raw, window, poly)
    axes[1].plot(df["time"], s_raw, color="#8fbc8f", lw=0.8, alpha=0.5, label="S(t) raw")
    axes[1].plot(df["time"], s_smooth, color="#2c7bb6", lw=1.8, label=f"S(t) SG window={window}")
    if start is not None:
        axes[1].axvline(df["time"].iloc[start], color="k", ls=":", lw=1)
    # peak
    peak_idx = start + int(np.argmax(s_smooth.to_numpy()[start:])) if start is not None else None
    if peak_idx is not None:
        axes[1].scatter(df["time"].iloc[peak_idx], s_smooth.iloc[peak_idx], color="red", zorder=5)
    axes[1].set_ylabel("S(t) = (R−R0)/R0")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Normalized response S(t)")

    # Feature summary box
    axes[2].axis("off")
    axes[2].set_title("Extracted features")
    feat_text = f"""
    r0: {feats['r0']:.1f} Ω
    peak_response: {feats['peak_response']:.4f}
    response_time: {feats['response_time']:.2f} s
    recovery_time: {feats['recovery_time']:.2f if not math.isnan(feats['recovery_time']) else 'N/A'} s
    rise_rate: {feats['rise_rate']:.4f}
    decay_rate: {feats['decay_rate']:.4f}
    tau_ad_1: {feats['tau_ad_1']:.2f}
    tau_ad_2: {feats['tau_ad_2']:.2f}
    tau_de_1: {feats['tau_de_1']:.2f}
    tau_de_2: {feats['tau_de_2']:.2f}
    peak_time: {feats['peak_time']:.2f}
    """
    axes[2].text(0.1, 0.9, feat_text, fontsize=10, va="top")
    plt.tight_layout()

    return fig, feats


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.title("VOC-Predict — Lab Debugger")
st.caption("Tune Savitzky-Golay smoothing, onset threshold, and inspect transient features.")

with st.sidebar:
    st.header("1) Upload data")
    uploaded = st.file_uploader("Choose CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        df = _load_data(uploaded)
        st.session_state["df"] = df
        st.success(f"Loaded {len(df)} rows — columns: {list(df.columns)}")
    else:
        st.info("Upload a file to begin.")

    if "df" in st.session_state:
        df = st.session_state["df"]
        st.header("2) Signal parameters")
        window = st.slider("Savitzky-Golay window (odd ≥ poly+1)", 5, 51, 11, 2)
        poly = st.slider("Polynomial order", 1, min(4, window - 1), 3)
        threshold_k = st.slider("Onset detection threshold k", 1.0, 5.0, 3.0, 0.1)
        st.caption(f"window={window} (must be odd); poly ≤ window-1")

st.header("3) Signal debug view")
if "df" in st.session_state:
    df = st.session_state["df"]
    fig, feats = _run_pipeline(df, window, poly, threshold_k)
    st.pyplot(fig, clear_figure=True)

    st.header("4) Feature values")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("peak_response", f"{feats['peak_response']:.3f}")
        st.metric("rise_rate", f"{feats['rise_rate']:.3f}")
    with col2:
        st.metric("tau_ad_1", f"{feats['tau_ad_1']:.2f}")
        st.metric("tau_de_1", f"{feats['tau_de_1']:.2f}")
    with col3:
        st.metric("response_time", f"{feats['response_time']:.2f}" if not math.isnan(feats['response_time']) else "N/A")
        st.metric("tau_ad_2", f"{feats['tau_ad_2']:.2f}")

    st.header("5) Export debug record")
    if st.button("Export debug JSONL record"):
        rec = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "n_rows": len(df),
            "window": window,
            "poly": poly,
            "threshold_k": threshold_k,
            "features": {k: (float(v) if not (isinstance(v, float) and math.isnan(v)) else None) for k, v in feats.items()},
        }
        log_path = "debug_runs.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        st.info(f"Appended debug record to {log_path}")


st.caption("""
**How to use:**
1. Upload the raw `Data Set.xlsx` (or a CSV with `time` + `resistance` columns).
2. Use the sidebar sliders to tune the Savitzky-Golay window and onset threshold.
3. The plots update instantly — verify the onset index isn't mistaking a desorption edge for the gas injection.
4. Click **Export debug JSONL** to append a minimal record; this does **not** modify `extracted_features.csv` or `model.joblib`.
5. When the signal looks clean, return to `segment_and_extract.py` or `full_retrain.py` with confidence.
""")