"""Extract dynamic transient features from 8 concentration pulses across 3 VOCs
on a single ZnO sensor. The dataset (Data Set.xlsx) contains 451 rows × 5 cols
(X, Y1, Y2, Y3) with 8 exposure/recovery pulses per channel.

Key constraints (per your guidance):
- distance=40 for find_peaks (array indices, not X-axis values)
- prominence=0.1 * np.max(signal) (handles baseline drift)
- concentration_idx is categorical metadata — DROPPED from the XGBoost feature matrix
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import pipeline

CHANNEL_MAP = {"Y1": "ethanol", "Y2": "acetone", "Y3": "isopropanol"}
CHANNELS = ["Y1", "Y2", "Y3"]
N_PULSES = 8  # 8 concentration levels per gas
WINDOW_SEC = 15  # baseline window before rising edge (seconds)
SAMPLES_PER_SEC = 1.0  # dataset acquired at ~1 Hz


def extract_segment(signal, peak_idx, X, peaks, i):
    """Extract the i-th pulse segment and return normalized (R-R0)/R0."""
    # --- start_idx: baseline trough before this peak ---
    search_start = peaks[i-1] if i > 0 else 0
    start_idx = signal[search_start:peak_idx].argmin() + search_start

    # --- end_idx: baseline trough after this peak ---
    search_end = peaks[i+1] if i < len(peaks)-1 else len(signal)
    end_idx = signal[peak_idx:search_end].argmin() + peak_idx

    # --- R0: median of the 15-s window immediately preceding start_idx ---
    window_samples = int(WINDOW_SEC * SAMPLES_PER_SEC)
    r0_start = max(0, start_idx - window_samples)
    baseline_window = signal[r0_start:start_idx]
    if len(baseline_window) == 0:
        baseline_window = signal[:start_idx] if start_idx > 0 else signal[:max(1, peak_idx // 4)]
    R0 = np.median(baseline_window)
    if len(baseline_window) == 0 or np.isnan(R0) or R0 == 0:
        R0 = 1e-6

    # --- segment & normalize ---
    seg = signal[start_idx:end_idx+1]
    S = (seg - R0) / R0
    t = X[start_idx:end_idx+1] - X[start_idx]  # time relative to pulse start
    return pd.DataFrame({"time": t, "resistance": S})


def main():
    df = pd.read_excel("data sets/Train/Data Set.xlsx", sheet_name="Sheet1")
    df = df.drop(columns=["Unnamed: 0"])  # all-NaN, not needed

    X = df["X"].values.astype(float)
    features_rows = []

    for col_name, gas_label in {"Y1": "ethanol", "Y2": "acetone", "Y3": "isopropanol"}.items():
        signal = df[col_name].values.astype(float)
        signal = np.nan_to_num(signal, nan=0.0)  # fill NaN with 0

        # --- 1) Peak detection (corrected params) ---
        prominence = 0.1 * np.max(signal)
        # 451 rows / 8 pulses ≈ 56 rows between peaks; distance=40 is safe
        peaks, _ = find_peaks(signal, prominence=prominence, distance=40)

        if len(peaks) != 8:
            print(f"WARNING: {gas_label} peaks={len(peaks)}, expected 8")

        # --- 2) Extract each pulse segment ---
        for i, peak_idx in enumerate(peaks):
            seg_df = extract_segment(signal, peak_idx, X, peaks, i)

            # --- 3) Extract features via existing pipeline fn ---
            feats = pipeline.extract_features(seg_df)
            feats["gas"] = gas_label
            feats["concentration_idx"] = i  # categorical metadata only
            features_rows.append(feats)

    # --- 4) Save CSV (concentration_idx dropped before model use) ---
    features_df = pd.DataFrame(features_rows)
    # NOTE: concentration_idx is metadata — dropped before model training
    features_df = features_df.drop(columns=["concentration_idx"])
    features_df.to_csv("extracted_features.csv", index=False)

    # --- 5) Summary print ---
    print(f"Extracted {len(features_df)} feature vectors")
    print(f"  gases: {sorted(features_df['gas'].unique())}")
    print(f"  features saved to extracted_features.csv")
    print(f"  concentration_idx column DROPPED before model training (categorical metadata only)")


if __name__ == "__main__":
    main()