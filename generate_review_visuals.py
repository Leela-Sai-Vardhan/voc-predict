"""Generate first-review presentation visuals (300 dpi, professional style)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 14, "font.family": "sans-serif"})

DATA = "data sets/Train/Data Set.xlsx"


def load_data():
    df = pd.read_excel(DATA, sheet_name="Sheet1")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


# =========================================================================
# Visual 1: Transient Profiles Across All Cycles
# =========================================================================
def make_visual1(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["X"], df["Y1"], label="Gas 1 (Ethanol)", linewidth=1.5)
    ax.plot(df["X"], df["Y2"], label="Gas 2 (Acetone)", linewidth=1.5)
    ax.plot(df["X"], df["Y3"], label="Gas 3 (Isopropanol)", linewidth=1.5)
    ax.set_title("Preliminary Transient Response Profiles Across Multiple Cycles",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Sensor Resistance Change")
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig("figures/visual1_transient_profiles.png", dpi=300)
    plt.close(fig)
    print("Saved visual1_transient_profiles.png")


# =========================================================================
# Visual 2: Zoomed Feature Extraction on Second Pulse of Y3
# =========================================================================
def make_visual2(df):
    signal = df["Y3"].values.astype(float)
    signal = np.nan_to_num(signal, nan=0.0)
    x = df["X"].values.astype(float)

    # Isolate strictly X > 590 and X < 750
    mask = (x > 590) & (x < 750)
    seg_x = x[mask]
    seg_y = signal[mask]

    if len(seg_x) == 0:
        print("Warning: no data in X>590,X<750 range for visual2")
        return

    R0 = np.median(seg_y[:5]) if len(seg_y) >= 5 else np.min(seg_y)
    R_max = np.max(seg_y)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(seg_x, seg_y, color="black", linewidth=2)
    ax.axhline(y=R0, color="grey", linestyle="--", linewidth=1.5, label=f"Baseline ($R_0$)")
    ax.axhline(y=R_max, color="red", linestyle="--", linewidth=1.5, label="Peak Response ($\\Delta R_{max}$)")

    # Annotate Response Time on rising edge (~X=630 to X=680)
    # Find the midpoint of the steep rise
    rise_mask = (seg_x >= 630) & (seg_x <= 680)
    if np.any(rise_mask):
        rise_x = seg_x[rise_mask][len(seg_x[rise_mask])//2]
        rise_y = seg_y[rise_mask][len(seg_y[rise_mask])//2]
        ax.annotate("Response Time",
                    xy=(rise_x, rise_y),
                    xytext=(rise_x + 5, R_max * 0.75),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=1.5),
                    fontsize=13, color="blue", fontweight="bold")

    ax.set_title("Dynamic Feature Extraction: Single Exposure Cycle",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Sensor Resistance Change")
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig("figures/visual2_feature_extraction.png", dpi=300)
    plt.close(fig)
    print("Saved visual2_feature_extraction.png")


# =========================================================================
# Visual 3: Boxplot of Peak Amplitudes Per Gas Channel
# =========================================================================
def make_visual3(df):
    peak_values = []
    channel_labels = ["Gas 1", "Gas 2", "Gas 3"]
    channels = ["Y1", "Y2", "Y3"]

    for col in channels:
        signal = df[col].values.astype(float)
        signal = np.nan_to_num(signal, nan=0.0)
        peaks, _ = find_peaks(signal, prominence=0.1 * np.max(signal), distance=40)
        peak_vals = signal[peaks]
        # Normalize to relative S (peak relative to baseline)
        baseline = np.median(signal[:max(1, peaks[0] // 2)]) if len(peaks) > 0 else 1.0
        peak_vals = peak_vals / baseline if baseline > 0 else peak_vals
        peak_values.append(peak_vals)

    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(peak_values, tick_labels=channel_labels, patch_artist=True, widths=0.5)

    # Color-code boxes
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_title("Discriminative Power of Peak Response Feature",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Gas Channel")
    ax.set_ylabel("Peak Amplitude (Relative S)")
    fig.tight_layout()
    fig.savefig("figures/visual3_feature_separation.png", dpi=300)
    plt.close(fig)
    print("Saved visual3_feature_separation.png")


# =========================================================================
# Main
# =========================================================================
def main():
    df = load_data()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    make_visual1(df)
    make_visual2(df)
    make_visual3(df)

    print("\nAll visuals generated successfully.")


if __name__ == "__main__":
    main()