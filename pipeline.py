"""VOC single-sensor discrimination pipeline.

Contract: any raw recording is a pandas DataFrame with `time` (seconds) and
`resistance` (ohm) columns. Pipeline: normalize -> smooth -> segment the gas
pulse -> extract kinetics features incl. a two-term exponential tau fit.

Also ships `generate_synthetic()` so the pipeline is validated on ground-truth
curves; the same DataFrame contract is used for real / open data.
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit


# ---------------------------------------------------------------- Step 1
def baseline_r0(df, window_s=5.0):
    """R0 = median resistance over the first window_s seconds (clean air)."""
    t = df["time"].to_numpy()
    r = df["resistance"].to_numpy()
    return float(np.median(r[t <= t[0] + window_s]))


def relative_response(df, r0=None):
    """S(t) = (R(t) - R0) / R0."""
    if r0 is None:
        r0 = baseline_r0(df)
    return (df["resistance"] - r0) / r0


def smooth_sg(s, window=11, poly=3):
    """Savitzky-Golay smoothing. window is forced odd and > poly."""
    if window % 2 == 0:
        window += 1
    return pd.Series(savgol_filter(s.to_numpy(float), window, poly), index=s.index)


def noise_std(s, df, window_s=5.0):
    """Noise floor measured on the clean-air baseline segment."""
    t = df["time"].to_numpy()
    return float(s.to_numpy()[t <= t[0] + window_s].std())


# ---------------------------------------------------------------- Step 2
def detect_exposure(s, df, threshold_k=3.0, window_s=5.0):
    """Exposure onset index; robust to slow baseline drift and steep decays.

    Onset = first point where |d|S|/dt| exceeds 50% of its max *before the
    peak* (a steep desorption must not be mistaken for the injection).
    Returns (None, None) if no excursion above threshold_k * baseline noise.
    """
    x = s.abs().to_numpy()
    dt = float(np.median(np.diff(df["time"].to_numpy())))
    peak = int(np.argmax(x))
    g = np.abs(np.gradient(x, dt))[:peak + 11]  # +margin: cover ramp rounded by smoothing
    valid = np.flatnonzero(g > 0.5 * g.max()) if g.size > 0 and g.max() > 0 else np.array([])
    if len(valid) == 0:
        return None, None
    start = int(valid[0])
    noise = threshold_k * noise_std(s, df, window_s)
    if x[start:].max() < noise:
        return None, None
    return start, None


def _desorption(t, A1, tau1, A2, tau2, c):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + c


def _adsorption(t, A1, tau1, A2, tau2, c):
    return A1 * (1 - np.exp(-t / tau1)) + A2 * (1 - np.exp(-t / tau2)) + c


def _fit_two_term(t, y, rising):
    t0 = t - t[0]
    y0 = y[0]
    peak = np.abs(y).max()
    p0 = [peak * 0.6, 2.0, peak * 0.4, 10.0, y0]
    f = _adsorption if rising else _desorption
    ub = max(100.0, 10.0 * (t[-1] - t[0]))  # keep taus physical, no degenerate flat tail
    bounds = ([-np.inf, 1e-3, -np.inf, 1e-3, -np.inf],
              [np.inf, ub, np.inf, ub, np.inf])
    try:
        popt, _ = curve_fit(f, t0, y, p0=p0, bounds=bounds, maxfev=20000)
        taus = np.sort(np.abs([popt[1], popt[3]]))
        return float(taus[0]), float(taus[1])
    except RuntimeError:
        return float("nan"), float("nan")


def extract_features(df, window_s=5.0, sg_window=11, sg_poly=3, threshold_k=3.0):
    """Full feature set from one gas-pulse recording -> dict / one row."""
    t = df["time"].to_numpy()
    r0 = baseline_r0(df, window_s)
    s_raw = relative_response(df, r0)
    s = smooth_sg(s_raw.abs(), sg_window, sg_poly)

    start, _ = detect_exposure(s, df, threshold_k, window_s)
    if start is None:
        return {"r0": r0, "peak_response": 0.0, "response_time": float("nan"),
                "recovery_time": float("nan"), "rise_rate": 0.0,
                "decay_rate": 0.0, "tau_ad_1": float("nan"),
                "tau_ad_2": float("nan"), "tau_de_1": float("nan"),
                "tau_de_2": float("nan"), "peak_time": float("nan"),
                "norm_slope": float("nan")}

    peak_idx = start + int(np.argmax(s.to_numpy()[start:]))
    peak = float(s.to_numpy()[peak_idx])
    dt = np.median(np.diff(t))

    ads = s.to_numpy()[start:peak_idx + 1]
    des = s.to_numpy()[peak_idx:]
    g_post = np.abs(np.gradient(des, dt)) if des.size > 4 else np.array([0.0])
    pit = np.flatnonzero(g_post < 0.02 * g_post.max())
    plateau = int(pit[0]) if pit.size else None  # where the decay flattens back

    def _first(arr, cond):
        hit = np.flatnonzero(cond)
        return int(hit[0]) if hit.size else None

    # t90 on the (smoothed) adsorption / desorption flanks; nan if phase absent
    i90 = _first(ads, ads >= 0.9 * peak)
    irec = _first(des, des <= 0.1 * peak)  # recovery = falls back to 10% above baseline
    response_time = float(t[start + i90] - t[start]) if i90 is not None else float("nan")
    recovery_time = float(t[peak_idx + irec] - t[peak_idx]) if irec is not None else float("nan")

    rise_rate = float(np.max(np.gradient(np.abs(ads), dt))) if len(ads) > 4 else 0.0
    decay_rate = float(np.max(-np.gradient(np.abs(des), dt))) if irec is not None and len(des) > 4 else float("nan")

    tau_ad = _fit_two_term(t[start:peak_idx + 1], ads, rising=True) if len(ads) >= 6 else (np.nan, np.nan)
    d_end = plateau + 1 if plateau is not None and plateau >= 5 else len(des)
    tau_de = _fit_two_term(t[peak_idx:peak_idx + d_end], des[:d_end],
                           rising=False) if d_end >= 6 and irec is not None else (np.nan, np.nan)

    return {"r0": r0, "peak_response": peak,
            "response_time": response_time, "recovery_time": recovery_time,
            "rise_rate": rise_rate, "decay_rate": decay_rate,
            "tau_ad_1": tau_ad[0], "tau_ad_2": tau_ad[1],
            "tau_de_1": tau_de[0], "tau_de_2": tau_de[1],
            "peak_time": float(t[peak_idx]),
            "norm_slope": float(rise_rate / peak) if peak > 0 else float("nan")}


# --------------------------------------------------- synthetic data (test)
GAS_PARAMS = {
    # gasswitch smax  adsorption taus  desorption taus  weight of tau1
    "ethanol":     dict(smax=0.15, ta=(2.0, 12.0), td=(3.0, 20.0), w=0.6),
    "methanol":    dict(smax=0.32, ta=(1.5, 8.0),  td=(2.5, 15.0), w=0.5),
    "acetone":     dict(smax=0.50, ta=(3.5, 18.0), td=(4.0, 30.0), w=0.5),
    "isopropanol": dict(smax=0.68, ta=(2.5, 10.0), td=(3.5, 24.0), w=0.4),
}

# Verified physical kinetics from the real 100 ppm ZnO recordings (session.md
# "Physical Grounding"). Same schema as GAS_PARAMS; use as
# generate_synthetic(gas, params=ZNO_PARAMS). The slow ta (~30-37 s) needs
# expose_s >= 120 s to resolve cleanly (default expose_s=40 truncates it).
ZNO_PARAMS = {
    "methanol":    dict(smax=0.191, ta=(10.6, 36.5), td=(5.0, 40.0), w=0.5),
    "acetone":     dict(smax=0.251, ta=(0.2, 29.7),  td=(5.0, 40.0), w=0.5),
    "isopropanol": dict(smax=0.282, ta=(5.0, 34.8),  td=(5.0, 40.0), w=0.5),
    "ethanol":     dict(smax=0.306, ta=(0.6, 29.4),  td=(5.0, 40.0), w=0.5),
}


def generate_synthetic(gas, r0=10e6, seed=0, baseline_s=5.0, expose_s=40.0,
                       recover_s=60.0, dt=0.1, noise_std_rel=0.002,
                       concentration=1.0, params=GAS_PARAMS):
    """One recording: clean-air baseline -> gas on -> gas off.

    n-type ZnO exposed to a reducing VOC => resistance drops => S(t) < 0.
    Peak response scales as concentration**0.5 (MOX power law).
    Returns a `time`/`resistance` DataFrame with known ground-truth kinetics.
    """
    rng = np.random.default_rng(seed)
    p = params[gas]
    smax = p["smax"] * float(concentration) ** 0.5
    t_end = baseline_s + expose_s + recover_s
    t = np.arange(0, t_end, dt)

    t_on = baseline_s
    t_off = baseline_s + expose_s
    rise = np.where(t >= t_on,
                    smax * (p["w"] * (1 - np.exp(-(t - t_on) / p["ta"][0]))
                            + (1 - p["w"]) * (1 - np.exp(-(t - t_on) / p["ta"][1]))),
                    0.0)
    full = float(rise[np.searchsorted(t, t_off) - 1])
    fall = np.where(t >= t_off,
                    full * (p["w"] * np.exp(-(t - t_off) / p["td"][0])
                            + (1 - p["w"]) * np.exp(-(t - t_off) / p["td"][1])),
                    0.0)
    s_abs = np.where(t < t_on, 0.0, np.where(t < t_off, rise, fall))
    s_abs += rng.normal(0, noise_std_rel, t.size)
    # n-type reducing gas -> negative S(t) -> resistance decreases
    resistance = r0 * (1 - np.maximum(s_abs, 0))
    return pd.DataFrame({"time": t, "resistance": resistance})


if __name__ == "__main__":
    rows = []
    for gas, p in GAS_PARAMS.items():
        df = generate_synthetic(gas)
        f = extract_features(df)
        f["gas"] = gas
        rows.append(f)

        # self-check against ground truth
        tol = 0.25
        assert abs(f["peak_response"] - p["smax"]) / p["smax"] < 0.15, gas
        assert abs(f["tau_ad_2"] - p["ta"][1]) / p["ta"][1] < tol, gas
        assert abs(f["tau_de_2"] - p["td"][1]) / p["td"][1] < tol, gas
        assert f["response_time"] > 0 and f["recovery_time"] > 0, gas
    print(pd.DataFrame(rows).set_index("gas").round(3))
    print("\nself-check OK: peaks and tau2 recovered within tolerance for all 4 gases")