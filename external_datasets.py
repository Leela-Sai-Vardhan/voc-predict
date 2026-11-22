"""Adapters for external VOC datasets -> (time, resistance) frames for pipeline.extract_features."""
import glob
import gzip
import os

import numpy as np
import pandas as pd

FEATURES = ["peak_response", "response_time", "rise_rate", "tau_ad_1", "tau_ad_2"]
RATE_HZ = 25.0
SMOOTH_S = 12.0  # ~1 respiration cycle; removes ~12 s ripple, keeps exposure envelope


def load_flow_modulation(path, sensor=None, gas_filter=("acetone", "ethanol")):
    """UCI 308 rawdata.csv.gz. dR is normalized delta-R, so resistance = 1 + dR.

    sensor=None keeps all 16 sensors. Returns list of {exp, gas, lab, sensor, time, resistance}.
    """
    out = {}
    with gzip.open(path, "rt") as f:
        f.readline()  # header
        for line in f:
            p = line.split(",", 9)
            if len(p) < 10:
                continue
            sid, gas, exp = int(p[0]), p[6], p[2]
            if gas not in gas_filter or (sensor is not None and sid != sensor):
                continue
            key = (exp, sid)
            out[key] = {"exp": exp, "gas": gas, "lab": p[7], "sensor": sid,
                        "dR": np.array([float(x) for x in p[9].split(",")])}

    win = int(SMOOTH_S * RATE_HZ)
    for v in out.values():
        env = np.convolve(v.pop("dR"), np.ones(win) / win, mode="same")[:: int(RATE_HZ)]  # 1 Hz
        v["time"] = np.arange(len(env), dtype=float)
        v["resistance"] = 1.0 + env
    return list(out.values())


def load_zenodo_drift(path, sensor="R1", stage=2):
    """One Zenodo day-file -> exposure segment for `sensor`. stage: 1=baseline,2=exposure,3=recovery.

    Returns dict: {gas, conc, day, stage, time, resistance, temp, humidity}.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", delimiter=r"\s+")
    col = f"{sensor}[Ohm]"
    seg = df[df["Cycle_Stage"] == stage]
    if seg.empty:
        raise ValueError(f"stage {stage} missing in {os.path.basename(path)}")
    name = os.path.basename(path).split("_")
    gas = name[0]
    conc = name[1] if len(name) > 1 and name[1][0].isdigit() else "unknown"
    day = int(os.path.basename(os.path.dirname(path)).split()[1])
    return {
        "gas": gas,
        "conc": conc,
        "day": day,
        "stage": stage,
        "time": np.arange(len(seg), dtype=float),
        "resistance": seg[col].to_numpy(dtype=float),
        "temp": float(seg["T01[degC]"].mean()),
        "humidity": float(seg["H01[%rh]"].mean()),
    }


def _frame(sample):
    return pd.DataFrame({"time": sample["time"], "resistance": sample["resistance"]})


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    flow = os.path.join(here, "data_external", "flow_modulation", "rawdata.csv.gz")
    s = load_flow_modulation(flow, sensor=1)
    assert len(s) == 30, f"expected 15 exp/gas * 2 = 30, got {len(s)}"
    assert len(load_flow_modulation(flow)) == 480, "all-sensor load should be 58 exp * 16"
    f0 = _frame(s[0])
    assert f0["time"].is_monotonic_increasing
    assert np.isfinite(f0["resistance"]).all() and (f0["resistance"] > 0).all()
    assert {x["gas"] for x in s} == {"acetone", "ethanol"}

    z = glob.glob(os.path.join(here, "data_external", "zenodo_drift", "Day 1", "EtOH_1.csv"))[0]
    d = load_zenodo_drift(z)
    assert d["gas"] == "EtOH" and d["day"] == 1 and len(d["time"]) > 100
    assert np.isfinite(d["resistance"]).all() and (d["resistance"] > 0).all()
    print("external_datasets self-check OK:", len(s), "flow samples,", d["gas"], "day", d["day"])
