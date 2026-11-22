"""External validation on UCI-308 flow-modulation (acetone/ethanol) and Zenodo drift (ethanol)."""
import glob
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict

import external_datasets as ed
import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = ed.FEATURES
FLOW = os.path.join(HERE, "data_external", "flow_modulation", "rawdata.csv.gz")
DRIFT = os.path.join(HERE, "data_external", "zenodo_drift")


def _features(sample):
    fr = pipeline.extract_features(ed._frame(sample))
    return {k: fr[k] for k in FEATS}


def _predict(model, gases, X):
    return [gases[i] for i in model.predict(X)]


def validate_flow(model, gases):
    samples = ed.load_flow_modulation(FLOW)
    rows = []
    for s in samples:
        rows.append({**_features(s), "gas": s["gas"], "exp": s["exp"], "lab": s["lab"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "flow_modulation_features.csv"), index=False)

    y, groups = df["gas"], df["exp"]
    X = df[FEATS].fillna(0.0)
    print("\n=== UCI-308 flow modulation (acetone vs ethanol, n=%d) ===" % len(df))
    print("Rule1 per-class feature means:")
    print(X.groupby(y).mean().round(4).to_string())

    pred = _predict(model, gases, X)
    print("current-model accuracy: %.3f" % accuracy_score(y, pred))
    print("predicted:", pd.Series(pred).value_counts().to_dict())

    cv = cross_val_predict(
        RandomForestClassifier(300, random_state=0), X, y, groups=groups, cv=GroupKFold(5)
    )
    print("Rule3 leave-one-experiment-out (in-domain RF): %.3f" % accuracy_score(y, cv))
    print(classification_report(y, cv, digits=2))
    return df


def validate_drift(model, gases):
    rows = []
    for f in sorted(glob.glob(os.path.join(DRIFT, "Day *", "EtOH_*.csv"))):
        d = ed.load_zenodo_drift(f, sensor="R1", stage=2)
        rows.append({**_features(d), "day": d["day"], "temp": d["temp"], "humidity": d["humidity"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "zenodo_drift_features.csv"), index=False)

    X = df[FEATS].fillna(0.0)
    pred = _predict(model, gases, X)
    print("\n=== Zenodo drift (ethanol exposure, n=%d, %d days) ===" % (len(df), df["day"].nunique()))
    print("predicted:", pd.Series(pred).value_counts().to_dict())
    print("ethanol-recall: %.3f" % np.mean(np.array(pred) == "ethanol"))
    print("temp %.1f-%.1f C, humidity %.1f-%.1f %%rh"
          % (df.temp.min(), df.temp.max(), df.humidity.min(), df.humidity.max()))
    drift = df.groupby("day")[FEATS].mean()
    print("Rule1 feature drift (max |day-range| / overall mean):")
    print((drift.max() - drift.min()).div(drift.mean().abs().replace(0, np.nan)).round(3).to_string())
    return df


if __name__ == "__main__":
    meta = json.load(open(os.path.join(HERE, "model_meta.json")))
    model = joblib.load(os.path.join(HERE, "model.joblib"))
    validate_flow(model, meta["gases"])
    validate_drift(model, meta["gases"])
