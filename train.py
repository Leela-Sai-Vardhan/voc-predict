"""Train VOC classifier on open data (UCI id 1081), classify later real data.

Row layout: label, concentration, 9000 response points = 10 sensors x 900.
Each per-sensor slice is one training sample -> matches the single-sensor
feature contract of pipeline.extract_features. Recovery-phase features are
dropped (this dataset records baseline + injection only; they would be NaN).
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import pipeline

GASES = ["ethanol", "acetone", "isopropanol"]  # 3/4 targets present in UCI 1081
CSV = "uci1081/gsalc.csv"
# protocol-invariant, sensor-normalized features (no r0 / absolute peak_time)
FEATURES = ["peak_response", "response_time", "rise_rate", "tau_ad_1", "tau_ad_2"]
MODEL = "model.joblib"
META = "model_meta.json"


def build_samples():
    df = pd.read_csv(CSV, header=None)
    rows, groups = [], []
    for i, r in df.iterrows():
        if r[0] not in GASES:
            continue
        D = r[2:].to_numpy().reshape(10, 900)
        for sensor in range(10):
            f = pipeline.extract_features(
                pd.DataFrame({"time": np.arange(900), "resistance": D[sensor]}))
            rows.append({k: f[k] for k in FEATURES} | {"label": r[0]})
            groups.append(i)  # row-level split so the 10 sensors of one sample never straddle folds
    return pd.DataFrame(rows), np.array(groups)


def main():
    X, groups = build_samples()
    y = X.pop("label").to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"samples={len(X)}  gases={list(le.classes_)}")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    models = {
        "RF": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        "XGB": XGBClassifier(n_estimators=300, learning_rate=0.1,
                             max_depth=5, eval_metric="mlogloss",
                             random_state=0, n_jobs=-1),
    }
    best, best_acc, best_cm = None, -1.0, None
    for name, clf in models.items():
        scores, f1s, cms = [], [], []
        for tr, te in cv.split(X, y, groups):
            clf.fit(X.iloc[tr], y[tr])
            p = clf.predict(X.iloc[te])
            scores.append(accuracy_score(y[te], p))
            f1s.append(f1_score(y[te], p, average="weighted"))
            cms.append(confusion_matrix(y[te], p, labels=list(range(len(le.classes_)))))
        acc = float(np.mean(scores))
        print(f"{name}: cv-acc={acc:.3f}  cv-f1={np.mean(f1s):.3f}")
        if acc > best_acc:
            best_acc, best, best_cm = acc, clf, np.mean(cms, axis=0)

    best.fit(X, y)  # final model on all open data
    joblib.dump(best, MODEL)
    with open(META, "w") as f:
        json.dump({"features": FEATURES, "gases": list(le.classes_)}, f)
    print(f"saved {MODEL} (CV acc {best_acc:.3f}) + {META}")
    print("best-fit confusion matrix (rows=pred, cols=true):\n", np.round(best_cm).astype(int))


def predict(df):
    """Predict gas for a recorded DataFrame with 'time' and 'resistance'."""
    model = joblib.load(MODEL)
    with open(META) as f:
        meta = json.load(f)
    f = pipeline.extract_features(df)
    row = pd.DataFrame([{k: f[k] for k in meta["features"]}])
    proba = model.predict_proba(row)[0]
    return meta["gases"][int(np.argmax(proba))], dict(zip(meta["gases"], np.round(proba, 3)))


if __name__ == "__main__":
    main()