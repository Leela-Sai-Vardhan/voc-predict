"""4-class model: real UCI 1081 (3 gases) + synthetic transients (4 gases).

Synthetic gives methanol its class today (methanol absent from open real data).
One StratifiedGroupKFold over the merged set; accuracy reported separately on
the real (UCI) and synthetic test slices of each fold, so one can see whether
adding synthetic data hurts the real-data performance.
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import pipeline
import train as base

GASES = ["ethanol", "methanol", "acetone", "isopropanol"]
N_SEEDS, CONCS = 40, [0.5, 1.0, 2.0, 4.0]  # 4 gases x 40 seeds x 4 conc = 640 synth samples


def build_synth():
    rows, groups = [], []
    for gi, gas in enumerate(GASES):
        for seed in range(N_SEEDS):
            for conc in CONCS:
                df = pipeline.generate_synthetic(gas, seed=seed, concentration=conc)
                f = pipeline.extract_features(df)
                rows.append({k: f[k] for k in base.FEATURES} | {"label": gas})
                groups.append(f"synth-{gas}-{seed}")  # same gas+seed across concs stay together
    return pd.DataFrame(rows), groups


def main():
    xr, _ = base.build_samples()
    xr["src"] = "real"
    xs, gs = build_synth()
    xs["src"] = "synth"
    X = pd.concat([xr, xs], ignore_index=True)
    yraw = X.pop("label").to_numpy()
    src = X.pop("src").to_numpy()
    # one group per real sample (its 10 sensors stay together) and per synth (gas,seed)
    groups = np.array([f"real-{i}" for i in range(len(xr))] + gs, dtype=object)
    le = LabelEncoder()
    y = le.fit_transform(yraw)
    print(f"merged samples={len(X)}  real={int((src=='real').sum())}  synth={int((src=='synth').sum())}  gases={list(le.classes_)}")

    clf = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                        eval_metric="mlogloss", random_state=0, n_jobs=-1)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    acc_real, acc_synth = [], []
    for tr, te in cv.split(X, y, groups):
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict(X.iloc[te])
        r = src[te] == "real"
        acc_real.append(accuracy_score(y[te][r], p[r]))
        acc_synth.append(accuracy_score(y[te][~r], p[~r]))
    print(f"4-class CV  real-acc={np.mean(acc_real):.3f}  synth-acc={np.mean(acc_synth):.3f}")

    clf.fit(X, y)
    joblib.dump(clf, base.MODEL)
    with open(base.META, "w") as f:
        json.dump({"features": base.FEATURES, "gases": list(le.classes_)}, f)
    print(f"saved {base.MODEL} + {base.META}  (4-class, {len(X)} samples)")


if __name__ == "__main__":
    main()