"""Full retraining: UCI + Data Set.xlsx + synthetic methanol = 4-class model."""
import sys
import json
import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import pipeline
import train as base

FEATURES = base.FEATURES

# 1) Load extracted features from Data Set.xlsx (cleaned single-scale)
new_df = pd.read_csv("extracted_features.csv")
new_df_renamed = new_df.rename(columns={"gas": "label"})
print(f"New data: {len(new_df)} samples")

# 2) Training domain selection:
# Pass --include-uci or --legacy to include the 450 UCI-1081 array samples.
# By default, trains on the single-sensor ZnO domain (extracted_features + ZNO_PARAMS synthetic)
# to avoid the cross-sensor domain gap.
include_uci = "--include-uci" in sys.argv or "--legacy" in sys.argv

if include_uci:
    print("Loading UCI real data (--include-uci enabled)...")
    xr, _ = base.build_samples()
    existing = xr.copy()
    print(f"Existing UCI: {len(existing)} samples")

    methanol_rows = []
    for seed in range(40):
        for conc in [0.5, 1.0, 2.0, 4.0]:
            df = pipeline.generate_synthetic("methanol", seed=seed, concentration=conc,
                                             params=pipeline.ZNO_PARAMS, expose_s=150.0)
            f = pipeline.extract_features(df)
            methanol_rows.append({k: f[k] for k in FEATURES} | {"label": "methanol"})
    synth_methanol = pd.DataFrame(methanol_rows)
    print(f"Synthetic methanol: {len(synth_methanol)} samples")

    merged = pd.concat([
        existing[FEATURES + ["label"]],
        new_df_renamed[FEATURES + ["label"]],
        synth_methanol[FEATURES + ["label"]],
    ], ignore_index=True)
else:
    print("Training on ZnO single-sensor domain (extracted_features + ZNO_PARAMS synthetic)...")
    synth_rows = []
    for gas in ["acetone", "ethanol", "isopropanol", "methanol"]:
        for seed in range(30):
            for conc in [0.75, 0.9, 1.0, 1.15, 1.3]:
                df = pipeline.generate_synthetic(gas, seed=seed, concentration=conc,
                                                 params=pipeline.ZNO_PARAMS, expose_s=150.0)
                f = pipeline.extract_features(df)
                synth_rows.append({k: f[k] for k in FEATURES} | {"label": gas})
    synth_df = pd.DataFrame(synth_rows)
    print(f"Calibrated ZnO synthetic data: {len(synth_df)} samples")

    merged = pd.concat([
        new_df_renamed[FEATURES + ["label"]],
        synth_df[FEATURES + ["label"]],
    ], ignore_index=True)

# 5) Encode labels
le = LabelEncoder()
y = le.fit_transform(merged["label"].values)
print(f"\nMerged: {len(merged)} samples, classes: {list(le.classes_)}")

# 6) Train/test split and train XGB
X_train, X_test, y_train, y_test = train_test_split(
    merged[FEATURES], y, test_size=0.3, random_state=42, stratify=y
)
clf = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=5,
    eval_metric="mlogloss", random_state=0, n_jobs=-1,
)
clf.fit(X_train, y_train)
p = clf.predict(X_test)
acc = accuracy_score(y_test, p)
f1 = f1_score(y_test, p, average="weighted")
print(f"Hold-out accuracy: {acc:.3f}, F1: {f1:.3f}")

# 7) Save model
joblib.dump(clf, "model.joblib")
with open("model_meta.json", "w") as f:
    json.dump({"features": FEATURES, "gases": list(le.classes_)}, f)
print(f"\nModel saved with {len(le.classes_)} classes: {list(le.classes_)}")

# 8) Quick predict check
sample = new_df_renamed.iloc[0][FEATURES]
proba = clf.predict_proba([sample])[0]
pred = le.classes_[np.argmax(proba)]
print(f"Predict check: predicted={pred} ({np.max(proba):.1%}), actual={new_df_renamed.iloc[0]['label']}")

# 9) Validate on held-out real data (your new 100 ppm recordings)
def validate_on_real_data(clf, le, features, xlsx_path):
    """Load Raw_Time_Series -> one recording per VOC+Cycle -> score model."""
    try:
        raw = pd.read_excel(xlsx_path, sheet_name="Raw_Time_Series")
    except FileNotFoundError:
        print(f"\nReal-data validation skipped: {xlsx_path} not found")
        return None
    gas_map = {"Ethanol": "ethanol", "Methanol": "methanol",
               "Acetone": "acetone", "Isopropanol": "isopropanol"}
    rows = []
    for (voc, cycle), sub in raw.groupby(["VOC", "Cycle"]):
        t = sub["Time_within_cycle_s"].to_numpy(float).reshape(-1, 1)
        i = sub["Current_nA"].to_numpy(float)
        t_dense = np.arange(t.min(), t.max() + 1.0, 1.0)
        i_dense = np.interp(t_dense, t.ravel(), i)
        resistance = 1.0 / i_dense  # R = V/I, V=1 (response is V-invariant)
        rec = pd.DataFrame({"time": t_dense, "resistance": resistance})
        f = pipeline.extract_features(rec)
        row = {k: f[k] for k in features}
        row["gas_true"] = gas_map[voc]
        rows.append(row)
    df = pd.DataFrame(rows).fillna(0.0)
    X = pd.DataFrame([{k: r[k] for k in features} for _, r in df.iterrows()])
    proba = clf.predict_proba(X)
    preds = [le.classes_[int(np.argmax(p))] for p in proba]
    correct = np.array(preds) == np.array(df["gas_true"].values)
    acc = float(correct.mean())
    print(f"\n=== REAL-DATA VALIDATION (holdout, {len(df)} recordings) ===")
    print(f"Accuracy on real 100 ppm data: {acc:.1%} ({int(correct.sum())}/{len(df)})")
    print(df.assign(pred=preds)[["gas_true", "pred"]].to_string(index=False))
    return acc


# 7.5) Log this run (lean JSONL — no MLflow server needed)
real_acc = validate_on_real_data(clf, le, FEATURES,
                                 r"data sets/new/ZnO_VOC_Feature_Extraction_100ppm.xlsx")
run_log = {
    "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
    "script": "full_retrain.py",
    "samples": len(merged),
    "classes": list(le.classes_),
    "accuracy": float(round(acc, 3)),
    "f1": float(round(f1, 3)),
    "model": "model.joblib",
    "real_data_accuracy": round(real_acc, 4) if real_acc is not None else None,
}
with open("runs.jsonl", "a") as f:
    f.write(datetime.datetime.now(datetime.UTC).isoformat() + "Z" + " " + json.dumps(run_log) + "\n")