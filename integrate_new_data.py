"""Integrate Data Set.xlsx features into the existing model (fast path)."""
import json
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

# 1) Load extracted features from Data Set.xlsx
new_df = pd.read_csv("extracted_features.csv")
print(f"New data: {len(new_df)} samples, gases: {sorted(new_df['gas'].unique())}")

# 2) Load existing UCI real data
xr, _ = base.build_samples()
existing = xr.copy()
existing = existing.rename(columns={"label": "label"})
print(f"Existing UCI data: {len(existing)} samples")

# 3) Merge: UCI + new features
new_df_renamed = new_df.rename(columns={"gas": "label"})
merged = pd.concat(
    [existing[FEATURES + ["label"]], new_df_renamed[FEATURES + ["label"]]],
    ignore_index=True,
)

# 4) Encode labels
le = LabelEncoder()
y = le.fit_transform(merged["label"].values)

print(f"\nMerged dataset: {len(merged)} samples")
print(f"Classes: {list(le.classes_)}")

# 5) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    merged[FEATURES], y, test_size=0.3, random_state=42, stratify=y
)

# 6) Train XGB
clf = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=5,
    eval_metric="mlogloss", random_state=0, n_jobs=-1,
)
clf.fit(X_train, y_train)
p = clf.predict(X_test)
acc = accuracy_score(y_test, p)
f1 = f1_score(y_test, p, average="weighted")
print(f"\nHold-out test accuracy: {acc:.3f}, F1: {f1:.3f}")

# 7) Save the updated model
joblib.dump(clf, "model.joblib")
with open("model_meta.json", "w") as f:
    json.dump({"features": FEATURES, "gases": list(le.classes_)}, f)
print(f"\nModel saved with {len(le.classes_)} classes: {list(le.classes_)}")

# 8) Quick predict check
sample = new_df_renamed.iloc[0][FEATURES]
proba = clf.predict_proba([sample])[0]
pred = le.classes_[np.argmax(proba)]
print(f"\nQuick predict check: predicted={pred} | confidence={np.max(proba):.2%}")
print(f"Actual label: {new_df_renamed.iloc[0]['label']}")