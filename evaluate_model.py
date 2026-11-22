"""Fast per-class precision/recall/F1 evaluation on saved model.
Evaluates: (a) extracted_features.csv [training-data eval, optimistic],
           (b) real 100 ppm recordings [honest holdout].
Skips the slow UCI rebuild; hold-out acc=0.634/F1=0.633 already from full_retrain.py.
"""
import numpy as np
import pandas as pd
import joblib, json
import pipeline

MODEL = "model.joblib"
META = "model_meta.json"
EXTRACTED = "extracted_features.csv"
REAL_XLSX = r"data sets/new/ZnO_VOC_Feature_Extraction_100ppm.xlsx"
GAS_MAP = {"Ethanol": "ethanol", "Methanol": "methanol",
           "Acetone": "acetone", "Isopropanol": "isopropanol"}
FEATURES = ["peak_response", "response_time", "rise_rate", "tau_ad_1", "tau_ad_2"]


def metrics_report(y_true, y_pred, classes, label):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix, classification_report)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*62}")
    print(f"EVALUATION: {label}")
    print(f"{'='*62}")
    print(f"Accuracy (overall): {acc:.4f}  ({int((y_true==y_pred).sum())}/{len(y_true)})")
    print()
    print("=== Per-class ===")
    for cls in classes:
        mask = y_true == cls
        n = int(mask.sum())
        tp = int(((y_pred == cls) & mask).sum())
        fn = int(((y_pred != cls) & mask).sum())
        fp = int(((y_pred == cls) & ~mask).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        print(f"  {cls:14s} n={n:3d}  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  "
              f"(TP={tp}, FP={fp}, FN={fn})")
    print()
    print("=== Averages ===")
    print(f"  Macro   precision={precision_score(y_true,y_pred,average='macro',labels=classes,zero_division=0):.3f}  "
          f"recall={recall_score(y_true,y_pred,average='macro',labels=classes,zero_division=0):.3f}  "
          f"f1={f1_score(y_true,y_pred,average='macro',labels=classes,zero_division=0):.3f}")
    print(f"  Weighted precision={precision_score(y_true,y_pred,average='weighted',labels=classes,zero_division=0):.3f}  "
          f"recall={recall_score(y_true,y_pred,average='weighted',labels=classes,zero_division=0):.3f}  "
          f"f1={f1_score(y_true,y_pred,average='weighted',labels=classes,zero_division=0):.3f}")
    print()
    print("=== Confusion matrix (rows=true, cols=pred) ===")
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    print("      " + "  ".join(f"{c[:4]:>4}" for c in classes))
    for i, cls in enumerate(classes):
        print(f"  {cls[:4]:>4}  " + "  ".join(f"{v:>4}" for v in cm[i]))
    print()
    print(classification_report(y_true, y_pred, labels=classes, zero_division=0))
    return acc


def eval_csv(path, label):
    df = pd.read_csv(path)
    # map gas -> label, ensure lowercase
    df = df.rename(columns={"gas": "label"}) if "gas" in df.columns else df
    df["label"] = df["label"].str.lower()
    X = df[FEATURES].fillna(0.0)
    y_true = df["label"].values
    model = joblib.load(MODEL)
    with open(META) as f:
        classes = list(json.load(f)["gases"])
    proba = model.predict_proba(X)
    y_pred = [classes[int(np.argmax(p))] for p in proba]
    return metrics_report(y_true, y_pred, classes, label)


def eval_real():
    import os
    if not os.path.exists(REAL_XLSX):
        print(f"\n{REAL_XLSX} not found — skipping real-data eval")
        return
    raw = pd.read_excel(REAL_XLSX, sheet_name="Raw_Time_Series")
    model = joblib.load(MODEL)
    with open(META) as f:
        classes = list(json.load(f)["gases"])
    rows = []
    for (voc, cycle), sub in raw.groupby(["VOC", "Cycle"]):
        t = sub["Time_within_cycle_s"].to_numpy(float).reshape(-1, 1)
        i = sub["Current_nA"].to_numpy(float)
        t_dense = np.arange(t.min(), t.max() + 1.0, 1.0)
        i_dense = np.interp(t_dense, t.ravel(), i)
        resistance = 1.0 / i_dense
        rec = pd.DataFrame({"time": t_dense, "resistance": resistance})
        f = pipeline.extract_features(rec)
        row = {k: f[k] for k in FEATURES}
        row["label"] = GAS_MAP[voc]
        rows.append(row)
    df = pd.DataFrame(rows).fillna(0.0)
    X = pd.DataFrame([{k: r[k] for k in FEATURES} for _, r in df.iterrows()])
    y_true = df["label"].values
    proba = model.predict_proba(X)
    y_pred = [classes[int(np.argmax(p))] for p in proba]
    return metrics_report(y_true, y_pred, classes, "Real 100 ppm data (holdout, 12 recordings)")


def benchmark_loco_cv():
    """Benchmark Linear SVM, RBF SVM, Logistic Regression, RF, and XGBoost via LOCO-CV."""
    import os
    if not os.path.exists(REAL_XLSX):
        return
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut

    raw = pd.read_excel(REAL_XLSX, sheet_name="Raw_Time_Series")
    rows = []
    for (voc, cycle), sub in raw.groupby(["VOC", "Cycle"]):
        t = sub["Time_within_cycle_s"].to_numpy(float).reshape(-1, 1)
        i = sub["Current_nA"].to_numpy(float)
        t_dense = np.arange(t.min(), t.max() + 1.0, 1.0)
        i_dense = np.interp(t_dense, t.ravel(), i)
        rec = pd.DataFrame({"time": t_dense, "resistance": 1.0 / i_dense})
        f = pipeline.extract_features(rec)
        rows.append({k: f[k] for k in FEATURES} | {
            "gas_true": GAS_MAP[voc], "voc": voc, "cycle": int(cycle)
        })
    df = pd.DataFrame(rows).fillna(0.0)
    y = df["gas_true"]
    groups = df["cycle"]
    logo = LeaveOneGroupOut()

    models = {
        "Linear SVM (C=1.0)": make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)),
        "RBF SVM (C=1.0)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0)),
        "Logistic Regression (L2)": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)),
        "Random Forest (depth=2)": RandomForestClassifier(n_estimators=50, max_depth=2, random_state=42),
    }

    print("\n" + "=" * 62)
    print("BENCHMARK: Leave-One-Cycle-Out (LOCO) CV on Real 100 ppm (12 recordings)")
    print("=" * 62)
    print(f"{'Model Architecture':<30} {'Features Used':<22} {'LOCO Accuracy':<14}")
    print("-" * 68)

    # 1) Peak response alone
    for name, clf in [("Linear SVM", make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0)))]:
        preds = np.empty(len(y), dtype=object)
        for tr, te in logo.split(df[["peak_response"]], y, groups):
            clf.fit(df[["peak_response"]].iloc[tr], y.iloc[tr])
            preds[te] = clf.predict(df[["peak_response"]].iloc[te])
        acc = np.mean(preds == y.values)
        print(f"{name:<30} {'[peak_response]':<22} {acc:.1%} ({int(np.sum(preds == y.values))}/12)")

    # 2) Peak response + rise rate
    for name, clf in [("Linear SVM", make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))),
                      ("Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(C=1.0)))]:
        preds = np.empty(len(y), dtype=object)
        feats2 = ["peak_response", "rise_rate"]
        for tr, te in logo.split(df[feats2], y, groups):
            clf.fit(df[feats2].iloc[tr], y.iloc[tr])
            preds[te] = clf.predict(df[feats2].iloc[te])
        acc = np.mean(preds == y.values)
        print(f"{name:<30} {'[peak, rise_rate]':<22} {acc:.1%} ({int(np.sum(preds == y.values))}/12)")

    # 3) All 4 clean kinetic features
    feats4 = ["peak_response", "rise_rate", "tau_ad_1", "tau_ad_2"]
    for name, clf in models.items():
        preds = np.empty(len(y), dtype=object)
        for tr, te in logo.split(df[feats4], y, groups):
            clf.fit(df[feats4].iloc[tr], y.iloc[tr])
            preds[te] = clf.predict(df[feats4].iloc[te])
        acc = np.mean(preds == y.values)
        print(f"{name:<30} {'[peak, rise, taus]':<22} {acc:.1%} ({int(np.sum(preds == y.values))}/12)")
    print("=" * 68)


def main():
    model = joblib.load(MODEL)
    with open(META) as f:
        meta = json.load(f)
    classes = list(meta["gases"])
    print(f"Model: {MODEL}  classes={classes}  features={FEATURES}")
    print()
    eval_csv(EXTRACTED, "extracted_features.csv [training-data eval, optimistic]")
    eval_real()
    benchmark_loco_cv()


if __name__ == "__main__":
    main()
