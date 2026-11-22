"""Validate the new 100 ppm dataset (data sets/new/*.xlsx) as a holdout set.

Reads Raw_Time_Series, builds one time/resistance recording per VOC+Cycle
(4 VOCs x 3 cycles = 12 recordings), converts current -> resistance (R = V/I,
V=1 arbitrary -> relative response is voltage-invariant), densifies by linear
interpolation, extracts pipeline features, and scores the current model.

This dataset is too small/sparse (12 recordings, 10 pts each, single 100 ppm)
to train from scratch, so it is used strictly as validation.
"""
import numpy as np
import pandas as pd
import joblib, json
import pipeline

XLSX = r"data sets/new/ZnO_VOC_Feature_Extraction_100ppm.xlsx"
MODEL = "model.joblib"
META = "model_meta.json"
GAS_MAP = {"Ethanol": "ethanol", "Methanol": "methanol",
           "Acetone": "acetone", "Isopropanol": "isopropanol"}


def to_recording(df_sub):
    """One VOC+Cycle -> DataFrame(time, resistance) with resistance=1/current."""
    t = df_sub["Time_within_cycle_s"].to_numpy(float).reshape(-1, 1)
    i = df_sub["Current_nA"].to_numpy(float)
    # densify to 1 Hz for robust feature extraction
    t_dense = np.arange(t.min(), t.max() + 1.0, 1.0)
    i_dense = np.interp(t_dense, t.ravel(), i)
    resistance = 1.0 / i_dense  # R = V/I, V=1 (relative response is V-invariant)
    return pd.DataFrame({"time": t_dense, "resistance": resistance})


def main():
    raw = pd.read_excel(XLSX, sheet_name="Raw_Time_Series")
    model = joblib.load(MODEL)
    with open(META) as f:
        meta = json.load(f)
    features = meta["features"]
    gases = meta["gases"]

    rows = []
    for (voc, cycle), sub in raw.groupby(["VOC", "Cycle"]):
        gas = GAS_MAP[voc]
        rec = to_recording(sub)
        f = pipeline.extract_features(rec)
        row = {k: f[k] for k in features}
        row["gas_true"] = gas
        row["voc"] = voc
        row["cycle"] = int(cycle)
        rows.append(row)

    df = pd.DataFrame(rows)
    X = pd.DataFrame([{k: r[k] for k in features} for _, r in df.iterrows()])

    # handle NaN features (sparse data may yield NaN taus) -> impute with 0
    X = X.fillna(0.0)

    proba = model.predict_proba(X)
    preds = [gases[int(np.argmax(p))] for p in proba]
    conf = [float(np.max(p)) for p in proba]
    df["pred"] = preds
    df["confidence"] = conf
    df["correct"] = df["pred"] == df["gas_true"]

    print("=== NEW DATASET VALIDATION (holdout, 12 recordings) ===")
    print(f"Model: {MODEL}  classes={gases}  features={features}")
    print(f"Accuracy: {df['correct'].mean():.1%}  ({df['correct'].sum()}/{len(df)})")
    print()
    print(df[["voc", "cycle", "gas_true", "pred", "confidence", "correct"]]
          .to_string(index=False))
    print()
    print("=== Per-VOC ===")
    for voc in sorted(df["voc"].unique()):
        sub = df[df["voc"] == voc]
        print(f"  {voc:14s} acc={sub['correct'].mean():.1%}  "
              f"({int(sub['correct'].sum())}/{len(sub)})  "
              f"mean_conf={sub['confidence'].mean():.2f}")
    print()
    df.to_csv("new_dataset_validation.csv", index=False)
    print("Saved detailed results to new_dataset_validation.csv")


if __name__ == "__main__":
    main()
