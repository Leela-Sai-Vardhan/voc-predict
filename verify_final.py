"""Verify final model and deliverables."""
import joblib, json, os
import pandas as pd
import numpy as np
import pipeline

# Load model and meta
model = joblib.load('model.joblib')
with open('model_meta.json') as f:
    meta = json.load(f)

print('=== FINAL MODEL STATUS ===')
print(f'Classes: {meta["gases"]}')
print(f'Features: {meta["features"]}')
print(f'Model type: {type(model).__name__}')
print()

# Verify predict works
excel_path = 'data sets/Train/Data Set.xlsx' if os.path.exists('data sets/Train/Data Set.xlsx') else 'data sets/Data Set.xlsx'
df = pd.read_excel(excel_path, sheet_name='Sheet1')
df = df.drop(columns=['Unnamed: 0'])
signal = df['Y3'].values.astype(float)
signal = np.nan_to_num(signal, nan=0.0)
from scipy.signal import find_peaks
peaks, _ = find_peaks(signal, prominence=0.1*np.max(signal), distance=40)
seg_df = pd.DataFrame({'time': np.arange(53), 'resistance': signal[:53]})
f = pipeline.extract_features(seg_df)
sample = pd.DataFrame([{k: f[k] for k in meta['features']}])
proba = model.predict_proba(sample)[0]
pred = meta['gases'][np.argmax(proba)]
print(f'Predict test: {pred} ({np.max(proba):.1%} confidence)')
print()

# Verify files
print('=== DELIVERABLES ===')
for fname in ['segment_and_extract.py', 'extracted_features.csv', 'model.joblib', 'model_meta.json', 'full_retrain.py']:
    exists = os.path.exists(fname)
    size = os.path.getsize(fname) if exists else 0
    print(f'  {fname}: {"OK" if exists else "MISSING"} ({size:,} bytes)')
print()
print('=== FIGURES ===')
if os.path.isdir('figures'):
    for fn in sorted(os.listdir('figures')):
        fp = os.path.join('figures', fn)
        print(f'  {fn}: {os.path.getsize(fp):,} bytes')
print()
print('=== EXTRACTED FEATURES ===')
ef = pd.read_csv('extracted_features.csv')
print(f'Shape: {ef.shape}')
print(f'Gases: {sorted(ef["gas"].unique())}')