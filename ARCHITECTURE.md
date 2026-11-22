# VOC-Predict — Architecture

Single-sensor **VOC (volatile organic compound) discrimination** from a metal-oxide
(MOX / ZnO, BME688-type) gas sensor's **dynamic transient response**. Instead of
using an array of sensors, this project heats/exposes one sensor and classifies the
gas from the *shape* of the resistance curve over time (kinetics), not just the
peak amplitude.

Target classes: **ethanol, methanol, acetone, isopropanol** (4-class model); the
baseline open-data model covers 3 of them (no methanol in open real data).

---

## 1. Core idea

A single MOX sensor is not selective on steady-state resistance alone: many gases
produce a similar peak. But the **rise (adsorption) and fall (desorption) kinetics**
differ per gas. The project:

1. Records `resistance` vs `time` during a clean-air baseline → gas pulse → recovery.
2. Normalizes to `S(t) = (R(t) − R0) / R0`.
3. Smooths (Savitzky–Golay) and detects the exposure onset.
4. Extracts transient features (peak, T90, rise/decay rate, two-term exponential
   time constants `tau`).
5. Feeds those 5 features to XGBoost / RandomForest.

The physics story (n-type ZnO, reducing VOC → resistance drops) is described in
`laborator_README.md` for the synthetic-data dataset.

---

## 2. Data contract (the single interface)

Every stage speaks the same shape:

```
pandas.DataFrame with columns:
  time        float   seconds
  resistance  float   ohms
```

`pipeline.extract_features(df)` consumes that contract and returns a feature dict.
Real recordings, UCI data, Excel pulses, and synthetic curves are all converted to
this contract before feature extraction.

---

## 3. Pipeline stages

```
Raw R(t)  ──►  S(t)=ΔR/R0  ──►  SavGol smoothing  ──►  onset detect
          ──►  transient features  ──►  XGBoost/RF  ──►  gas + probabilities
```

Implemented in `pipeline.py`:

| Function | Purpose |
|---|---|
| `baseline_r0(df, window_s=5)` | R0 = median resistance over first 5 s (clean air) |
| `relative_response(df, r0)` | `S(t) = (R−R0)/R0` |
| `smooth_sg(s, window=11, poly=3)` | Savitzky–Golay smoothing; window forced odd |
| `noise_std(s, df, window_s)` | Noise floor on the baseline segment |
| `detect_exposure(s, df, threshold_k=3)` | Onset index = first steep |dS/dt| before the peak, gated on 3× noise |
| `_fit_two_term` | Bounded two-exponential curve fit (adsorption or desorption) |
| `extract_features(df)` | Full feature row (see §4) |
| `generate_synthetic(gas, ...)` | Ground-truth transient generator for 4 gases |

Onset detection takes the **first** point where `|d|S|/dt|` exceeds 50% of its
maximum before the peak. This avoids mistaking a steep desorption for the injection
and tolerates slow baseline drift.

---

## 4. Feature vector

`pipeline.extract_features` returns 12 fields:

| Feature | Meaning |
|---|---|
| `r0` | baseline resistance (excluded from model: sensor-specific scale) |
| `peak_response` | magnitude of the peak |S| |
| `response_time` | T90 — time from onset to 90% of peak |
| `recovery_time` | time from peak to 10% above baseline |
| `rise_rate` | max positive dS/dt on the adsorption flank |
| `decay_rate` | max negative dS/dt on the desorption flank |
| `tau_ad_1`, `tau_ad_2` | two exponential time constants, adsorption |
| `tau_de_1`, `tau_de_2` | two exponential time constants, desorption |
| `peak_time` | absolute time of the peak (excluded: protocol-dependent) |
| `norm_slope` | `rise_rate / peak_response` — scale-invariant rise rate |

**Model features (5):** `peak_response`, `response_time`, `rise_rate`,
`tau_ad_1`, `tau_ad_2` — chosen to be protocol-invariant and sensor-normalized
(no `r0`, no absolute `peak_time`). Defined once in `train.FEATURES`; every trainer
and meta file reads it from there.

**`norm_slope`** (new) is computed and stored but is **not yet a model input**:
adding it to `train.FEATURES` changes the input dimension and invalidates the
committed `model.joblib`. Because `rise_rate` and `peak_response` both scale with
concentration, their ratio cancels it — the intended concentration-invariant
feature (Golden Rule 5). Adopt it at the next retrain by adding `norm_slope` to
`train.FEATURES`, regenerating all feature CSVs, and re-running `full_retrain.py`.

---

## 5. Datasets

### `uci1081/gsalc.csv` — UCI Gas Sensor Array Low-Concentration (id 1081)
- 90 rows × 9002 columns. Row layout: `label, concentration, 9000 points`.
- 9000 = **10 sensors × 900 time steps** (one sensor-array recording per row).
- 6 gases, 15 rows each: ethanol, acetone, toluene, ethyl acetate, isopropanol, hexane.
- Only the 3 gases overlapping our targets are used (ethanol, acetone, isopropanol).
- Each row is split into 10 per-sensor samples. Every one of the 10 sensors of a row
  is treated as a separate sample, but they stay in the same CV fold (`groups`).
- Records baseline + injection only → recovery features are NaN there.
- Download: `python dl_1081.py` (writes `uci1081.zip`, extracts `uci1081/`).

### `data sets/Train/Data Set.xlsx` — own ZnO sensor recording
- Sheet `Sheet1`, 451 rows × 5 columns: `X` (time axis, steps of 10), `Y1`, `Y2`, `Y3`.
- Channels map to **Y1=ethanol, Y2=acetone, Y3=isopropanol**.
- 8 concentration pulses per channel; `find_peaks(distance=40, prominence=0.1·max)`.
- `Unnamed: 0` is all-NaN → dropped. Missing values filled with 0.
- Processed by `segment_and_extract.py` → `extracted_features.csv` (24 rows = 3 gases × 8 pulses).
- **Gotcha — Y is a zero-baseline response amplitude, not ohms.** Between pulses
  Y ≈ 0; pulses reach 40–385. `pipeline.extract_features` normalizes by the segment
  baseline, so feeding raw Y divides by ~0 and blows `peak_response` up to ~100–1000
  (the historical "double normalization" bug: the old code pre-divided by a baseline,
  then `extract_features` divided again). The fix: `segment_and_extract.py` adds a
  fixed `NOMINAL_R0 = 1000` offset (`resistance = NOMINAL_R0 + Y`), giving a valid
  `R(t)` that `extract_features` normalizes **once**. Resulting peaks are 0.04–0.39,
  matching the real-ZnO scale. `extracted_features.csv` was regenerated with this fix;
  `model.joblib` is unchanged, so re-run `full_retrain.py` to actually train on the
  corrected values.

### Synthetic data (`pipeline.generate_synthetic`)
- Physics-based (simplified) generator for `ethanol, methanol, acetone, isopropanol`.
- Clean-air baseline → gas on → gas off, with two-term adsorption/desorption
  kinetics, concentration power law (`smax × C^0.5`), 0.2% Gaussian noise.
- Used to give **methanol** a class (absent from open real data) and to validate the
  feature extractor against known ground-truth `tau` and peak values.
- `train_synth.py` builds 640 samples: 4 gases × 40 seeds × 4 concentrations.
- **Two parameter sets:** `GAS_PARAMS` (generic, default) and `ZNO_PARAMS` (verified
  from the real 100 ppm ZnO recordings — `smax` 0.191–0.306, slow `ta` ≈ 29–37 s).
  Use `generate_synthetic(gas, params=ZNO_PARAMS)`. The slow `ta` needs
  `expose_s >= 120` to resolve cleanly; the default `expose_s=40` truncates it.
- Dataset card / physics model: `laborator_README.md`.

---

## 6. Training paths

| Script | What it trains | Output |
|---|---|---|
| `train.py` | 3-class baseline (UCI only), RF vs XGB, `StratifiedGroupKFold(5)` | `model.joblib`, `model_meta.json` |
| `train_synth.py` | 4-class merged: UCI real + synthetic, grouped CV, reports real vs synth accuracy separately | `model.joblib`, `model_meta.json` |
| `full_retrain.py` | **Final 4-class**: UCI + Data Set.xlsx + synthetic methanol, one 70/30 hold-out split | `model.joblib`, `model_meta.json` |
| `integrate_new_data.py` | 3-class fast path: UCI + Data Set.xlsx (no synthetic) | `model.joblib`, `model_meta.json` |

Model output contract (`model_meta.json`):

```json
{"features": ["peak_response","response_time","rise_rate","tau_ad_1","tau_ad_2"],
 "gases": ["acetone","ethanol","isopropanol","methanol"]}
```

Inference helper: `train.predict(df)` loads the model + meta, extracts features,
returns `(gas, {gas: probability})`.

**Currently committed model** is the 4-class XGBoost from `full_retrain.py`.

### Validation notes
- `train.py` / `train_synth.py` use grouped CV so the 10 sensors of one recording
  never straddle train/test. This is the honest estimate.
- `full_retrain.py` and `integrate_new_data.py` use a plain stratified 70/30
  `train_test_split` (no group awareness) — do not compare their accuracy directly
  with the grouped-CV numbers.

---

## 7. Repository map

```
pipeline.py               Core signal processing + feature extraction + synthetic generator
train.py                  Feature contract (FEATURES), UCI loader, 3-class training, predict()
train_synth.py            4-class training with synthetic data (grouped CV)
full_retrain.py           Final 4-class training: UCI + Excel data + synthetic methanol
integrate_new_data.py     Merge Excel features into model without synthetic methanol
segment_and_extract.py    Detect 8 pulses per channel in Data Set.xlsx -> extracted_features.csv
verify_final.py           Smokes the saved model, checks deliverables/figures exist
figures.py                Report figures (Pareto-free, 300 dpi): responses, features, PCA/LDA, confusion, flow
generate_review_visuals.py First-review visuals from Data Set.xlsx
dl_1081.py                Download + extract UCI dataset 1081
dl_probe.py               Fetch UCI dataset page HTML (debugging download URL)
list_files.py             Probe HuggingFace smell-net dataset tree (exploratory)
model.joblib              Trained model (joblib)
model_meta.json           Feature list + class list for the saved model
extracted_features.csv    Cached features from Data Set.xlsx (24 rows)
uci1081/gsalc.csv         UCI 1081 raw CSV
uci1081.zip               Raw UCI download
data sets/                Own sensor measurements (Excel)
Literature Survey/        Background PPTX/DOCX
figures/                  Generated PNGs (report + review visuals)
laborator_README.md       Dataset card for the synthetic MOX dataset
page.html                 Cached UCI dataset page (artifact, not source)
```

---

## 8. Reproduce

```bash
pip install numpy pandas scipy scikit-learn xgboost joblib matplotlib openpyxl requests
```

```bash
python dl_1081.py            # fetch UCI 1081 (if uci1081/ is missing)
python train.py              # 3-class baseline, prints grouped CV acc/F1
python segment_and_extract.py# Data Set.xlsx -> extracted_features.csv
python full_retrain.py       # final 4-class model -> model.joblib
python figures.py            # report figures -> figures/
python generate_review_visuals.py
python verify_final.py       # smoke test + deliverable check
```

Self-check built into `pipeline.py` `__main__` asserts recovered peak and `tau2`
within tolerance for all 4 synthetic gases:

```bash
python pipeline.py
```

---

## 9. Known limitations / caveats

- **Synthetic data is simplified**: no drift, aging, cross-sensitivity, or nonlinear
  humidity. It exists to bootstrap the methanol class and validate feature extraction,
  not as a substitute for real measurements.
- **Single sensor, small real set**: 90 UCI recordings (10 sensors each) + 24 pulses
  from one own recording. Results are indicative, not production-grade.
- **`segment_and_extract.py` normalization quirk**: it passed already-normalized
  `S(t)` as the `resistance` column into `extract_features`, which normalized again.
  Consequence: `r0` in `extracted_features.csv` was negative/relative (-0.2 to -1.0),
  blowing up `peak_response` to 100–1080 and `rise_rate` to 2–18. Merging this with
  UCI (0.01–0.10) and uncalibrated synthetic data created a multi-scale domain gap
  where real 100 ppm sensor signals collapsed to Acetone. Fixed by single-scale normalization.
- **Current to Resistance Physics Contract**: For sensor recordings measuring current
  (`Current_nA` in `data sets/new/ZnO_VOC_Feature_Extraction_100ppm.xlsx`), resistance
  is $R = V/I$. Because $R \propto 1/I$, the resistance relative response relates to
  the current response by:
  $$|S_{\text{resistance}}| = \frac{\Delta I / I_0}{1 + \Delta I / I_0}$$
  Real ZnO measurements at 100 ppm yield: Methanol = 0.191, Acetone = 0.251,
  Isopropanol = 0.282, Ethanol = 0.306, with $\tau_{ad,1}$ monotonically separating
  the 4 gases (0.19 s -> 0.63 s -> 4.98 s -> 10.62 s).
- **Recovery features are NaN for UCI** (no recovery phase recorded): models use
  only the 5 adsorption-side features. Non-NaN-safe transforms (PCA/LDA in `figures.py`)
  median-impute for projection only.
- **Full retrain uses a non-grouped split**, which can overstate accuracy relative
  to the grouped-CV reports.
- `page.html` and `graphify_output.txt` are generated artifacts and are not part of
  the pipeline.

---

## 10. Extension points

- Add a gas: append to `pipeline.GAS_PARAMS` (synthetic) and to the label set used in
  training; add real recordings under the same `time`/`resistance` contract.
- Swap the classifier: anything with `fit` / `predict_proba` works; keep
  `model_meta.json` in sync (`features`, `gases`).
- Real BME688 support: a recording driver only has to emit `time`/`resistance` and the
  whole pipeline above is reusable unchanged.

---

## 11. Strengths worth preserving

- **Single data contract**: every source (UCI CSV, custom Excel, synthetic curves) is
  normalized to a `time`/`resistance` DataFrame before feature extraction, so new data
  sources plug in without changing the pipeline.
- **Feature robustness**: model features are protocol-invariant and sensor-normalized
  (T90 response time, rise rate, `tau` constants) so the model learns gas kinetics,
  not a single sensor's baseline drift.
- **Honest validation**: grouped CV keeps the 10 sensors of one recording out of
  different folds; trainer scripts report real-vs-synthetic accuracy separately so
  synthetic augmentation's effect on real-data performance is visible.
- **Transparent limitations**: documents the double-normalization quirk in
  `segment_and_extract.py`, missing recovery data in UCI, and grouped vs. non-grouped
  CV discrepancies up front — saving future debugging time.

---

## 12. Deployment & hardware — recommended defaults

The repo documents none of these. Recommended starting position (YAGNI: add only when
measured need justifies it), overridable later:

1. **Heater profile**: run a **constant heater temperature**. Don't add a pulsed /
   stepped program until selectivity measurements show it's needed. Document the
   chosen temperature alongside every recording.
2. **Environmental compensation**: **don't add T/RH features yet**. Record ambient
   temperature and humidity if your hardware can, but only feed them into the model
   once drift or cross-sensitivity is actually observed.
3. **Desorption dependency**: **stick with the 5 adsorption-side features** for speed.
   Only add `tau_de` recovery features if accuracy falls short and you can record full
   injection→recovery cycles.
4. **Deployment target**: **`model.joblib` on Python/PC** for now. No embedded
   porting until you have a target device in hand; if you later target an ARM
   Cortex-M3, use `emLearn` or `treelite` to port the XGBoost/RF, or swap in a smaller
   ONNX/TFLite model.

These are implementation decisions, not pipeline constraints — the `time`/`resistance`
data contract and 5-feature vector stay the same regardless of which option you pick.

---

## 13. Metadata sidecar contract (lab practice)

The signal contract (`time`, `resistance`) is not enough for lab data. MOX response
depends on ambient conditions and sensor age, so every lab recording must save a
JSON sidecar next to its raw CSV/Excel so you can later distinguish algorithm drift
from physical sensor aging.

**File:** `<recording>.meta.json` alongside `<recording>.csv` (or `.xlsx`).

```json
{
  "session_id": "2026-11-12-01",
  "sensor_id": "ZnO-01",
  "sensor_age_hours": 48,
  "ambient_temp_c": 23.5,
  "ambient_humidity_rh": 45.0,
  "heater_voltage_v": 5.0,
  "heater_pwm_duty": 100,
  "baseline_r0_ohms": 9800000.0,
  "baseline_start_timestamp": "2026-11-12T09:00:00Z",
  "gas": "ethanol",
  "concentration_ppm": 50,
  "notes": "first lab session, fresh sensor"
}
```

No model code reads this yet — it is purely diagnostic. Record it anyway; it costs
one line of Python and is the only way to debug a failure a month from now.

---

## 14. Edge deployment limitations (LPC1769 ARM Cortex-M3)

The pipeline's PC-side feature extraction is not directly portable to bare metal:

- `_fit_two_term` (`pipeline.py:71`) runs a bounded non-linear least-squares
  exponential fit (`scipy.optimize.curve_fit`) — **do not port this to the LPC1769**.
  The Cortex-M3 has no hardware FPU; iterative exponential curve fitting will be a
  severe bottleneck.
- Savitzky-Golay smoothing (`pipeline.py:31`) requires floating-point matrix math;
  on the M3 prefer a simple moving average or fixed-point discrete derivative.

**Porting strategy (future):** replace `_fit_two_term` with a cheaper computational
proxy — e.g., integral of curve area, time-to-peak, or fixed-point first/second
differences — and smooth with a fixed-point moving average. Document the proxy's
accuracy trade-off against `scipy` before committing to it.

This section is a constraint notice, not a task to implement today.

---

## 15. Long-term drift calibration (lab practice)

MOX sensors permanently drift over weeks; a 50 ppm ethanol peak on Day 1 will not
look the same on Day 30. `S(t) = (R(t) − R0) / R0` normalizes within a session but
not across sessions.

**Required per lab session:**

1. **10-minute clean-air purge** before any gas pulse (establishes a fresh R0).
2. **Single reference pulse** of a known gas (e.g., 50 ppm ethanol) to verify the
   sensor response is still within the model's training distribution.
3. Log the reference-pulse features in the metadata sidecar (§13). If they drift
   beyond tolerance, recalibrate or retire the sensor before collecting more data.

Do not build drift-compensation models yet — just enforce this protocol so you have
the data to detect drift when it happens.

---

## 16. External dataset validation

The model has only 12 real 100 ppm recordings, so generalization is checked against
two open datasets that are **not** in the training set. The goal is to separate two
questions that a single accuracy number conflates:

1. **Are the features valid?** Do the 5 transient features actually separate the
   gases *in-domain* (grouped CV on the external data itself)?
2. **Does the model transfer?** Does the ZnO model classify a *different sensor
   platform* correctly?

Code:

| File | Role |
|---|---|
| `external_datasets.py` | Adapters that turn each dataset into the `time`/`resistance` contract (§2). `__main__` self-check. |
| `validate_external.py` | Runs feature extraction + model prediction on both sets; writes `flow_modulation_features.csv`, `zenodo_drift_features.csv`. |

Both datasets live under `data_external/` (≈146 MB — keep out of version control;
there is no `.gitignore` yet).

### 16.1 UCI-308 — Gas sensor array under flow modulation

- **Source:** UCI ML Repository, dataset id 308, CC-BY.
  `https://archive.ics.uci.edu/static/public/308/gas+sensor+array+under+flow+modulation.zip`
- **Contents:** 16 MOX sensors, 58 time series, **25 Hz**, 5-minute exposures,
  12 concentration classes of acetone/ethanol/mixtures. We use the pure
  **acetone vs ethanol** subsets only.
- **File:** `data_external/flow_modulation/rawdata.csv.gz` (4.76 MB, 928 rows =
  58 exp × 16 sensors, 7509 columns).
- **Schema:** `sensor,sample,exp,batch,ace_conc,eth_conc,gas,lab,col,dR_t1..dR_t7500`.
- **Gotcha — `dR` is normalized ΔR, not ohms.** The stored values are already
  `(R−R0)/R0` with a baseline near 0. `external_datasets.load_flow_modulation`
  reconstructs `resistance = 1 + dR`, which makes `pipeline.relative_response`
  recover exactly the stored `dR` — correct relative response, no double
  normalization (§ rule 2).
- **Gotcha — ~12 s respiration ripple at 25 Hz.** Plain decimation aliases it.
  A **12 s moving average** (`np.convolve(dR, ones(300)/300)`) then 1 Hz decimation
  gives a clean baseline→exposure→recovery envelope (peak ≈ 120 s) that
  `extract_features` handles without the 7500-point `curve_fit` timeout.

### 16.2 Zenodo 15681119 — Long-term drift E-nose

- **Source:** Zenodo, DOI `10.5281/zenodo.15681119` (Nature Sci Data 2025, 141 MB).
- **Contents:** 62 MOX sensors, 40 day-folders (`Day 1` … `Day 40`), 701 CSVs over
  ~12 months. Analytes: **ethanol** (`EtOH_1..6.csv`), diacetyl (0.1/1 ppm),
  2-phenylethanol (200/1000 ppm). Each file = baseline → exposure → recovery,
  5 min each. On-board temp/humidity.
- **Gotcha — space-delimited with a UTF-8 BOM.** Must read
  `encoding='utf-8-sig', delimiter=r'\s+'`. A default comma `DictReader`/`read_csv`
  concatenates the header and raises `KeyError: 'R1[Ohm]'`.
- **Schema:** `Time[hh:mm:ss] Date[yyyy-mm-dd] R1[Ohm]..R62[Ohm] T01[degC] H01[%rh] Cycle_Stage`.
  ~957 rows @ ~1 Hz; `Cycle_Stage` 1/2/3 = baseline/exposure/recovery.
  `load_zenodo_drift(stage=2)` returns the exposure segment.
- **Limitation — ethanol concentration unknown.** `EtOH_*.csv` filenames carry no
  ppm, so ethanol validation is concentration-blind here.

### 16.3 Results (current `model.joblib`)

| Dataset | n | In-domain grouped CV | Current model | Reading |
|---|---|---|---|---|
| UCI-308 flow, acetone vs ethanol | 480 | **0.831** (leave-one-experiment-out) | **0.160** | Features separate the gases (83%); the model fails to transfer across sensor platform. |
| Zenodo drift, ethanol exposure | 232 (39 days) | — | **0.336** ethanol-recall | Predictions scatter acetone/isopropanol; strong per-day feature drift. |

Per-class feature means (UCI-308, Rule 1 — scales are comparable, no renormalization needed):

| gas | peak_response | response_time | rise_rate | tau_ad_1 | tau_ad_2 |
|---|---|---|---|---|---|
| acetone | 0.634 | 51.9 | 0.0167 | 18.7 | 19.3 |
| ethanol | 0.579 | 69.5 | 0.0113 | 19.8 | 21.7 |

Feature drift across 39 days (max day-range / overall mean):

| feature | drift |
|---|---|
| peak_response | 2.23 |
| response_time | 1.63 |
| rise_rate | 1.11 |
| tau_ad_1 | 11.73 |
| tau_ad_2 | 3.06 |

**Interpretation.** The 16% transfer score is a **sensor-domain gap, not a feature
bug**: the same 5 features reach 83% grouped CV on the external sensor. Figaro TGS
array at vol% concentrations vs ZnO at 100 ppm is a genuinely different physical
system. Do not "fix" the model to chase these numbers; treat external accuracy as a
diagnostic, and only merge external data into training if you also apply the ratio
features (§ rule 5) and grouped CV (§ rule 3).

### 16.4 How to run

```powershell
python external_datasets.py     # adapter self-check (480 flow samples, Zenodo parse)
python validate_external.py     # prints both reports, writes *_features.csv
```

Golden-Rule coverage here: Rule 1 (per-class distributions printed), Rule 3
(leave-one-experiment-out grouped CV). Rule 4 (CI assert) and Rule 5 (ratio
features) are intentionally **not** applied — this is validation-only, not a
training merge, so they would be dead code today.

---

## 17. Recent fixes and feature additions

Three follow-ups from the generalization review, all implemented and verified.

### 17.1 `ZNO_PARAMS` — verified synthetic kinetics (`pipeline.py`)
Added alongside the generic `GAS_PARAMS`, holding the physical kinetics measured
from the real 100 ppm ZnO recordings (`session.md`):

| gas | smax | ta (s) | td (s) |
|---|---|---|---|
| methanol | 0.191 | (10.6, 36.5) | (5.0, 40.0) |
| acetone | 0.251 | (0.2, 29.7) | (5.0, 40.0) |
| isopropanol | 0.282 | (5.0, 34.8) | (5.0, 40.0) |
| ethanol | 0.306 | (0.6, 29.4) | (5.0, 40.0) |

`generate_synthetic` gained a `params=` argument (default `GAS_PARAMS`), so
`generate_synthetic(gas, params=ZNO_PARAMS)` now emits realistic ZnO transients.
The slow `ta` needs `expose_s >= 120` to be recovered; with that, the fitted
`tau_ad_2` matches the target within ~1 s for all four gases (verified).

### 17.2 `norm_slope` — concentration-invariant feature (`pipeline.py`)
`extract_features` now returns `norm_slope = rise_rate / peak_response` (NaN when
the peak is 0). Both terms scale with concentration, so the ratio cancels it
(Golden Rule 5). It is stored in the feature dict and in regenerated CSVs, but is
**not** a model input yet — see §4 for the adoption path.

### 17.3 `segment_and_extract.py` — double-normalization fix
The old code pre-normalized each pulse to `(seg − R0)/R0` and then passed it as
`resistance`; `extract_features` normalized *again*, driving `peak_response` to
~100–1000. `Data Set.xlsx` Y-channels are a **zero-baseline response** (baseline ≈ 0,
pulses 40–385), so simply passing raw values also divides by ~0. The fix removes the
pre-normalization **and** adds a fixed `NOMINAL_R0 = 1000` offset to form a valid
`R(t)`. A `peak_response < 10` assertion in `main()` guards against regression.
`extracted_features.csv` was regenerated: peaks are now 0.04–0.39.

---

## 18. Sampling Rate Physics, Robust Fitting & Multi-Model Benchmark

### 18.1 Sampling Rate Discrepancy ($\Delta t = 20\text{ s}$ vs $\Delta t = 0.1\text{ s}$)
* **Real Hardware DAQ**: The physical setup (`Raw_Time_Series` in `ZnO_VOC_...100ppm.xlsx`) samples every **20 seconds**. Linearly interpolating to 1 Hz caps the numerical derivative at $\Delta S / 20\text{ s} \approx 0.0075\text{ }\Omega/\text{s}$.
* **Synthetic Simulator**: Runs at $\Delta t = 0.1\text{ s}$ ($10\text{ Hz}$). Across the fast kinetic phase ($\tau_1 \approx 0.2\text{--}0.6\text{ s}$), `np.gradient` captures instantaneous sub-second slopes yielding $dS/dt \approx 0.18\text{--}0.25\text{ }\Omega/\text{s}$ (a 25× numerical discrepancy).
* **Impact**: Decision trees with fixed thresholds on $dS/dt$ route out-of-distribution real samples into fallback leaves. Linear models (SVM) and regularized shallow trees remain immune by leveraging monotonic thermodynamic scaling ($S_{max}$).

### 18.2 Robust Two-Term Exponential Fitting
When unconstrained curve fitting (`_fit_two_term`) encounters smooth or sparse transients, symmetric parameterization allows $\tau_1$ and $\tau_2$ to collapse into identical values ($\tau_1 \approx \tau_2 \approx 30\text{ s}$, observed in Methanol Cycle 1).
* **Fix**: Added an automated degeneracy guard. If $\tau_2 / \tau_1 < 1.3$ on pulses $\ge 30\text{ s}$, `_fit_two_term` retries with separated fast ($[10^{-3}, 15\text{ s}]$) and slow ($[15\text{ s}, ub]$) boundaries.
* **Result**: Methanol Cycle 1 recovered to $\tau_1 = 0.87\text{ s}, \tau_2 = 38.68\text{ s}$ (matching Cycles 2 & 3 at $0.93\text{ s}$ and $1.29\text{ s}$).
* **Holdout Impact**: Retrained XGBoost holdout accuracy rose from **83.3% to 91.7% (11/12 correct)** with 100% precision and recall on Methanol ($99.85\%$ confidence).

### 18.3 Multi-Model Benchmark (Real 100 ppm ZnO Holdout)

| Model Architecture | Features Used | Evaluation Protocol | Accuracy | Key Strengths |
| :--- | :--- | :--- | :--- | :--- |
| **Linear SVM** | $S_{max}$ alone | Leave-One-Cycle-Out (LOCO) | **100.0% (12/12)** | Zero hyperparameter sensitivity; optimal linear margin |
| **Linear SVM** | $S_{max} + dS/dt$ | Leave-One-Cycle-Out (LOCO) | **100.0% (12/12)** | Monotonic separation across all 4 gases |
| **Logistic Regression** | $S_{max} + dS/dt$ | Leave-One-Cycle-Out (LOCO) | **100.0% (12/12)** | Fast, calibrated probabilistic outputs |
| **Random Forest (`max_depth=2`)** | All 4 kinetic features | Leave-One-Cycle-Out (LOCO) | **100.0% (12/12)** | Robust against single-feature threshold drift |
| **Retrained XGBoost** | All 5 features | Holdout (100 ppm) | **91.7% (11/12)** | High precision (Acetone 100%, Isopropanol 100%, Methanol 100%) |
| **Legacy Multi-Sensor** | All 5 features | Holdout (100 ppm) | **25.0% (3/12)** | Collapsed all predictions to Acetone |

### 18.4 Camera-Ready IEEE Figures (300 DPI)
All figures generated by `python generate_ieee_figures.py` into `figures/`:
* `fig6_confusion_retrained.png`: 4×4 confusion matrix on real 100 ppm holdout (91.7% accuracy).
* `fig7_feature_importance.png`: XGBoost split gain breakdown ($dS/dt = 47.3\%$).
* `fig8_feature_violin.png`: Clean per-gas distributions for $S_{max}, dS/dt, \tau_1, \tau_2$.
* `fig9_accuracy_comparison.png`: Architectural progression (25% → 91.7% → 100% SVM), legend overlap resolved.
* `fig10_real_feature_scatter.png`: 2D scatter plots demonstrating linear cluster separability.
* `fig11_training_distribution.png`: Complete 4-class training distributions (including Methanol).
* `fig12_confidence_analysis.png`: Prediction confidence markers across all 12 holdout recordings.




