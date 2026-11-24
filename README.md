# VOC-Predict: Selective VOC Identification Using a Single ZnO Gas Sensor

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-91.7%25%20Holdout-green.svg)](full_retrain.py)
[![Linear SVM](https://img.shields.io/badge/Linear%20SVM-100%25%20LOCO--CV-brightgreen.svg)](evaluate_model.py)

**VOC-Predict** is a machine learning system for classifying Volatile Organic Compounds (VOCs)—**Acetone, Ethanol, Isopropanol, and Methanol**—using dynamic transient kinetics from a **single nanostructured Zinc Oxide (ZnO) chemoresistive sensor**, bypassing the need for multi-sensor e-nose arrays.

---

## 🚀 Performance Highlights

Evaluated on **12 real 100 ppm physical sensor recordings** (3 independent measurement cycles × 4 gases):

| Model Architecture | Feature Subset | Evaluation Scheme | Overall Accuracy | Acetone | Ethanol | Isopropanol | Methanol |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM** | $S_{max}$ alone | Leave-One-Cycle-Out (LOCO) | **100.0% (12/12)** | 100% | 100% | 100% | 100% |
| **Linear SVM** | $S_{max} + dS/dt$ | Leave-One-Cycle-Out (LOCO) | **100.0% (12/12)** | 100% | 100% | 100% | 100% |
| **Logistic Regression** | $S_{max} + dS/dt$ | Leave-One-Cycle-Out (LOCO) | **100.0% (12/12)** | 100% | 100% | 100% | 100% |
| **Random Forest (`max_depth=2`)** | $S_{max}, dS/dt, \tau_1, \tau_2$ | Leave-One-Cycle-Out (LOCO) | **100.0% (12/12)** | 100% | 100% | 100% | 100% |
| **Retrained XGBoost** | All 5 features | Unseen 100 ppm Holdout | **91.7% (11/12)** | 100% | 66.7% | 100% | 100% |
| **Legacy Multi-Sensor** | All 5 features | Unseen 100 ppm Holdout | **25.0% (3/12)** | 100% | 0.0% | 0.0% | 0.0% |

* **Zero Cluster Overlap:** All 4 gases exhibit clean, non-overlapping thermodynamic response boundaries ($S_{max}$: Methanol $0.19$ < Acetone $0.25$ < Isopropanol $0.28$ < Ethanol $0.31$).
* **Robust Kinetics:** Automated non-degenerate exponential fitting separates fast adsorption ($\tau_1 \le 15\text{ s}$) from slow diffusion kinetics ($\tau_2 \ge 15\text{ s}$), eliminating estimation variance.

---

## 📁 Repository Layout

```
voc-predict/
├── pipeline.py                 # Core physics engine: Savitzky-Golay, onset, feature extraction
├── segment_and_extract.py      # Segments multi-pulse lab data (NOMINAL_R0=1000 offset)
├── full_retrain.py             # Trains domain-aligned 4-class classifier on ZnO features
├── evaluate_model.py           # Per-class metrics & Leave-One-Cycle-Out CV benchmark
├── validate_new_dataset.py     # Continuous holdout validation on 12 real 100 ppm recordings
├── verify_final.py             # Deliverable regression & sanity check
├── generate_ieee_figures.py    # Generates all 7 camera-ready IEEE publication figures (300 DPI)
├── model.joblib                # Trained 4-class model artifact
├── model_meta.json             # Feature names and class label mappings
├── runs.jsonl                  # Append-only run tracking log
│
├── api/                        # FastAPI inference microservice
│   └── main.py                 # Endpoints: /predict, /health, /features
├── Dashboard/                  # React + Vite + Tailwind interactive UI
│   ├── src/                    # Visual components & prediction charts
│   └── package.json            # Node dependencies
├── streamlit_app.py            # Streamlit lab signal debugger (port 8501)
│
├── data sets/                  # In-repo experimental sensor data
│   ├── Train/                  # Data Set.xlsx (multi-pulse laboratory recordings)
│   └── new/                    # ZnO_VOC_Feature_Extraction_100ppm.xlsx (12 real holdouts)
├── figures/                    # Publication figures (fig1-12, visual1-3)
├── docs/                       # Project documentation & laboratory protocols
│   ├── LAB_PROTOCOL.md         # Standard operating procedure (SOP) for sensor collection
│   └── reference/              # Reference dataset cards (BME688 specification)
├── tools/                      # Offline utilities & maintenance scripts
│   ├── data_fetch/             # Scrape & open-data download helpers
│   ├── archive/                # Deprecated prototype scripts
│   └── scratch/                # Temporary inspection logs
├── ARCHITECTURE.md             # Complete system architecture specification (§1 - §18)
└── session.md                  # Development history, retrain logs & benchmark log
```

---

## 🛠️ Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/Leela-Sai-Vardhan/voc-predict.git
cd voc-predict
pip install -r requirements.txt
```

### 2. Feature Extraction & Verification
Extract dynamic features from the laboratory pulse recordings:
```bash
python segment_and_extract.py
python verify_final.py
```

### 3. Model Retraining & Benchmarking
Train on the domain-specific ZnO feature distribution and run the multi-model benchmark:
```bash
# Retrain 4-class XGBoost model
python full_retrain.py

# Run comprehensive Leave-One-Cycle-Out (LOCO) CV evaluation
python evaluate_model.py
```

### 4. Generate IEEE Publication Figures
Produces all 7 camera-ready 300 DPI PNG figures in `figures/`:
```bash
python generate_ieee_figures.py
```

### 5. Launch Inference Microservice & Dashboard
```bash
# Start FastAPI backend (port 8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start Streamlit lab signal debugger (port 8501)
streamlit run streamlit_app.py
```

---

## 🔬 Scientific Core: Physics & Features

### Current-to-Resistance Inversion
For an n-type semiconductor gas sensor (where reducing VOC gases donate electrons to increase conduction current $I$):
$$S_{\text{resistance}} = \frac{R_0 - R(t)}{R_0} = \frac{\Delta I / I_0}{1 + \Delta I / I_0}$$

### Model Features (Protocol-Invariant)
1. **$S_{\text{max}}$ (Peak Response):** Relative response magnitude at equilibrium exposure.
2. **$t_{90}$ (Response Time):** Time elapsed from onset to $90\%$ of peak response.
3. **$dS/dt$ (Rise Rate):** Maximum numerical gradient during initial gas adsorption.
4. **$\tau_{\text{ad}, 1}$ (Fast Kinetic Constant):** Sub-second surface chemisorption rate constant.
5. **$\tau_{\text{ad}, 2}$ (Slow Kinetic Constant):** Bulk/grain-boundary diffusion time constant.

---

## 📚 Documentation Links

* **[ARCHITECTURE.md](ARCHITECTURE.md):** Master technical reference (§1–§18) detailing mathematical proofs, the 5 Golden Rules, hardware DAQ physics ($\Delta t = 20\text{ s}$ vs $\Delta t = 0.1\text{ s}$), and external validation on UCI-308 and Zenodo drift datasets.
* **[docs/LAB_PROTOCOL.md](docs/LAB_PROTOCOL.md):** Experimental standard operating procedure (SOP), sensor warm-up, and drift-canary reference pulse tracking.
* **[session.md](session.md):** Session changelog, validation history, and reproduction logs.

---

## 📄 License
This project is licensed under the Apache 2.0 / MIT License.
