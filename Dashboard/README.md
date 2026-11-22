# VOC-Predict — MLOps & Sensor Dashboard

Modern React + Vite + Tailwind CSS + Lucide Icons + Framer Motion analytics dashboard designed with shadcn/ui aesthetic principles.

## Features
- **Live Monitor & Inference**: Real-time transient response waveform S(t), hardware UART COM connection controls, gas probability breakdown with custom chemical markers, and 5-feature extraction indicators.
- **Model Performance & Clusters**: Out-of-fold 4-class Confusion Matrix with grouped CV, XGBoost Feature Importance gain rankings, and 2D PCA / LDA manifold cluster embeddings.
- **Signal Pipeline Architecture**: Six-stage pipeline breakdown ($R_0$ Baseline -> $S(t)$ Normalization -> Savitzky-Golay -> Exposure Onset -> Dual-Exponential Fit -> XGBoost).
- **Research Figures Gallery**: Interactive 300 DPI publication figure viewer with lightbox modal zoom and high-res asset downloads.
- **Lab Test Bench**: Smoke testing on pre-segmented laboratory pulses (`Data Set.xlsx`) and synthetic transients.

## Development & Build

```bash
# Navigate to Dashboard folder
cd Dashboard

# Start development server (runs on localhost:5173)
npm run dev

# Build production bundle
npm run build

# Preview production build
npm run preview
```
