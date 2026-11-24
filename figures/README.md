# Figure Catalog & IEEE Publication Guide

This directory contains visual deliverables organized into camera-ready paper figures and archived preliminary assets.

```
figures/
├── ieee/                       # Camera-ready, 300 DPI figures formatted for IEEE submission
│   ├── fig1_transient_responses.png
│   ├── fig2_feature_boxplots.png
│   ├── fig3_feature_scatter.png
│   ├── fig4_feature_importance.png
│   ├── fig5_confusion_matrix.png
│   ├── fig6_accuracy_comparison.png
│   └── fig7_confidence_analysis.png
│
├── old/                        # Archived preliminary, exploratory, and review assets
│   ├── fig1_synthetic_responses.png    (preliminary synthetic curves)
│   ├── fig2_feature_separation.png     (old multi-sensor UCI boxplot)
│   ├── fig3_projection.png             (old exploratory PCA/LDA)
│   ├── fig4_confusion_real.png         (old 25% collapsed confusion matrix)
│   ├── fig5_pipeline_flow.png          (initial system diagram)
│   ├── fig6_confusion_retrained.png    (intermediate retrained matrix)
│   ├── fig7_feature_importance.png     (intermediate gain bar chart)
│   ├── fig8_feature_violin.png         (deprecated wavy violin plot)
│   ├── fig9_accuracy_comparison.png    (intermediate accuracy chart)
│   ├── fig10_real_feature_scatter.png  (intermediate 2D scatter)
│   ├── fig11_training_distribution.png (training distribution histograms)
│   ├── fig12_confidence_analysis.png   (intermediate confidence plot)
│   └── visual1_3_*.png                 (review presentation visuals)
└── README.md                   # This catalog
```

---

## 📸 IEEE Submission Figure Reference Table

All figures in `figures/ieee/` follow standard **IEEE Transactions format** (Times New Roman serif typography, 300 DPI resolution, Okabe-Ito color-blind friendly palette, and mathematical axis notation).

| Figure in Paper | File Name | Recommended Paper Section | Caption Suggestion |
| :--- | :--- | :--- | :--- |
| **Fig. 1** | `fig1_transient_responses.png` | **II. Sensor Operation & Physical Dynamics** | *(a) Measured conduction current $I(t)$ and (b) normalized chemoresistive response $S(t) = (R_0 - R)/R_0$ of a single nanostructured ZnO sensor across 100 ppm pulses of Acetone, Ethanol, Isopropanol, and Methanol.* |
| **Fig. 2** | `fig2_feature_boxplots.png` | **III. Feature Extraction & Separation** | *Feature distributions across four VOC gases extracted from 12 physical 100 ppm sensor recordings. Rectangular boxplots display interquartile ranges, medians (black solid lines), and overlaid discrete cycle measurements.* |
| **Fig. 3** | `fig3_feature_scatter.png` | **IV. Feature Space Analysis** | *Two-dimensional feature projections demonstrating inter-gas separability: (a) Peak Response $S_{\max}$ versus Rise Rate $dS/dt$, and (b) Adsorption time constant $\tau_{\text{ad}, 2}$ versus Peak Response $S_{\max}$.* |
| **Fig. 4** | `fig4_feature_importance.png` | **V. Classification & Feature Ranking** | *Relative feature gain importance from gradient boosted decision trees. Rise rate $dS/dt$ contributes 47.3% of total split gain, followed by kinetic adsorption parameters.* |
| **Fig. 5** | `fig5_confusion_matrix.png` | **VI. Experimental Results** | *Confusion matrix of the retrained 4-class classifier evaluated on 12 unseen real 100 ppm sensor recordings, achieving 91.7% overall holdout accuracy with 100% precision on Methanol, Acetone, and Isopropanol.* |
| **Fig. 6** | `fig6_accuracy_comparison.png` | **VI. Experimental Results & Discussion** | *Comparative classification performance across model architectures: Legacy cross-sensor model (25.0%), retrained domain-aligned XGBoost (91.7%), and maximum-margin Linear SVM (100.0% under Leave-One-Cycle-Out cross-validation).* |
| **Fig. 7** | `fig7_confidence_analysis.png` | **VI. Experimental Results & Discussion** | *Per-sample prediction confidence across 3 experimental measurement cycles. Correct predictions exhibit high confidence ($0.71\text{--}0.999$), with a single interpolation-induced misclassification marked.* |

---

## 📐 Standardization: Fig. 2 Rectangular Box Plots vs. Fig. 8 Wavy Violins

* **Previous Issue (Deprecated `fig8_feature_violin.png`):** Kernel density estimation on small sample cohorts ($n=3$ measurement cycles per gas) generated wavy, deformed violin profiles that do not adhere to IEEE reporting conventions.
* **Standardized Format (`fig2_feature_boxplots.png`):** Adopted clean rectangular boxes (`widths=0.52`) with crisp black outlines, prominent median bars (`lw=1.6`), solid whisker bounds, and overlaid discrete measurement points (`s=36`). This ensures unambiguous visual interpretation for journal reviewers.

---

## 🔄 Reproduction
To regenerate all camera-ready figures at any time:
```bash
python generate_ieee_figures.py
```
