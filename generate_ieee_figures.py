#!/usr/bin/env python3
"""
Generate IEEE-quality publication figures (300 DPI) for the VOC-Predict paper.

Outputs 7 figures to figures/ directory:
  fig6_confusion_retrained.png   – Confusion matrix on real 100 ppm holdout (retrained model)
  fig7_feature_importance.png    – XGBoost feature importance bar chart
  fig8_feature_violin.png        – Per-gas feature distributions (violin plots)
  fig9_accuracy_comparison.png   – Before/after accuracy comparison
  fig10_real_feature_scatter.png – 2D scatter of real holdout features colored by gas
  fig11_training_distribution.png– Training data feature histograms per gas
  fig12_confidence_analysis.png  – Model confidence per prediction (correct vs incorrect)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
import joblib
import json

# ── IEEE Style ──────────────────────────────────────────────────────
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.2,
})

OUT = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT, exist_ok=True)

# ── Load data ───────────────────────────────────────────────────────
val = pd.read_csv('new_dataset_validation.csv')
train_extracted = pd.read_csv('extracted_features.csv')
model = joblib.load('model.joblib')
meta = json.load(open('model_meta.json'))
features = meta['features']
gases = meta['gases']

# Build complete 4-class training representation (extracted + calibrated synthetic)
# so Fig 11 shows all 4 classes including Methanol
import pipeline
synth_rows = []
for gas in ['acetone', 'ethanol', 'isopropanol', 'methanol']:
    for seed in range(10):
        for conc in [0.8, 1.0, 1.2]:
            df_s = pipeline.generate_synthetic(gas, seed=seed, concentration=conc, params=pipeline.ZNO_PARAMS, expose_s=150.0)
            f_s = pipeline.extract_features(df_s)
            synth_rows.append({k: f_s[k] for k in features} | {'gas': gas})
train_synth = pd.DataFrame(synth_rows)
train = pd.concat([train_extracted, train_synth], ignore_index=True)

# Color palette – color-blind friendly
COLORS = {
    'acetone':     '#0072B2',  # blue
    'ethanol':     '#D55E00',  # vermillion
    'isopropanol': '#009E73',  # green
    'methanol':    '#CC79A7',  # pink
}
GAS_ORDER = ['acetone', 'ethanol', 'isopropanol', 'methanol']
GAS_LABELS = ['Acetone', 'Ethanol', 'Isopropanol', 'Methanol']

# ═══════════════════════════════════════════════════════════════════
# Fig 6 – Confusion Matrix (retrained model, real 100 ppm holdout)
# ═══════════════════════════════════════════════════════════════════
print("Generating fig6_confusion_retrained.png ...")
fig, ax = plt.subplots(figsize=(4.5, 4.0))

cm = np.zeros((4, 4), dtype=int)
for _, row in val.iterrows():
    t = GAS_ORDER.index(row['gas_true'])
    p = GAS_ORDER.index(row['pred'])
    cm[t, p] += 1

im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=3)
fig.colorbar(im, ax=ax, shrink=0.8, label='Count')

for i in range(4):
    for j in range(4):
        color = 'white' if cm[i, j] >= 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=16, fontweight='bold', color=color)

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(GAS_LABELS, rotation=30, ha='right')
ax.set_yticklabels(GAS_LABELS)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix — Real 100 ppm Holdout\n(Retrained XGBoost, Accuracy = 91.7%)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig6_confusion_retrained.png'))
plt.close(fig)
print("  ✓ fig6_confusion_retrained.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 7 – Feature Importance
# ═══════════════════════════════════════════════════════════════════
print("Generating fig7_feature_importance.png ...")
fig, ax = plt.subplots(figsize=(5.0, 3.0))

importances = model.feature_importances_
feat_labels = ['Peak\nResponse', 'Response\nTime', 'Rise\nRate', r'$\tau_{ad,1}$', r'$\tau_{ad,2}$']
sort_idx = np.argsort(importances)[::-1]
sorted_imp = importances[sort_idx]
sorted_lbl = [feat_labels[i] for i in sort_idx]
colors = ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7']

bars = ax.bar(range(5), sorted_imp, color=colors, edgecolor='#333333', linewidth=0.6)
ax.set_xticks(range(5))
ax.set_xticklabels(sorted_lbl)
ax.set_ylabel('Feature Importance (Gain)')
ax.set_title('XGBoost Feature Importances')
ax.set_ylim(0, max(sorted_imp) * 1.15)

for bar, v in zip(bars, sorted_imp):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig7_feature_importance.png'))
plt.close(fig)
print("  ✓ fig7_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 8 – Feature Violin Plots (per gas, real 100 ppm data)
# ═══════════════════════════════════════════════════════════════════
print("Generating fig8_feature_violin.png ...")
feat_plot = ['peak_response', 'rise_rate', 'tau_ad_1', 'tau_ad_2']
feat_names = ['Peak Response ($S_{max}$)', 'Rise Rate ($dS/dt$)',
              r'$\tau_{ad,1}$ (s)', r'$\tau_{ad,2}$ (s)']

fig, axes = plt.subplots(1, 4, figsize=(10, 3.2))

for ax, feat, name in zip(axes, feat_plot, feat_names):
    data_by_gas = []
    for g in GAS_ORDER:
        vals = val[val['gas_true'] == g][feat].values
        data_by_gas.append(vals)

    parts = ax.violinplot(data_by_gas, positions=range(4), showmeans=True,
                          showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(list(COLORS.values())[i])
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(1.5)

    # overlay individual points
    for i, g in enumerate(GAS_ORDER):
        vals = val[val['gas_true'] == g][feat].values
        ax.scatter([i]*len(vals), vals, c=COLORS[g], s=30, zorder=5,
                   edgecolors='black', linewidth=0.5)

    ax.set_xticks(range(4))
    ax.set_xticklabels(GAS_LABELS, rotation=35, ha='right', fontsize=8)
    ax.set_title(name, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Feature Distributions — Real 100 ppm ZnO Sensor Data', fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig8_feature_violin.png'))
plt.close(fig)
print("  ✓ fig8_feature_violin.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 9 – Before/After Accuracy Comparison
# ═══════════════════════════════════════════════════════════════════
print("Generating fig9_accuracy_comparison.png ...")
fig, ax = plt.subplots(figsize=(6.5, 3.8))

categories = ['Overall', 'Acetone', 'Ethanol', 'Isopropanol', 'Methanol']
legacy_acc = [25.0, 100.0, 0.0, 0.0, 0.0]
retrained_acc = [91.7, 100.0, 66.7, 100.0, 100.0]
svm_acc = [100.0, 100.0, 100.0, 100.0, 100.0]

x = np.arange(len(categories))
width = 0.26

bars1 = ax.bar(x - width, legacy_acc, width, label='Legacy (UCI + polluted)',
               color='#D32F2F', edgecolor='#333', linewidth=0.5, alpha=0.85)
bars2 = ax.bar(x, retrained_acc, width, label='Retrained XGBoost (91.7%)',
               color='#388E3C', edgecolor='#333', linewidth=0.5, alpha=0.85)
bars3 = ax.bar(x + width, svm_acc, width, label='Linear SVM / LOCO CV (100%)',
               color='#1976D2', edgecolor='#333', linewidth=0.5, alpha=0.85)

for bar, v in zip(bars1, legacy_acc):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=7.5, color='#D32F2F', fontweight='bold')
for bar, v in zip(bars2, retrained_acc):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5, color='#2E7D32', fontweight='bold')
for bar, v in zip(bars3, svm_acc):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
            f'{v:.0f}%', ha='center', va='bottom', fontsize=7.5, color='#1565C0', fontweight='bold')

ax.set_ylabel('Accuracy (%)')
ax.set_title('Classification Performance on Real 100 ppm Holdout Data')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 130)
ax.axhline(y=25, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
ax.legend(loc='upper right', framealpha=0.92, fontsize=8, ncol=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig9_accuracy_comparison.png'))
plt.close(fig)
print("  ✓ fig9_accuracy_comparison.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 10 – 2D Feature Scatter (Peak Response vs Rise Rate)
# ═══════════════════════════════════════════════════════════════════
print("Generating fig10_real_feature_scatter.png ...")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

# Panel A: peak_response vs rise_rate
ax = axes[0]
for g, lbl in zip(GAS_ORDER, GAS_LABELS):
    mask = val['gas_true'] == g
    ax.scatter(val.loc[mask, 'peak_response'], val.loc[mask, 'rise_rate'],
               c=COLORS[g], label=lbl, s=70, edgecolors='black', linewidth=0.6, zorder=5)
    # Mark misclassified
    mis = mask & (~val['correct'])
    if mis.any():
        ax.scatter(val.loc[mis, 'peak_response'], val.loc[mis, 'rise_rate'],
                   c='none', edgecolors='red', linewidth=2, s=120, marker='o', zorder=6)

ax.set_xlabel('Peak Response ($S_{max}$)')
ax.set_ylabel('Rise Rate ($dS/dt$, Ω/s)')
ax.set_title('(a) Peak Response vs. Rise Rate')
ax.legend(fontsize=8, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: tau_ad_2 vs peak_response
ax = axes[1]
for g, lbl in zip(GAS_ORDER, GAS_LABELS):
    mask = val['gas_true'] == g
    ax.scatter(val.loc[mask, 'tau_ad_2'], val.loc[mask, 'peak_response'],
               c=COLORS[g], label=lbl, s=70, edgecolors='black', linewidth=0.6, zorder=5)
    mis = mask & (~val['correct'])
    if mis.any():
        ax.scatter(val.loc[mis, 'tau_ad_2'], val.loc[mis, 'peak_response'],
                   c='none', edgecolors='red', linewidth=2, s=120, marker='o', zorder=6)

ax.set_xlabel(r'Adsorption Time Constant $\tau_{ad,2}$ (s)')
ax.set_ylabel('Peak Response ($S_{max}$)')
ax.set_title(r'(b) $\tau_{ad,2}$ vs. Peak Response')
ax.legend(fontsize=8, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add "misclassified" annotation
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markeredgecolor='red',
                          markeredgewidth=2, markersize=10, label='Misclassified')]
axes[0].legend(handles=axes[0].get_legend_handles_labels()[0] + legend_elements[:1],
               labels=[l for l in GAS_LABELS] + ['Misclassified'],
               fontsize=8, loc='upper left')

fig.suptitle('Real 100 ppm Holdout — Feature Space Separation', y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig10_real_feature_scatter.png'))
plt.close(fig)
print("  ✓ fig10_real_feature_scatter.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 11 – Training Data Distribution (Histograms per gas)
# ═══════════════════════════════════════════════════════════════════
print("Generating fig11_training_distribution.png ...")
fig, axes = plt.subplots(1, 4, figsize=(10, 3.0))

for ax, feat, name in zip(axes, feat_plot, feat_names):
    for g, lbl in zip(GAS_ORDER, GAS_LABELS):
        vals = train[train['gas'] == g][feat].dropna()
        if len(vals) > 0:
            ax.hist(vals, bins=8, alpha=0.5, label=lbl, color=COLORS[g],
                    edgecolor=COLORS[g], linewidth=0.5)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Count')
axes[-1].legend(fontsize=7, loc='upper right')
fig.suptitle('Training Feature Distributions (ZnO Domain: Clean Extracted + Calibrated)', fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig11_training_distribution.png'))
plt.close(fig)
print("  ✓ fig11_training_distribution.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 12 – Prediction Confidence Analysis
# ═══════════════════════════════════════════════════════════════════
print("Generating fig12_confidence_analysis.png ...")
fig, ax = plt.subplots(figsize=(6, 3.5))

for i, (_, row) in enumerate(val.iterrows()):
    color = COLORS[row['gas_true']]
    marker = 'o' if row['correct'] else 'X'
    edge = 'black' if row['correct'] else 'red'
    size = 80 if row['correct'] else 120
    ax.scatter(i, row['confidence'], c=color, marker=marker, s=size,
               edgecolors=edge, linewidth=1.0, zorder=5)

# Add gas labels on x-axis
xlabels = [f"{row['voc']}\nC{row['cycle']}" for _, row in val.iterrows()]
ax.set_xticks(range(len(val)))
ax.set_xticklabels(xlabels, fontsize=7, rotation=45, ha='right')
ax.set_ylabel('Prediction Confidence')
ax.set_title('Per-Sample Prediction Confidence (Real 100 ppm Holdout)')
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.7, label='Chance level')
ax.set_ylim(0.4, 1.05)

# Custom legend
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markeredgecolor='black', markersize=8, label='Correct'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
           markeredgecolor='red', markersize=10, label='Misclassified'),
]
for g, lbl in zip(GAS_ORDER, GAS_LABELS):
    handles.append(Line2D([0], [0], marker='s', color='w',
                          markerfacecolor=COLORS[g], markersize=8, label=lbl))
ax.legend(handles=handles, fontsize=7, loc='lower right', ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig12_confidence_analysis.png'))
plt.close(fig)
print("  ✓ fig12_confidence_analysis.png")

print(f"\n✅ All 7 figures saved to {OUT}/")
print("   fig6_confusion_retrained.png")
print("   fig7_feature_importance.png")
print("   fig8_feature_violin.png")
print("   fig9_accuracy_comparison.png")
print("   fig10_real_feature_scatter.png")
print("   fig11_training_distribution.png")
print("   fig12_confidence_analysis.png")
