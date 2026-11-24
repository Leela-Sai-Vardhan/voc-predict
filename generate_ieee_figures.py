#!/usr/bin/env python3
"""
Generate IEEE Transactions-Standard Camera-Ready Figures (300 DPI).

Outputs all 7 standardized publication figures to figures/ieee/:
  fig1_transient_responses.png – (a) Measured Current I(t), (b) Chemoresistive Response S(t)
  fig2_feature_boxplots.png    – Standard rectangular box plots of Smax, dS/dt, tau1, tau2
  fig3_feature_scatter.png     – 2D feature space projections showing cluster separation
  fig4_feature_importance.png  – XGBoost feature importance ranking (split gain)
  fig5_confusion_matrix.png    – 4x4 Confusion matrix on 12 real 100 ppm holdouts (91.7% acc)
  fig6_accuracy_comparison.png – Architectural comparison (Legacy 25% vs XGBoost 91.7% vs SVM 100%)
  fig7_confidence_analysis.png – Per-sample prediction confidence and stability across cycles
"""
import os
import json
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

# ── IEEE Style Guide & Typography ────────────────────────────────────
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.25,
    'lines.linewidth': 1.2,
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, 'figures', 'ieee')
os.makedirs(OUT, exist_ok=True)

# ── Load Validation & Training Artifacts ──────────────────────────────
val = pd.read_csv(os.path.join(BASE_DIR, 'new_dataset_validation.csv'))
model = joblib.load(os.path.join(BASE_DIR, 'model.joblib'))
meta = json.load(open(os.path.join(BASE_DIR, 'model_meta.json')))
features = meta['features']
gases = meta['gases']

# Color-blind safe palette (Okabe-Ito derived)
COLORS = {
    'acetone':     '#0072B2',  # Deep Blue
    'ethanol':     '#D55E00',  # Vermillion
    'isopropanol': '#009E73',  # Emerald Green
    'methanol':    '#CC79A7',  # Reddish Purple
}
GAS_ORDER = ['acetone', 'ethanol', 'isopropanol', 'methanol']
GAS_LABELS = ['Acetone', 'Ethanol', 'Isopropanol', 'Methanol']

# ═══════════════════════════════════════════════════════════════════
# Fig 1 – Physical Transient Profiles & Normalized Response
# ═══════════════════════════════════════════════════════════════════
print("Generating fig1_transient_responses.png ...")
raw_path = os.path.join(BASE_DIR, 'data sets', 'new', 'ZnO_VOC_Feature_Extraction_100ppm.xlsx')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))

if os.path.exists(raw_path):
    raw_df = pd.read_excel(raw_path, sheet_name='Raw_Time_Series')
    for g, lbl in zip(GAS_ORDER, GAS_LABELS):
        sub = raw_df[(raw_df['VOC'] == lbl) & (raw_df['Cycle'] == 1)]
        t = sub['Time_within_cycle_s'].to_numpy(float)
        i = sub['Current_nA'].to_numpy(float)
        
        # Dense interpolation for smooth curve
        t_dense = np.linspace(t.min(), t.max(), 200)
        i_dense = np.interp(t_dense, t, i)
        
        # Invert to resistance: R = 1/I, S(t) = (R0 - R)/R0
        r_dense = 1.0 / i_dense
        r0 = r_dense[0]
        s_dense = (r0 - r_dense) / r0
        
        ax1.plot(t_dense, i_dense, label=lbl, color=COLORS[g], linewidth=1.8)
        ax2.plot(t_dense, s_dense, label=lbl, color=COLORS[g], linewidth=1.8)
        
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Sensor Current $I(t)$ (nA)')
    ax1.set_title('(a) Measured Conduction Current', pad=8)
    ax1.axvspan(40, 120, color='gray', alpha=0.12, label='Exposure Phase')
    ax1.legend(loc='upper left', framealpha=0.9)
    ax1.grid(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Chemoresistive Response $S(t) = \\frac{R_0 - R}{R_0}$')
    ax2.set_title('(b) Normalized Dynamic Response', pad=8)
    ax2.axvspan(40, 120, color='gray', alpha=0.12)
    ax2.grid(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

fig.suptitle('Dynamic Physical Sensor Response at 100 ppm Concentration', fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig1_transient_responses.png'))
plt.close(fig)
print("  ✓ fig1_transient_responses.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 2 – Rectangular Box Plots (Feature Separation across Gases)
# ═══════════════════════════════════════════════════════════════════
print("Generating fig2_feature_boxplots.png ...")
feat_plot = ['peak_response', 'rise_rate', 'tau_ad_1', 'tau_ad_2']
feat_titles = [
    r'(a) Peak Response ($S_{\max}$)',
    r'(b) Rise Rate ($dS/dt$, $\Omega$/s)',
    r'(c) Adsorption $\tau_{\text{ad}, 1}$ (s)',
    r'(d) Diffusion $\tau_{\text{ad}, 2}$ (s)'
]

fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.4))

for ax, feat, title in zip(axes, feat_plot, feat_titles):
    order = [val[val['gas_true'] == g][feat].dropna().values for g in GAS_ORDER]
    
    # Standard rectangular boxplot layout
    bp = ax.boxplot(
        order,
        tick_labels=GAS_LABELS,
        patch_artist=True,
        widths=0.52,
        medianprops=dict(color='black', lw=1.6),
        whiskerprops=dict(color='#222222', lw=1.1),
        capprops=dict(color='#222222', lw=1.1),
        boxprops=dict(edgecolor='#222222', lw=1.0)
    )
    
    # Fill rectangular boxes with class colors
    for patch, g in zip(bp['boxes'], GAS_ORDER):
        patch.set_facecolor(COLORS[g])
        patch.set_alpha(0.72)
    
    # Overlay individual cycle measurements (n=3 per gas)
    np.random.seed(42)
    for i, g in enumerate(GAS_ORDER):
        vals = val[val['gas_true'] == g][feat].dropna().values
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(
            np.array([i + 1] * len(vals)) + jitter,
            vals,
            color=COLORS[g],
            edgecolors='black',
            linewidth=0.8,
            s=36,
            zorder=4
        )
        
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

fig.suptitle('Feature Distributions across VOC Gases (Real 100 ppm ZnO Sensor Recordings)', fontsize=12, y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_feature_boxplots.png'))
plt.close(fig)
print("  ✓ fig2_feature_boxplots.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 3 – 2D Feature Space Scatter (Cluster Separation)
# ═══════════════════════════════════════════════════════════════════
print("Generating fig3_feature_scatter.png ...")
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

# Panel A: Smax vs Rise Rate
ax = axes[0]
for g, lbl in zip(GAS_ORDER, GAS_LABELS):
    mask = val['gas_true'] == g
    ax.scatter(val.loc[mask, 'peak_response'], val.loc[mask, 'rise_rate'],
               c=COLORS[g], label=lbl, s=75, edgecolors='black', linewidth=0.7, zorder=5)
    mis = mask & (~val['correct'])
    if mis.any():
        ax.scatter(val.loc[mis, 'peak_response'], val.loc[mis, 'rise_rate'],
                   c='none', edgecolors='red', linewidth=2.0, s=130, marker='o', zorder=6)

ax.set_xlabel(r'Peak Response ($S_{\max}$)')
ax.set_ylabel(r'Rise Rate ($dS/dt$, $\Omega$/s)')
ax.set_title(r'(a) Peak Response vs. Rise Rate', pad=8)
ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: tau_ad_2 vs Smax
ax = axes[1]
for g, lbl in zip(GAS_ORDER, GAS_LABELS):
    mask = val['gas_true'] == g
    ax.scatter(val.loc[mask, 'tau_ad_2'], val.loc[mask, 'peak_response'],
               c=COLORS[g], label=lbl, s=75, edgecolors='black', linewidth=0.7, zorder=5)
    mis = mask & (~val['correct'])
    if mis.any():
        ax.scatter(val.loc[mis, 'tau_ad_2'], val.loc[mis, 'peak_response'],
                   c='none', edgecolors='red', linewidth=2.0, s=130, marker='o', zorder=6)

ax.set_xlabel(r'Adsorption Time Constant $\tau_{\text{ad}, 2}$ (s)')
ax.set_ylabel(r'Peak Response ($S_{\max}$)')
ax.set_title(r'(b) $\tau_{\text{ad}, 2}$ vs. Peak Response', pad=8)
ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[g],
                         markeredgecolor='black', markersize=8, label=lbl)
                  for g, lbl in zip(GAS_ORDER, GAS_LABELS)]
legend_handles.append(Line2D([0], [0], marker='o', color='w', markeredgecolor='red',
                             markeredgewidth=2, markersize=10, label='Misclassified'))
axes[0].legend(handles=legend_handles, fontsize=8.5, loc='upper left')

fig.suptitle('Real 100 ppm Holdout — Inter-Gas Cluster Separability', y=1.03, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_feature_scatter.png'))
plt.close(fig)
print("  ✓ fig3_feature_scatter.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 4 – XGBoost Feature Importance
# ═══════════════════════════════════════════════════════════════════
print("Generating fig4_feature_importance.png ...")
fig, ax = plt.subplots(figsize=(5.2, 3.2))

importances = model.feature_importances_
feat_labels = ['Rise Rate\n($dS/dt$)', r'$\tau_{\text{ad}, 1}$', r'$\tau_{\text{ad}, 2}$',
               'Peak\nResponse', 'Response\nTime ($t_{90}$)']
sort_idx = np.argsort(importances)[::-1]
sorted_imp = importances[sort_idx]
sorted_lbl = [feat_labels[i] for i in sort_idx]
bar_colors = ['#1F78B4', '#33A02C', '#FB9A99', '#E31A1C', '#B2DF8A']

bars = ax.bar(range(5), sorted_imp, color=bar_colors, edgecolor='#222222', linewidth=0.7)
ax.set_xticks(range(5))
ax.set_xticklabels(sorted_lbl)
ax.set_ylabel('Split Gain Importance')
ax.set_title('XGBoost Relative Feature Importance', pad=8)
ax.set_ylim(0, max(sorted_imp) * 1.18)

for bar, v in zip(bars, sorted_imp):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
            f'{v:.1%}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.35)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig4_feature_importance.png'))
plt.close(fig)
print("  ✓ fig4_feature_importance.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 5 – Confusion Matrix (Retrained 91.7% Model)
# ═══════════════════════════════════════════════════════════════════
print("Generating fig5_confusion_matrix.png ...")
fig, ax = plt.subplots(figsize=(4.6, 4.0))

cm = np.zeros((4, 4), dtype=int)
for _, row in val.iterrows():
    t_idx = GAS_ORDER.index(row['gas_true'])
    p_idx = GAS_ORDER.index(row['pred'])
    cm[t_idx, p_idx] += 1

im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=3)
fig.colorbar(im, ax=ax, shrink=0.8, label='Number of Samples')

for i in range(4):
    for j in range(4):
        color = 'white' if cm[i, j] >= 2 else 'black'
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=15, fontweight='bold', color=color)

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(GAS_LABELS, rotation=30, ha='right')
ax.set_yticklabels(GAS_LABELS)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix — Real 100 ppm Holdout\n(Retrained XGBoost, Accuracy = 91.7%)', pad=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig5_confusion_matrix.png'))
plt.close(fig)
print("  ✓ fig5_confusion_matrix.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 6 – Model Performance & Architecture Comparison
# ═══════════════════════════════════════════════════════════════════
print("Generating fig6_accuracy_comparison.png ...")
fig, ax = plt.subplots(figsize=(6.5, 3.8))

categories = ['Overall', 'Acetone', 'Ethanol', 'Isopropanol', 'Methanol']
legacy_acc = [25.0, 100.0, 0.0, 0.0, 0.0]
retrained_acc = [91.7, 100.0, 66.7, 100.0, 100.0]
svm_acc = [100.0, 100.0, 100.0, 100.0, 100.0]

x = np.arange(len(categories))
width = 0.26

bars1 = ax.bar(x - width, legacy_acc, width, label='Legacy (UCI + polluted)',
               color='#D32F2F', edgecolor='#222', linewidth=0.6, alpha=0.85)
bars2 = ax.bar(x, retrained_acc, width, label='Retrained XGBoost (91.7%)',
               color='#388E3C', edgecolor='#222', linewidth=0.6, alpha=0.85)
bars3 = ax.bar(x + width, svm_acc, width, label='Linear SVM / LOCO CV (100%)',
               color='#1976D2', edgecolor='#222', linewidth=0.6, alpha=0.85)

for bar, v in zip(bars1, legacy_acc):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8,
                f'{v:.0f}%', ha='center', va='bottom', fontsize=7.5, color='#D32F2F', fontweight='bold')
for bar, v in zip(bars2, retrained_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8,
            f'{v:.1f}%', ha='center', va='bottom', fontsize=7.5, color='#2E7D32', fontweight='bold')
for bar, v in zip(bars3, svm_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.8,
            f'{v:.0f}%', ha='center', va='bottom', fontsize=7.5, color='#1565C0', fontweight='bold')

ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('Classification Performance on Real 100 ppm Holdout Data', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9.5)
ax.set_ylim(0, 130)
ax.axhline(y=25, color='gray', linestyle='--', linewidth=0.6, alpha=0.6)
ax.legend(loc='upper right', framealpha=0.92, fontsize=8.5, ncol=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.35)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig6_accuracy_comparison.png'))
plt.close(fig)
print("  ✓ fig6_accuracy_comparison.png")

# ═══════════════════════════════════════════════════════════════════
# Fig 7 – Prediction Confidence & Stability
# ═══════════════════════════════════════════════════════════════════
print("Generating fig7_confidence_analysis.png ...")
fig, ax = plt.subplots(figsize=(6.2, 3.6))

for i, (_, row) in enumerate(val.iterrows()):
    color = COLORS[row['gas_true']]
    marker = 'o' if row['correct'] else 'X'
    edge = 'black' if row['correct'] else 'red'
    size = 85 if row['correct'] else 130
    ax.scatter(i, row['confidence'], c=color, marker=marker, s=size,
               edgecolors=edge, linewidth=1.1, zorder=5)

xlabels = [f"{row['voc']}\nC{row['cycle']}" for _, row in val.iterrows()]
ax.set_xticks(range(len(val)))
ax.set_xticklabels(xlabels, fontsize=7.5, rotation=45, ha='right')
ax.set_ylabel('Predicted Class Probability')
ax.set_title('Prediction Confidence on Real 100 ppm Holdout', pad=10)
ax.axhline(y=0.25, color='gray', linestyle=':', linewidth=0.7, alpha=0.7, label='Random Chance (25%)')
ax.set_ylim(0.40, 1.05)

handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
           markeredgecolor='black', markersize=8, label='Correct (11/12)'),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
           markeredgecolor='red', markersize=10, label='Misclassified (1/12)'),
]
for g, lbl in zip(GAS_ORDER, GAS_LABELS):
    handles.append(Line2D([0], [0], marker='s', color='w',
                          markerfacecolor=COLORS[g], markersize=8, label=lbl))
ax.legend(handles=handles, fontsize=7.5, loc='lower right', ncol=2, framealpha=0.92)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.35)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig7_confidence_analysis.png'))
plt.close(fig)
print("  ✓ fig7_confidence_analysis.png")

print(f"\n✅ All 7 IEEE camera-ready figures generated successfully in {OUT}/:")
for fn in sorted(os.listdir(OUT)):
    print(f"   {fn}")
