"""Generate report-ready figures for the VOC classifier write-up."""
import os
from functools import lru_cache
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import pipeline
import train as base
import train_synth as ts

GASES = ["acetone", "ethanol", "isopropanol", "methanol"]
GAS_COL = {"ethanol": "#1f77b4", "methanol": "#ff7f0e",
           "acetone": "#2ca02c", "isopropanol": "#d62728"}

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


@lru_cache(maxsize=1)
def _real():
    return base.build_samples()


@lru_cache(maxsize=1)
def _synth():
    return ts.build_synth()


def _merged():
    xr, _ = _real(); xs, _ = _synth()
    return pd.concat([xr, xs], ignore_index=True)


def fig_pair():
    return plt.subplots(1, 2, figsize=(11, 4.2))


def tidy(ax, t, xl, yl):
    ax.set_title(t, fontsize=12, fontweight="bold")
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    return ax


def fig1_synthetic_responses():
    """Raw resistance trace of one gas + normalized S(t) overlay of all four."""
    f, (a, b) = fig_pair()
    g = "ethanol"
    df = pipeline.generate_synthetic(g, seed=3)
    a.plot(df["time"], df["resistance"] / 1e6, color=GAS_COL[g], lw=1.5)
    a.axvspan(5, 45, color=GAS_COL[g], alpha=0.08); a.axvline(45, color="k", ls=":", lw=1)
    a.annotate("gas on", (47, 9.996), fontsize=9)
    a.annotate("gas off", (6, 9.996), fontsize=9)
    tidy(a, f"Raw ZnO resistance — {g} (synthetic, r\u2080=10 M\u03a9)", "time (s)", "R (M\u03a9)")
    for gas in GASES:
        df = pipeline.generate_synthetic(gas, seed=3)
        s = pipeline.relative_response(df)
        b.plot(df["time"], s, lw=1.8, label=gas, color=GAS_COL[gas])
    b.axvline(5, color="k", ls=":", lw=1); b.axvline(45, color="k", ls=":", lw=1)
    tidy(b, "Normalized response S(t) = (R\u2212R\u2080)/R\u2080", "time (s)", "|S|")
    b.legend(frameon=False, ncols=2, fontsize=9)
    f.tight_layout(); f.savefig("figures/fig1_synthetic_responses.png", dpi=300); plt.close(f)


def fig2_feature_separation():
    """Per-gate boxplots of the five model features (merged real + synthetic)."""
    X = _merged()
    f, axes = plt.subplots(1, len(base.FEATURES), figsize=(16, 3.6))
    for ax, feat in zip(axes, base.FEATURES):
        order = [X[X.label == g][feat] for g in GASES]
        bp = ax.boxplot(order, tick_labels=GASES, patch_artist=True, widths=0.55,
                        medianprops=dict(color="k", lw=1.2))
        for patch, g in zip(bp["boxes"], GASES):
            patch.set_facecolor(GAS_COL[g]); patch.set_alpha(0.75)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.set_ylim(0, np.nanpercentile(X[feat].to_numpy(), 98) * 1.15)
    f.suptitle("Feature separation across gases (merged real UCI 1081 + synthetic)", fontsize=13, fontweight="bold")
    f.tight_layout(rect=[0, 0, 1, 0.94]); f.savefig("figures/fig2_feature_separation.png", dpi=300); plt.close(f)


def fig3_projection():
    """PCA and LDA projections of the 4-gas feature space (what the model sees)."""
    X = _merged()
    y = X.label.to_numpy()
    # PCA/LDA reject NaN (unlike XGBoost); impute medians for the projection only
    Z = StandardScaler().fit_transform(X[base.FEATURES].fillna(X[base.FEATURES].median()))
    f, (a, b) = fig_pair()
    pca = PCA(n_components=2).fit_transform(Z)
    lda = LinearDiscriminantAnalysis(n_components=2).fit_transform(Z, y)
    for gas in GASES:
        m = y == gas
        a.scatter(pca[m, 0], pca[m, 1], s=14, alpha=0.65, color=GAS_COL[gas], label=gas)
        b.scatter(lda[m, 0], lda[m, 1], s=14, alpha=0.65, color=GAS_COL[gas])
    tidy(a, "PCA projection (unsupervised)", "PC1", "PC2")
    tidy(b, "LDA projection (supervised)", "LD1", "LD2")
    a.legend(frameon=False, fontsize=9, loc="best")
    f.suptitle("4-gas feature space, n=" + str(len(X)), fontsize=13, fontweight="bold")
    f.tight_layout(rect=[0, 0, 1, 0.93]); f.savefig("figures/fig3_projection.png", dpi=300); plt.close(f)


def fig4_confusion_real():
    """Aggregated confusion matrix on real UCI test slices (4-class CV)."""
    xr, _ = _real(); xs, gs = _synth()
    xr["src"] = "real"; xs["src"] = "synth"
    X = pd.concat([xr, xs], ignore_index=True)
    yraw = X.pop("label").to_numpy(); src = X.pop("src").to_numpy()
    groups = np.array([f"real-{i}" for i in range(len(xr))] + gs, dtype=object)
    le = LabelEncoder(); y = le.fit_transform(yraw)
    cls = list(le.classes_)
    clf = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                        eval_metric="mlogloss", random_state=0, n_jobs=-1)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    cm = np.zeros((len(cls), len(cls)))
    for tr, te in cv.split(X, y, groups):
        clf.fit(X.iloc[tr], y[tr]); p = clf.predict(X.iloc[te])
        r = src[te] == "real"
        cm += confusion_matrix(y[te][r], p[r], labels=range(len(cls)))
    row = cm.sum(axis=1, keepdims=True)
    cmn = np.divide(cm, row, out=np.zeros_like(cm), where=row > 0)  # empty row = class absent from real data
    f, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cls)), cls, rotation=30, ha="right")
    ax.set_yticks(range(len(cls)), cls)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Real-data confusion (UCI 1081, grouped CV, acc={cm.trace()/cm.sum():.3f})",
                 fontsize=12, fontweight="bold")
    for i in range(len(cls)):
        for j in range(len(cls)):
            txt = f"{cmn[i,j]:.2f}" if row[i, 0] > 0 else "\u2014"
            ax.text(j, i, txt, ha="center", va="center",
                    color="w" if cmn[i, j] > 0.5 else "k", fontsize=11)
    f.colorbar(im, fraction=0.046, pad=0.03)
    f.tight_layout(); f.savefig("figures/fig4_confusion_real.png", dpi=300); plt.close(f)


def fig5_pipeline_flow():
    """Processing pipeline schematic: raw signal -> features -> classifier -> gas."""
    f, ax = plt.subplots(figsize=(11.5, 2.6)); ax.axis("off")
    steps = [
        ("Raw\nR(t)", "#e8eef7"),
        ("S(t)=\u0394R/R\u2080\nbaseline-corrected", "#e8eef7"),
        ("SavGol\nsmoothing", "#e8eef7"),
        ("Transient features\nT\u2089\u2080, \u03c4\u2090\u1d4f, rates", "#ebf5e1"),
        ("XGBoost / RF\n4 classes", "#fdece3"),
    ]
    n = len(steps)
    for i, (txt, col) in enumerate(steps):
        box = plt.Rectangle((i / n + 0.006, 0.18), 0.19, 0.62, facecolor=col,
                            edgecolor="#444", lw=1.2, transform=ax.transAxes)
        ax.add_patch(box)
        ax.text((i + 0.5) / n, 0.49, txt, ha="center", va="center", fontsize=10.5,
                transform=ax.transAxes)
        if i < n - 1:
            ax.annotate("", xy=((i + 1) / n + 0.0, 0.49), xytext=((i + 1) / n - 0.025, 0.49),
                        arrowprops=dict(arrowstyle="->", color="#333", lw=1.6),
                        transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Single ZnO sensor \u2192 dynamic response \u2192 VOC classification",
                 fontsize=13, fontweight="bold")
    f.tight_layout(); f.savefig("figures/fig5_pipeline_flow.png", dpi=300); plt.close(f)


if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)
    for fn in (fig1_synthetic_responses, fig2_feature_separation, fig3_projection,
               fig4_confusion_real, fig5_pipeline_flow):
        fn()
        print("done", fn.__name__, flush=True)
    print("figures written to figures/")