"""All figures for the dashboard, report and notebook.

Every plot writes a PNG into ``artifacts/figures`` and returns the path, so the
Streamlit app and the notebook can reuse exactly the same images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from . import config as C

# ---- shared visual identity (matches the dashboard's dark industrial theme) --
BG = "#0e1726"
PANEL = "#152238"
TEXT = "#e6edf7"
MUTED = "#8fa3bf"
TEAL = "#2dd4bf"
AMBER = "#fbbf24"
RED = "#ef4444"
BLUE = "#60a5fa"
PURPLE = "#a78bfa"
GRID = "#24344d"

MODEL_COLORS = {
    "logistic_regression": BLUE,
    "decision_tree": AMBER,
    "random_forest": TEAL,
}


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(PANEL)
    ax.figure.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    return ax


def _save(fig, name: str) -> str:
    path = C.FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return str(path)


def _new(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    return fig, ax


# --------------------------------------------------------------------------
# EDA
# --------------------------------------------------------------------------
def fig_class_imbalance(df):
    fig, ax = _new(7, 4.2)
    counts = df[C.TARGET].value_counts().sort_index()
    bars = ax.bar(
        ["Healthy (0)", "Failed (1)"], counts.values,
        color=[TEAL, RED], width=0.55, edgecolor=BG,
    )
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 120, f"{v:,}\n{v/len(df):.2%}",
                ha="center", color=TEXT, fontsize=11, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.22)
    _style(ax, "Class imbalance: only 3.4% of records are failures",
           ylabel="Records")
    return _save(fig, "01_class_imbalance.png")


def fig_sensor_distributions(df):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    fig.patch.set_facecolor(BG)
    for ax, col in zip(axes.ravel(), C.SENSOR_COLS):
        for label, colour in [(0, TEAL), (1, RED)]:
            sub = df[df[C.TARGET] == label][col]
            ax.hist(sub, bins=45, alpha=0.62, label="Healthy" if label == 0 else "Failed",
                    color=colour, density=True)
        _style(ax, col, ylabel="density")
    axes.ravel()[-1].axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TEAL, alpha=0.7),
        plt.Rectangle((0, 0), 1, 1, color=RED, alpha=0.7),
    ]
    axes.ravel()[-1].legend(handles, ["Healthy", "Failed"], loc="center",
                            facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=12)
    fig.suptitle("Sensor distributions split by failure outcome",
                 color=TEXT, fontsize=14, fontweight="bold", y=1.00)
    fig.tight_layout()
    return _save(fig, "02_sensor_distributions.png")


def fig_correlation(df):
    cols = C.SENSOR_COLS + C.ENGINEERED_COLS + [C.TARGET]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()
    fig, ax = _new(9, 7.5)
    cmap = LinearSegmentedColormap.from_list("ind", [BLUE, PANEL, RED])
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, vmin=-1, vmax=1,
                ax=ax, cbar_kws={"label": "Pearson r"},
                annot_kws={"size": 7.5, "color": TEXT}, linewidths=0.5, linecolor=BG)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_color(MUTED)
    cbar.ax.tick_params(colors=MUTED)
    ax.set_title("Correlation matrix - sensors, engineered features, target",
                 color=TEXT, fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    fig.patch.set_facecolor(BG)
    return _save(fig, "03_correlation.png")


def fig_failure_modes(df):
    """Descriptive only - these columns are EXCLUDED from every model."""
    fig, ax = _new(8, 4.2)
    modes = df[C.FAILURE_MODE_COLS].sum().sort_values(ascending=False)
    names = {
        "TWF": "Tool wear failure", "HDF": "Heat dissipation failure",
        "PWF": "Power failure", "OSF": "Overstrain failure", "RNF": "Random failure",
    }
    bars = ax.barh([names[m] for m in modes.index], modes.values,
                   color=[RED, AMBER, PURPLE, BLUE, MUTED], edgecolor=BG)
    for b, v in zip(bars, modes.values):
        ax.text(v + 1.2, b.get_y() + b.get_height() / 2, str(v),
                va="center", color=TEXT, fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    _style(ax, "Failure modes (description only - excluded from model inputs)",
           xlabel="Records flagged")
    return _save(fig, "04_failure_modes.png")


def fig_torque_speed(df):
    fig, ax = _new(8.5, 5.5)
    healthy = df[df[C.TARGET] == 0]
    failed = df[df[C.TARGET] == 1]
    ax.scatter(healthy["Rotational speed [rpm]"], healthy["Torque [Nm]"],
               s=5, color=TEAL, alpha=0.20, label="Healthy", edgecolors="none")
    ax.scatter(failed["Rotational speed [rpm]"], failed["Torque [Nm]"],
               s=32, color=RED, alpha=0.85, label="Failed", edgecolors="white", linewidths=0.4)
    # iso-power curves: explain the hyperbolic boundary
    for kw, ls in [(3.5, "--"), (6.0, ":"), (9.0, "-.")]:
        sp = np.linspace(1100, 2900, 200)
        tq = kw * 1000 / (sp * 2 * np.pi / 60)
        mask = tq < 85
        ax.plot(sp[mask], tq[mask], ls, color=MUTED, lw=1.1, alpha=0.8)
        ax.text(sp[mask][-1], tq[mask][-1] + 1.5, f"{kw} kW", color=MUTED, fontsize=8)
    ax.set_xlim(1050, 2950)
    ax.set_ylim(0, 85)
    _style(ax, "Torque vs rotational speed - failures sit on high-power operating points",
           "Rotational speed [rpm]", "Torque [Nm]")
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=10)
    for t in leg.get_texts():
        t.set_color(TEXT)
    return _save(fig, "05_torque_speed.png")


def fig_wear_vs_target(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    sns.boxplot(data=df, x=C.TARGET, y="Tool wear [min]", ax=ax,
                palette=[TEAL, RED], hue=C.TARGET, legend=False, width=0.5)
    _style(ax, "Tool wear by outcome", "0 = healthy, 1 = failed", "Tool wear [min]")
    ax = axes[1]
    rates = df.groupby(C.CATEGORICAL_COL)[C.TARGET].agg(["mean", "count"]).reindex(["L", "M", "H"])
    bars = ax.bar(rates.index, rates["mean"], color=[TEAL, AMBER, RED], width=0.5, edgecolor=BG)
    for b, (m, c) in zip(bars, zip(rates["mean"], rates["count"])):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.0015, f"{m:.2%}\nn={c:,}",
                ha="center", color=TEXT, fontsize=10, fontweight="bold")
    ax.set_ylim(0, rates["mean"].max() * 1.3)
    _style(ax, "Failure rate by machine type", "Machine type", "Failure rate")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    fig.tight_layout()
    return _save(fig, "06_wear_type.png")


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def fig_confusion_matrices(results: Dict[str, Any], split="random", threshold="tuned"):
    classes = results["classes"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.patch.set_facecolor(BG)
    for ax, (key, val) in zip(axes, classes.items()):
        m = val["test_at_tuned"] if threshold == "tuned" else val["test_at_0.5"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        cmap = LinearSegmentedColormap.from_list("cm", [PANEL, MODEL_COLORS[key]])
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max() * 1.15)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color=TEXT, fontsize=17, fontweight="bold")
        ax.set_xticks([0, 1], ["Pred healthy", "Pred failure"])
        ax.set_yticks([0, 1], ["Actual healthy", "Actual failure"])
        thr = m["threshold"]
        ax.set_title(f"{val['label']}\nthr={thr:.2f} | F1={m['f1']:.3f}",
                     color=TEXT, fontsize=11, fontweight="bold")
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)
    fig.suptitle("Confusion matrices on the held-out test set (tuned threshold)",
                 color=TEXT, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "07_confusion_matrices.png")


def fig_pr_roc(results: Dict[str, Any]):
    X_tr, X_te, y_tr, y_te = results["_random_split"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    fig.patch.set_facecolor(BG)
    for key, model in results["_fitted"].items():
        p = model.predict_proba(X_te)[:, 1]
        colour = MODEL_COLORS[key]
        label = results["classes"][key]["label"]
        prec, rec, _ = precision_recall_curve(y_te, p)
        ap = average_precision_score(y_te, p)
        axes[0].plot(rec, prec, color=colour, lw=2.2, label=f"{label} (AP={ap:.3f})")
        fpr, tpr, _ = roc_curve(y_te, p)
        axes[1].plot(fpr, tpr, color=colour, lw=2.2, label=label)
        thr = results["classes"][key]["tuned_threshold"]
        pr_t = results["classes"][key]["test_at_tuned"]
        axes[0].scatter([pr_t["recall"]], [pr_t["precision"]], s=110, color=colour,
                        edgecolors="white", zorder=5, linewidths=1.4)
    axes[0].axhline(y_te.mean(), ls="--", color=MUTED, lw=1.2)
    axes[0].text(0.02, y_te.mean() + 0.015, f"baseline ({y_te.mean():.3f})",
                 color=MUTED, fontsize=9)
    _style(axes[0], "Precision-Recall (the honest view when data is imbalanced)",
           "Recall", "Precision")
    axes[1].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1.1)
    _style(axes[1], "ROC curve", "False positive rate", "True positive rate")
    for ax in axes:
        leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9, loc="lower left" if ax is axes[0] else "lower right")
        for t in leg.get_texts():
            t.set_color(TEXT)
    fig.suptitle("Discrimination curves - dots mark each model's tuned operating point",
                 color=TEXT, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "08_pr_roc.png")


def fig_threshold_curves(results: Dict[str, Any]):
    X_tr, X_te, y_tr, y_te = results["_random_split"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.patch.set_facecolor(BG)
    for ax, (key, model) in zip(axes, results["_fitted"].items()):
        p = model.predict_proba(X_te)[:, 1]
        ths = np.linspace(0.05, 0.95, 120)
        prec, rec, f1 = [], [], []
        for t in ths:
            pred = (p >= t).astype(int)
            tp = int(((pred == 1) & (y_te == 1)).sum())
            fp = int(((pred == 1) & (y_te == 0)).sum())
            fn = int(((pred == 0) & (y_te == 1)).sum())
            prec.append(tp / (tp + fp) if tp + fp else 1.0)
            rec.append(tp / (tp + fn) if tp + fn else 0.0)
            pr = tp / (tp + fp) if tp + fp else 0.0
            f1.append(2 * pr * rec[-1] / (pr + rec[-1]) if pr + rec[-1] else 0.0)
        ax.plot(ths, prec, color=BLUE, lw=2, label="Precision")
        ax.plot(ths, rec, color=AMBER, lw=2, label="Recall")
        ax.plot(ths, f1, color=TEAL, lw=2.4, label="F1")
        t = results["classes"][key]["tuned_threshold"]
        ax.axvline(t, color=RED, ls="--", lw=1.6)
        ax.text(t + 0.02, 0.06, f"tuned = {t:.2f}", color=RED, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.04)
        _style(ax, results["classes"][key]["label"], "Decision threshold", "Score")
        leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8.5)
        for tx in leg.get_texts():
            tx.set_color(TEXT)
    fig.suptitle("Where to put the alarm threshold (test set shown for interpretation)",
                 color=TEXT, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "09_thresholds.png")


def fig_model_comparison(results: Dict[str, Any]):
    classes = results["classes"]
    keys = list(classes)
    metrics = ["precision", "recall", "f1"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.0))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    x = np.arange(len(metrics))
    w = 0.26
    for i, k in enumerate(keys):
        vals05 = [classes[k]["test_at_0.5"][m] for m in metrics]
        valst = [classes[k]["test_at_tuned"][m] for m in metrics]
        ax.bar(x + (i - 1) * w, valst, w * 0.88, color=MODEL_COLORS[k],
               label=classes[k]["label"], edgecolor=BG)
        ax.scatter(x + (i - 1) * w, vals05, marker="_", s=260, color="white",
                   zorder=5, linewidths=2.2)
    ax.set_xticks(x, [m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    _style(ax, "Test metrics at tuned threshold (white tick = default 0.5)", ylabel="Score")
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TEXT)

    ax = axes[1]
    for i, k in enumerate(keys):
        cv = classes[k]["cv"]["at_tuned"]
        pos = np.arange(len(metrics)) + (i - 1) * w
        means = [cv[m]["mean"] for m in metrics]
        errs = [cv[m]["std"] for m in metrics]
        ax.bar(pos, means, w * 0.88, yerr=errs, capsize=4, color=MODEL_COLORS[k],
               label=classes[k]["label"], edgecolor=BG,
               error_kw={"ecolor": TEXT, "lw": 1.3})
    ax.set_xticks(np.arange(len(metrics)), [m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    _style(ax, "5-fold cross-validation inside train (mean ± std)", ylabel="Score")
    fig.suptitle("Model comparison", color=TEXT, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "10_model_comparison.png")


def fig_importance(results: Dict[str, Any], model_key: str = "random_forest"):
    exp = results["classes"][model_key]["explanation"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4))
    fig.patch.set_facecolor(BG)

    ranked = list(exp["permutation_importance"]["ranked"].items())[:14]
    names = [r[0] for r in ranked][::-1]
    vals = [r[1] for r in ranked][::-1]
    std_map = dict(zip(exp["permutation_importance"]["features"],
                       exp["permutation_importance"]["std"]))
    errs = [std_map.get(n, 0) for n in names]
    ax = axes[0]
    colours = [TEAL if v > 0 else RED for v in vals]
    ax.barh(names, vals, xerr=errs, color=colours, edgecolor=BG,
            error_kw={"ecolor": MUTED, "lw": 1})
    ax.axvline(0, color=MUTED, lw=1)
    _style(ax, f"Permutation importance ({results['classes'][model_key]['label']})\n"
               "drop in PR-AUC when the feature is shuffled",
           "Mean decrease in average precision", None)
    ax.tick_params(labelsize=8.5)

    ax = axes[1]
    tree_imp = exp.get("tree_importance") or {}
    ranked2 = list(tree_imp.items())[:14]
    if ranked2:
        n2 = [r[0] for r in ranked2][::-1]
        v2 = [r[1] for r in ranked2][::-1]
        ax.barh(n2, v2, color=PURPLE, edgecolor=BG)
        _style(ax, "Impurity-based (Gini) importance\n"
                   "caution: biased toward high-cardinality features",
               "Relative importance", None)
        ax.tick_params(labelsize=8.5)
    fig.suptitle("Which inputs actually drive the prediction?",
                 color=TEXT, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "11_feature_importance.png")


def fig_lr_coefficients(results: Dict[str, Any]):
    coefs = results["classes"]["logistic_regression"]["explanation"]["lr_coefficients"]
    items = list(coefs.items())[:16][::-1]
    fig, ax = _new(9, 6.4)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colours = [RED if v > 0 else TEAL for v in vals]
    ax.barh(names, vals, color=colours, edgecolor=BG)
    ax.axvline(0, color=MUTED, lw=1)
    ax.tick_params(labelsize=8.5)
    _style(ax, "Logistic Regression coefficients (standardised inputs)\n"
               "red = pushes toward failure, teal = pushes toward healthy",
           "Coefficient (log-odds per standard deviation)", None)
    return _save(fig, "12_lr_coefficients.png")


def fig_partial_dependence(results: Dict[str, Any], model_key="random_forest"):
    pdp = results["classes"][model_key]["explanation"]["partial_dependence"]
    pdp = {k: v for k, v in pdp.items() if "x" in v}
    if not pdp:
        return None
    fig, axes = plt.subplots(1, len(pdp), figsize=(5.2 * len(pdp), 4.3), squeeze=False)
    fig.patch.set_facecolor(BG)
    for ax, (feat, vals) in zip(axes[0], pdp.items()):
        ax.plot(vals["x"], vals["y"], color=TEAL, lw=2.4, marker="o", ms=4)
        ax.fill_between(vals["x"], 0, vals["y"], color=TEAL, alpha=0.13)
        _style(ax, feat, feat, "Predicted failure probability")
        ax.set_ylim(0, min(1, max(vals["y"]) * 1.35 + 0.02))
    fig.suptitle("Partial dependence - how risk moves as one sensor changes",
                 color=TEXT, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "13_partial_dependence.png")


def fig_split_comparison(results: Dict[str, Any]):
    if "temporal" not in results["splits"]:
        return None
    keys = list(results["classes"])
    fig, ax = _new(8.5, 4.6)
    x = np.arange(len(keys))
    rand = [results["classes"][k]["test_at_tuned"]["f1"] for k in keys]
    temp = [results["splits"]["temporal"]["classes"][k]["test_at_tuned"]["f1"] for k in keys]
    ax.bar(x - 0.19, rand, 0.36, color=TEAL, label="Random stratified split", edgecolor=BG)
    ax.bar(x + 0.19, temp, 0.36, color=AMBER, label="Time-ordered split", edgecolor=BG)
    for xi, (a, b) in enumerate(zip(rand, temp)):
        ax.text(xi - 0.19, a + 0.02, f"{a:.2f}", ha="center", color=TEXT, fontsize=9)
        ax.text(xi + 0.19, b + 0.02, f"{b:.2f}", ha="center", color=TEXT, fontsize=9)
    ax.set_xticks(x, [results["classes"][k]["label"] for k in keys])
    ax.set_ylim(0, 1.05)
    _style(ax, "Does the random split flatter the models? (F1 at tuned threshold)",
           None, "F1")
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9)
    for t in leg.get_texts():
        t.set_color(TEXT)
    return _save(fig, "14_split_comparison.png")


def make_all(results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    out["imbalance"] = fig_class_imbalance(df)
    out["distributions"] = fig_sensor_distributions(df)
    out["correlation"] = fig_correlation(df)
    out["failure_modes"] = fig_failure_modes(df)
    out["torque_speed"] = fig_torque_speed(df)
    out["wear_type"] = fig_wear_vs_target(df)
    out["confusion"] = fig_confusion_matrices(results)
    out["pr_roc"] = fig_pr_roc(results)
    out["thresholds"] = fig_threshold_curves(results)
    out["comparison"] = fig_model_comparison(results)
    out["importance"] = fig_importance(results, results["best_model"])
    out["lr_coef"] = fig_lr_coefficients(results)
    out["pdp"] = fig_partial_dependence(results, results["best_model"]) or ""
    out["split"] = fig_split_comparison(results) or ""
    return out
