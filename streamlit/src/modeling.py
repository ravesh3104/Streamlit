"""Modelling: design matrices, the three classifiers, threshold tuning,
evaluation, and explanation.

Design decisions encoded here (all user-approved):
  * Three comparable models: Logistic Regression, Decision Tree, Random Forest.
  * Class imbalance handled with class_weight="balanced" (no resampling).
  * Decision threshold tuned by maximising F1 on *training-fold* out-of-fold
    predictions only. The test set is never used to pick the threshold.
  * BOTH a random stratified split and a time-ordered split are evaluated.
  * Explanation is native: standardised LR coefficients, tree impurity
    importances, and permutation importance (model-agnostic).
"""
from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from . import config as C
from . import cleaning

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


# --------------------------------------------------------------------------
# Design matrix
# --------------------------------------------------------------------------
def load_clean() -> pd.DataFrame:
    if not C.CLEAN_CSV.exists():
        cleaning.run()
    return pd.read_csv(C.CLEAN_CSV)


def build_design_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series], List[str]]:
    """Build X, y, and the list of excluded (leaky/non-feature) columns.

    Works in inference mode too: if the target column is absent (e.g. a batch
    file uploaded for scoring), ``y`` is returned as ``None`` instead of raising.
    """
    d = df.copy()

    excluded = list(C.FAILURE_MODE_COLS) + [C.TARGET]
    excluded += [c for c in d.columns if c.endswith("outlier") or c == "any_outlier"]

    if C.USE_ID_COLUMNS_AS_FEATURES:
        # Product ID cannot be used raw (10k unique strings) so keep an ordinal
        # view of its serial number. It is expected to be pure noise.
        d["Product ID serial"] = cleaning.parse_product_id(d[C.ID_COLS[1]])
        d = d.drop(columns=[C.ID_COLS[1]])
    else:
        d = d.drop(columns=[c for c in C.ID_COLS if c in d.columns])
        excluded += C.ID_COLS

    # One-hot the machine type (kept as three flags for interpretability).
    d = pd.get_dummies(d, columns=[C.CATEGORICAL_COL], prefix="Type", drop_first=False)

    feature_cols = [c for c in d.columns if c not in excluded]
    X = d[feature_cols].astype(float)
    y = d[C.TARGET].astype(int) if C.TARGET in d.columns else None
    return X, y, excluded


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def get_models() -> Dict[str, Any]:
    """The three classifiers, all class-weight balanced."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=C.RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=6,
            min_samples_leaf=10,
            random_state=C.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            max_depth=14,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=C.N_JOBS,
            random_state=C.RANDOM_STATE,
        ),
    }


def feature_names_of(model, X: pd.DataFrame) -> List[str]:
    if isinstance(model, Pipeline):
        return list(X.columns)
    return list(X.columns)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics_at(y_true, y_proba, threshold: float) -> Dict[str, Any]:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_predicted_positive": int(tp + fp),
        "n_actual_positive": int(tp + fn),
    }


def best_threshold(y_true, y_proba) -> Tuple[float, float]:
    """Pick the probability cutoff that maximises F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in C.THRESHOLD_GRID:
        f1 = f1_score(y_true, (np.asarray(y_proba) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def oof_probabilities(model, X, y, folds: int = C.CV_FOLDS) -> np.ndarray:
    """Out-of-fold probabilities inside the training set only."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.RANDOM_STATE)
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def bootstrap_ci(y_true, y_proba, threshold, n=C.N_BOOTSTRAP, seed=C.RANDOM_STATE):
    """Percentile bootstrap CIs - with 68 test failures, intervals matter."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    stats: Dict[str, List[float]] = {"f1": [], "precision": [], "recall": [], "pr_auc": []}
    n_obs = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        yt, yp = y_true[idx], y_proba[idx]
        if yt.sum() == 0:
            continue
        pred = (yp >= threshold).astype(int)
        stats["f1"].append(f1_score(yt, pred, zero_division=0))
        stats["precision"].append(precision_score(yt, pred, zero_division=0))
        stats["recall"].append(recall_score(yt, pred, zero_division=0))
        try:
            stats["pr_auc"].append(average_precision_score(yt, yp))
        except ValueError:
            pass
    out = {}
    for k, v in stats.items():
        if v:
            out[k] = {
                "low": float(np.percentile(v, 2.5)),
                "high": float(np.percentile(v, 97.5)),
                "mean": float(np.mean(v)),
            }
    return out


def cv_metrics(model, X, y, folds: int = C.CV_FOLDS) -> Dict[str, Any]:
    """5-fold stratified CV: metric mean/std at 0.5 and at a tuned threshold."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=C.RANDOM_STATE)
    rows_05, rows_tuned, aucs = [], [], []
    for tr, va in skf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        p_tr = m.predict_proba(X.iloc[tr])[:, 1]
        p_va = m.predict_proba(X.iloc[va])[:, 1]
        t, _ = best_threshold(y.iloc[tr], p_tr)
        rows_05.append(metrics_at(y.iloc[va], p_va, 0.5))
        rows_tuned.append(metrics_at(y.iloc[va], p_va, t))
        aucs.append(average_precision_score(y.iloc[va], p_va))

    def agg(rows):
        keys = ["precision", "recall", "f1", "balanced_accuracy", "roc_auc", "pr_auc"]
        return {
            k: {
                "mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows])),
            }
            for k in keys
        }

    return {"at_0.5": agg(rows_05), "at_tuned": agg(rows_tuned), "pr_auc_folds": aucs}


# --------------------------------------------------------------------------
# Explanations
# --------------------------------------------------------------------------
def explain(model, X_train, y_train, X_test, y_test, feature_names) -> Dict[str, Any]:
    """Native, dependency-free explanations."""
    result: Dict[str, Any] = {}

    if isinstance(model, Pipeline):
        coefs = model.named_steps["clf"].coef_[0]
        result["lr_coefficients"] = dict(
            sorted(
                {f: float(c) for f, c in zip(feature_names, coefs)}.items(),
                key=lambda kv: -abs(kv[1]),
            )
        )
    elif isinstance(model, DecisionTreeClassifier):
        result["tree_importance"] = dict(
            sorted(
                {f: float(v) for f, v in zip(feature_names, model.feature_importances_)}.items(),
                key=lambda kv: -kv[1],
            )
        )
    elif isinstance(model, RandomForestClassifier):
        result["tree_importance"] = dict(
            sorted(
                {f: float(v) for f, v in zip(feature_names, model.feature_importances_)}.items(),
                key=lambda kv: -kv[1],
            )
        )

    # Model-agnostic permutation importance on the held-out set (PR-AUC drop).
    pi = permutation_importance(
        model, X_test, y_test, scoring="average_precision",
        n_repeats=10, random_state=C.RANDOM_STATE, n_jobs=C.N_JOBS,
    )
    result["permutation_importance"] = {
        "features": list(feature_names),
        "mean": [float(v) for v in pi.importances_mean],
        "std": [float(v) for v in pi.importances_std],
        "ranked": dict(
            sorted(
                {f: float(v) for f, v in zip(feature_names, pi.importances_mean)}.items(),
                key=lambda kv: -kv[1],
            )
        ),
    }

    # Partial dependence for the top three raw sensors.
    top = [k for k in result["permutation_importance"]["ranked"] if k in C.SENSOR_COLS][:3]
    pdp = {}
    for feat in top:
        try:
            idx = list(feature_names).index(feat)
            grid = np.quantile(X_train[feat], np.linspace(0.02, 0.98, 12))
            pd_res = partial_dependence(
                model, X_train, features=[idx], grid_resolution=12,
                percentiles=(0.02, 0.98), kind="average",
            )
            pdp[feat] = {
                "x": [float(v) for v in pd_res["grid_values"][0]],
                "y": [float(v) for v in pd_res["average"][0]],
            }
        except Exception as exc:  # pragma: no cover - defensive
            pdp[feat] = {"error": str(exc)}
    result["partial_dependence"] = pdp
    return result


# --------------------------------------------------------------------------
# Full experiment
# --------------------------------------------------------------------------
def split_random(X, y):
    return train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )


def split_temporal(df: pd.DataFrame):
    """Sort by UDI (production order) and cut the last 20% as the test set."""
    order = df.sort_values("UDI").reset_index(drop=True)
    X, y, _ = build_design_matrix(order)
    cut = int(len(order) * (1 - C.TEST_SIZE))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def run_experiment(df: pd.DataFrame) -> Dict[str, Any]:
    X, y, excluded = build_design_matrix(df)
    feature_names = list(X.columns)
    results: Dict[str, Any] = {
        "feature_names": feature_names,
        "excluded_columns": excluded,
        "n_rows": int(len(X)),
        "n_features": len(feature_names),
        "n_failures": int(y.sum()),
        "failure_rate": float(y.mean()),
        "splits": {},
        "classes": {},
    }

    # ---------------- random stratified split ----------------
    X_tr, X_te, y_tr, y_te = split_random(X, y)
    results["splits"]["random"] = {
        "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
        "test_failures": int(y_te.sum()),
        "test_failure_rate": float(y_te.mean()),
    }

    models = get_models()
    fitted = {}
    for key, model in models.items():
        m = clone(model)
        m.fit(X_tr, y_tr)
        fitted[key] = m

        oof = oof_probabilities(model, X_tr, y_tr)
        thr, thr_f1 = best_threshold(y_tr, oof)
        p_te = m.predict_proba(X_te)[:, 1]

        results["classes"][key] = {
            "label": MODEL_LABELS[key],
            "cv": cv_metrics(model, X_tr, y_tr),
            "tuned_threshold": thr,
            "tuned_threshold_train_f1": thr_f1,
            "test_at_0.5": metrics_at(y_te, p_te, 0.5),
            "test_at_tuned": metrics_at(y_te, p_te, thr),
            "test_ci_at_tuned": bootstrap_ci(y_te, p_te, thr),
            "explanation": explain(m, X_tr, y_tr, X_te, y_te, feature_names),
        }
    results["_fitted"] = fitted
    results["_random_split"] = (X_tr, X_te, y_tr, y_te)

    # ---------------- temporal split ----------------
    if C.RUN_TEMPORAL_SPLIT:
        Xt_tr, Xt_te, yt_tr, yt_te = split_temporal(df)
        temp_block = {
            "train_rows": int(len(Xt_tr)), "test_rows": int(len(Xt_te)),
            "test_failures": int(yt_te.sum()),
            "test_failure_rate": float(yt_te.mean()) if len(yt_te) else 0.0,
            "classes": {},
        }
        for key, model in models.items():
            m = clone(model)
            m.fit(Xt_tr, yt_tr)
            oof = oof_probabilities(model, Xt_tr, yt_tr)
            thr, _ = best_threshold(yt_tr, oof)
            p = m.predict_proba(Xt_te)[:, 1]
            temp_block["classes"][key] = {
                "label": MODEL_LABELS[key],
                "tuned_threshold": thr,
                "test_at_0.5": metrics_at(yt_te, p, 0.5),
                "test_at_tuned": metrics_at(yt_te, p, thr),
            }
        results["splits"]["temporal"] = temp_block

    # ---------------- pick the winner (tuned F1, random split) -------------
    ranking = sorted(
        results["classes"].items(),
        key=lambda kv: kv[1]["test_at_tuned"]["f1"],
        reverse=True,
    )
    results["best_model"] = ranking[0][0]
    results["ranking_by_test_f1"] = [
        {
            "model": k,
            "f1": v["test_at_tuned"]["f1"],
            "precision": v["test_at_tuned"]["precision"],
            "recall": v["test_at_tuned"]["recall"],
        }
        for k, v in ranking
    ]
    return results


__all__ = [
    "load_clean", "build_design_matrix", "get_models", "metrics_at",
    "best_threshold", "run_experiment", "MODEL_LABELS", "explain",
]
