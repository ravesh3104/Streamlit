"""Train every model, evaluate, explain, and persist all artifacts.

Run:  python -m src.train

Produces:
  data/clean_ai4i2020.csv          cleaned + engineered dataset
  data/synthetic_corrupted_ai4i.csv deliberately corrupted copy (pipeline test)
  artifacts/models.joblib          fitted models + metadata for the dashboard
  artifacts/results.json           all metrics / explanations (JSON-safe)
  artifacts/cleaning_log.json      every cleaning action taken
  artifacts/test_predictions.csv   per-row test-set predictions
  artifacts/figures/*.png          every chart
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from . import cleaning, config as C, modeling, plots


def _lib_versions() -> Dict[str, str]:
    """Record the versions used for training.

    Scikit-learn warns when unpickling a model created by a different version.
    Recording ours lets the app explain that warning in plain language instead
    of letting it look like a failure.
    """
    import sklearn

    return {
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def json_safe(obj: Any):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def main() -> Dict[str, Any]:
    t0 = time.time()
    print("=" * 74)
    print("STEP 1/5  Import + clean")
    print("=" * 74)
    clean_out = cleaning.run()
    df = clean_out["data"]
    summary = clean_out["summary"]

    # persist the corrupted copy so the cleaning test is reproducible/auditable
    raw = cleaning.read_raw()
    cleaning.corrupt_copy(raw).to_csv(C.CORRUPTED_CSV, index=False)

    print(f"  rows={len(df):,}  cols={df.shape[1]}  "
          f"failures={int(df[C.TARGET].sum())}  removed={summary['rows_removed']}")
    print(f"  corruption test passed: {summary['corruption_test']['pipeline_recovered']}")

    print("=" * 74)
    print("STEP 2/5  Build design matrix")
    print("=" * 74)
    print("STEP 3/5  Train + evaluate (LR / Decision Tree / Random Forest)")
    print("=" * 74)
    results = modeling.run_experiment(df)
    print(f"  features used: {results['n_features']}")
    print(f"  excluded (leakage/labels): {results['excluded_columns']}")
    for k, v in results["classes"].items():
        m = v["test_at_tuned"]
        print(f"  {v['label']:<20} thr={v['tuned_threshold']:.3f} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"PR-AUC={m['pr_auc']:.3f}")
    print(f"  best model -> {results['best_model']}")

    print("=" * 74)
    print("STEP 4/5  Explanations + figures")
    print("=" * 74)
    fig_paths = plots.make_all(results, df)
    print(f"  {len([p for p in fig_paths.values() if p])} figures written")

    print("=" * 74)
    print("STEP 5/5  Persist artifacts")
    print("=" * 74)
    X_tr, X_te, y_tr, y_te = results["_random_split"]
    bundle = {
        "models": results["_fitted"],
        "best_model": results["best_model"],
        "feature_names": results["feature_names"],
        "tuned_thresholds": {
            k: v["tuned_threshold"] for k, v in results["classes"].items()
        },
        "metrics_summary": {
            k: {"label": v["label"], "test_at_tuned": v["test_at_tuned"],
                "test_at_0.5": v["test_at_0.5"]}
            for k, v in results["classes"].items()
        },
        "training_reference": {
            "sensor_medians": {c: float(df[c].median()) for c in C.SENSOR_COLS},
            "sensor_ranges": {
                c: {"min": float(df[c].min()), "max": float(df[c].max()),
                    "q05": float(df[c].quantile(0.05)),
                    "q95": float(df[c].quantile(0.95))}
                for c in C.SENSOR_COLS
            },
            "type_share": df[C.CATEGORICAL_COL].value_counts(normalize=True).round(4).to_dict(),
            "n_rows": int(len(df)),
            "failure_rate": float(df[C.TARGET].mean()),
        },
        "disclaimer": C.DISCLAIMER,
        "library_versions": _lib_versions(),
    }
    joblib.dump(bundle, C.MODELS_PATH)

    # per-row test predictions for the dashboard's batch tab
    best = results["_fitted"][results["best_model"]]
    thr = results["classes"][results["best_model"]]["tuned_threshold"]
    p = best.predict_proba(X_te)[:, 1]
    preds = X_te.copy()
    preds["actual_failure"] = y_te.values
    preds["predicted_probability"] = p
    preds["predicted_failure"] = (p >= thr).astype(int)
    preds.to_csv(C.TEST_PREDICTIONS_PATH, index=False)

    out = json_safe(
        {
            "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "runtime_seconds": round(time.time() - t0, 1),
            "dataset": summary["audit"],
            "cleaning": summary,
            "features": {
                "used": results["feature_names"],
                "excluded": results["excluded_columns"],
                "engineered": C.ENGINEERED_COLS if C.ADD_ENGINEERED_FEATURES else [],
                "ids_used_as_features": C.USE_ID_COLUMNS_AS_FEATURES,
            },
            "n_rows": results["n_rows"],
            "n_failures": results["n_failures"],
            "failure_rate": results["failure_rate"],
            "splits": results["splits"],
            "models": results["classes"],
            "best_model": results["best_model"],
            "ranking_by_test_f1": results["ranking_by_test_f1"],
            "figures": fig_paths,
            "config": {
                "random_state": C.RANDOM_STATE,
                "test_size": C.TEST_SIZE,
                "cv_folds": C.CV_FOLDS,
                "iqr_multiplier": C.IQR_MULTIPLIER,
                "class_weight": "balanced",
                "bootstrap_resamples": C.N_BOOTSTRAP,
            },
        }
    )
    C.RESULTS_PATH.write_text(json.dumps(out, indent=2))
    print(f"  models   -> {C.MODELS_PATH}")
    print(f"  results  -> {C.RESULTS_PATH}")
    print(f"  preds    -> {C.TEST_PREDICTIONS_PATH}")
    print(f"\nDone in {time.time() - t0:.1f}s")
    return out


if __name__ == "__main__":
    main()
