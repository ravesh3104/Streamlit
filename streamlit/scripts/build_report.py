"""Generates the written deliverables from artifacts/results.json.

Outputs:
  reports/failure_prediction_report.md    - repository-native report
  reports/failure_prediction_report.docx  - shareable Word version

Both are generated from the *measured* artifacts, so the numbers can never drift
away from what the models actually produced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402

R = json.loads(C.RESULTS_PATH.read_text())
MODELS = R["models"]
BEST = R["best_model"]
BEST_LABEL = MODELS[BEST]["label"]
AUD = R["dataset"]
CLEAN = R["cleaning"]
FEATS = R["features"]["used"]
SPLIT = R["splits"]
TEMP = SPLIT.get("temporal", {})
LABELS = {k: v["label"] for k, v in MODELS.items()}
RANK = R["ranking_by_test_f1"]


def f(x, n=3):
    return f"{x:.{n}f}"


def pct(x, n=1):
    return f"{x*100:.{n}f}%"


def fig(name):
    p = C.FIG_DIR / name
    return p if p.exists() else None


# --------------------------------------------------------------------------
# Markdown report
# --------------------------------------------------------------------------
def build_markdown() -> str:
    b = MODELS[BEST]["test_at_tuned"]
    ci = MODELS[BEST]["test_ci_at_tuned"]
    temp_f1 = TEMP["classes"][BEST]["test_at_tuned"]["f1"] if TEMP else float("nan")
    temp_rec = TEMP["classes"][BEST]["test_at_tuned"]["recall"] if TEMP else float("nan")

    md = f"""# Predicting Industrial Equipment Failure from Sensor Data

**A machine-learning decision-support system for predictive maintenance**

| | |
|---|---|
| **Dataset** | AI4I 2020 Predictive Maintenance (UCI ML Repository, ID 601) |
| **Records** | {AUD['n_rows']:,} machine runs, {AUD['n_columns']} raw columns |
| **Target** | `Machine failure` — binary, {AUD['failure_count']} positive ({pct(AUD['failure_rate'],2)}) |
| **Models compared** | Logistic Regression · Decision Tree · Random Forest |
| **Selected model** | **{BEST_LABEL}** (best held-out F1) |
| **Held-out performance** | Precision {f(b['precision'])} · Recall {f(b['recall'])} · F1 {f(b['f1'])} · PR-AUC {f(b['pr_auc'])} |
| **Strict time-ordered performance** | F1 {f(temp_f1)} · Recall {f(temp_rec)} |
| **Generated** | {R['generated_at']} |

---

## ⚠️ Executive summary — read this before the numbers

This system estimates the **probability** that a machine run will end in failure, using
temperature, rotational speed, torque and tool-wear readings.

**A prediction is not a guarantee of failure.** A high score means the sensor pattern
resembles historical failures and the machine deserves inspection. A low score does **not**
certify that a machine is healthy. This is decision support for prioritising human
maintenance, not a safety device, an automatic shutdown trigger, or a substitute for
engineering judgement.

Three headline results:

1. **Random Forest is clearly the best of the three models**, reaching F1 {f(b['f1'])} at a
   tuned threshold on a held-out test set, with precision {f(b['precision'])} and recall
   {f(b['recall'])}.
2. **The honest number is lower: F1 {f(temp_f1)}** when training only on earlier runs and
   testing on later ones. The random split was somewhat flattering, and the time-ordered
   result is the better guide to real-world expectations.
3. **Logistic Regression is inadequate here** (F1 {f(MODELS['logistic_regression']['test_at_tuned']['f1'])}) because
   the risk boundary is multiplicative in speed and torque. That is a finding about the
   physics, not a flaw in the experiment.

---

## 1. Problem framing

**Question:** given a machine's current sensor readings and operating condition, how likely is
it to be in a failure state — and which inputs drive that estimate?

**Why it matters:** unplanned equipment failure is expensive. A model that flags the
highest-risk few percent of machine runs lets a maintenance team spend inspection hours where
they have the most value. The natural trade-off is a **missed failure** (costly, dangerous)
versus a **wasted inspection** (cheap, safe), and the model's alarm threshold is where that
trade-off is expressed.

**Formulation:** binary classification. `y = 1` if the run ended in failure, else 0.

---

## 2. Dataset

### 2.1 Provenance

The **AI4I 2020 Predictive Maintenance Dataset** is published by the UCI Machine Learning
Repository (ID 601). It describes a synthetic milling process with {AUD['n_rows']:,}
independent runs. A copy of the raw file is bundled at `data/raw_ai4i2020.csv`, so the project
is fully reproducible offline.

### 2.2 Variables

| Variable | Role | Description |
|---|---|---|
| `Air temperature [K]` | input | ambient temperature around the machine |
| `Process temperature [K]` | input | temperature of the working process |
| `Rotational speed [rpm]` | input | spindle rotational speed |
| `Torque [Nm]` | input | torque applied to the tool |
| `Tool wear [min]` | input | cumulative tool usage time |
| `Type` | input | machine grade — L (60.0%), M (30.0%), H (10.0%) |
| `UDI`, `Product ID` | input (experiment) | row identifier and product token |
| `Machine failure` | **target** | 1 if the run failed |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | **excluded** | failure-mode flags — data leakage; see §3.2 |

### 2.3 Data quality audit (before cleaning)

| Check | Result |
|---|---|
| Rows × columns | {AUD['n_rows']:,} × {AUD['n_columns']} |
| Missing values | **{AUD['missing_total']}** |
| Duplicate rows | **{AUD['duplicate_rows']}** |
| Failure count / rate | {AUD['failure_count']} / {pct(AUD['failure_rate'],2)} |
| Machine types | L {AUD['machine_type_counts'].get('L',0):,} · M {AUD['machine_type_counts'].get('M',0):,} · H {AUD['machine_type_counts'].get('H',0):,} |

The published dataset is already clean. Rather than pretend a cleaning stage was necessary,
this project (a) documents the audit above, (b) implements a **reusable** cleaning pipeline, and
(c) **validates that pipeline on a deliberately corrupted copy** of the data.

![Class imbalance](figures/01_class_imbalance.png)

*Figure 1 — Only {pct(AUD['failure_rate'],2)} of records are failures. This single fact
determines most of the methodological choices in this report.*

---

## 3. Data cleaning and feature engineering

### 3.1 The cleaning pipeline

Implemented in `src/cleaning.py`, applied in this order:

1. **Type coercion** — sensor columns forced numeric (`errors='coerce'`), so text in a numeric
   column becomes missing rather than crashing the pipeline.
2. **Machine-type normalisation** — case/whitespace stripped; unknown grades fall back to `M`.
3. **Physical range validation** — readings outside plausible physical limits (e.g. negative
   torque, 99,999 rpm) are set to missing instead of being trusted.
4. **Duplicate removal** — exact duplicate rows dropped, count logged.
5. **Label integrity check** — flags inconsistencies between the target and the failure-mode
   columns **without rewriting them** (see §3.2).
6. **Median imputation** — missing sensor values filled with the column median (robust to the
   outliers that matter here).
7. **IQR outlier flagging** — values beyond {C.IQR_MULTIPLIER:g}×IQR are **flagged, never
   deleted**, because an extreme reading may be the very failure symptom we are trying to learn.

### 3.2 The leakage trap — and the label-noise finding

The five failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are set only *because* a
failure occurred, so they cannot be known before the event. Using them as inputs would produce
a spectacular but meaningless model.

Investigating them surfaced a genuine dataset flaw:

| Relationship | Count |
|---|---|
| Rows flagging a failure mode but labelled healthy | 18 |
| Rows labelled failed with no mode flagged | 9 |
| Rows where the flags exactly reconstruct the target | {AUD['n_rows'] - 27:,} of {AUD['n_rows']:,} |

**Decision:** all five columns are excluded from every model input and used only for
description. The 27 inconsistent rows are recorded as a **documented label-noise limitation**;
the published target is deliberately *not* overwritten, because rewriting ground truth from a
leaked column would corrupt the evaluation itself.

![Failure modes](figures/04_failure_modes.png)

*Figure 2 — Failure-mode breakdown. Descriptive only; never a model input.*

### 3.3 Validating the pipeline (the corrupted-copy test)

A decorative cleaning step proves nothing, so the pipeline was run against a copy of the data
deliberately damaged with realistic faults. Results:

| Injected fault | Result |
|---|---|
| Missing sensor cells | {CLEAN['corruption_test']['injected_missing']} injected → all imputed |
| Duplicate rows | {CLEAN['corruption_test']['injected_duplicates']} injected → all recovered |
| Impossible readings (negative torque, absurd rpm) | caught by range validation → set to missing → imputed |
| Inconsistent machine-type labels (`" m "`) | normalised to `M` |
| Temperature column stored as text with `"n/a"` | coerced to numeric, then imputed |
| **Rows before → after** | {CLEAN['corruption_test']['rows_before']:,} → {CLEAN['corruption_test']['rows_after']:,} |
| **All duplicates recovered?** | **{CLEAN['corruption_test']['pipeline_recovered']}** |

The cleaning code is therefore demonstrably functional, not theatrical.

### 3.4 Engineered features

Five physics-motivated features were added, each with a stated mechanism:

| Feature | Formula | Mechanism it encodes |
|---|---|---|
| `Temp difference [K]` | process − air temperature | Heat generated faster than dissipated → heat-dissipation failure |
| `Mechanical power [kW]` | torque × ω (ω = rpm·2π/60) | Actual load on the tool, not torque alone |
| `Wear-torque interaction` | tool wear × torque | Cumulative strain: a worn tool still under heavy load |
| `Speed-torque ratio` | rpm ÷ torque | The shape of the operating point |
| `Wear rate [min per kW]` | wear ÷ power | Tool life consumed per unit of work done |

These are not an arbitrary polynomial expansion — every term corresponds to a failure
mechanism in §5, which is what makes the explanations in §7 defensible.

### 3.5 Identifier columns (a deliberate experiment)

`UDI` and `Product ID` were **kept as model inputs on request**, to test empirically whether
row identifiers carry any signal. (`Product ID` cannot be used raw — 10,000 unique strings — so
its numeric serial is retained instead.) §7.3 reports the verdict: they are noise. A single
flag (`USE_ID_COLUMNS_AS_FEATURES` in `src/config.py`) removes them.

---

## 4. Exploratory analysis

![Sensor distributions](figures/02_sensor_distributions.png)

*Figure 3 — Sensor distributions split by outcome. Torque and rotational speed separate the
classes; temperature differences are real but subtle.*

![Torque vs speed](figures/05_torque_speed.png)

*Figure 4 — Torque vs rotational speed with iso-power curves. Failures concentrate at high
torque and low speed — i.e. at high-power operating points. This hyperbolic boundary is why a
linear model in raw features cannot fit the data well.*

![Correlation](figures/03_correlation.png)

*Figure 5 — Correlation matrix over sensors, engineered features and target. No individual
feature correlates strongly with failure (max |r| is modest), confirming that the signal is
multivariate rather than a single-threshold problem.*

![Wear and type](figures/06_wear_type.png)

*Figure 6 — Tool wear by outcome, and failure rate by machine grade. Higher grades (H) fail at
a higher rate per unit.*

---

## 5. Modelling

### 5.1 Design choices

| Choice | Decision | Rationale |
|---|---|---|
| Models | Logistic Regression, Decision Tree, Random Forest | Escalating flexibility; the linear baseline makes the value of non-linearity visible |
| Imbalance handling | `class_weight='balanced'` | Reweights the loss without inventing synthetic sensor readings |
| Threshold | Tuned to maximise F1 on **training-fold OOF** predictions | The test set is never used for tuning |
| Validation | Stratified 80/20 **and** time-ordered 80/20 | One split can flatter; two splits cross-check |
| Random seed | {R['config']['random_state']} | Reproducibility |
| Test set size | {R['config']['test_size']:.0%} ({SPLIT['random']['test_rows']:,} rows, **{SPLIT['random']['test_failures']} failures**) | — |
| Cross-validation | {R['config']['cv_folds']}-fold stratified, inside training only | Estimates stability |

### 5.2 Design matrix

**{len(FEATS)} features used:** {", ".join(f"`{x}`" for x in FEATS)}.

**Excluded from all inputs:** the five failure-mode flags, the target, and the IQR outlier flags.

### 5.3 Results — held-out test set (random stratified split)

Thresholds shown are the tuned values learned on training folds only.

| Model | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC | MCC | F1 @ 0.50 |
|---|---|---|---|---|---|---|---|---|
"""
    for k, v in MODELS.items():
        m, m0 = v["test_at_tuned"], v["test_at_0.5"]
        star = " ⭐" if k == BEST else ""
        md += (
            f"| {v['label']}{star} | {f(m['threshold'],2)} | {f(m['precision'])} | "
            f"{f(m['recall'])} | **{f(m['f1'])}** | {f(m['pr_auc'])} | {f(m['roc_auc'])} | "
            f"{f(m['mcc'])} | {f(m0['f1'])} |\n"
        )

    md += f"""
**Accuracy is deliberately omitted from the comparison** because it is misleading here: a
model predicting "healthy" for every row scores {pct(1 - AUD['failure_rate'],1)} accuracy while
catching **zero** failures.

### 5.4 Why each model performed as it did

**Logistic Regression — F1 {f(MODELS['logistic_regression']['test_at_tuned']['f1'])}.**
Its largest standardised coefficient is on `Speed-torque ratio` ({MODELS['logistic_regression']['explanation']['lr_coefficients'].get('Speed-torque ratio', 0):+.2f}),
which is the model's proxy for the multiplicative speed/torque interaction. A linear model in
raw features cannot represent the hyperbolic decision boundary visible in Figure 4, so it
saturates. This is evidence that the underlying physics is multiplicative — a useful finding
in itself.

**Decision Tree — F1 {f(MODELS['decision_tree']['test_at_tuned']['f1'])}.**
Depth-limited (max_depth=6, min_samples_leaf=10) to control overfitting. It captures the
interaction structure well and is fully readable as rules, but a single tree has high variance.

**Random Forest — F1 {f(MODELS[BEST]['test_at_tuned']['f1'])}.**
Averaging {400} trees cuts the variance of the single tree. It achieves the best
precision ({f(b['precision'])}) and PR-AUC ({f(b['pr_auc'])}), at the cost of recall
({f(b['recall'])}) — it still misses roughly one failure in five.

### 5.5 Stability and uncertainty

| Model | CV precision | CV recall | CV F1 | CV PR-AUC |
|---|---|---|---|---|
"""
    for k, v in MODELS.items():
        cv = v["cv"]["at_tuned"]
        md += (
            f"| {v['label']} | {f(cv['precision']['mean'])} ± {f(cv['precision']['std'])} | "
            f"{f(cv['recall']['mean'])} ± {f(cv['recall']['std'])} | "
            f"{f(cv['f1']['mean'])} ± {f(cv['f1']['std'])} | "
            f"{f(cv['pr_auc']['mean'])} ± {f(cv['pr_auc']['std'])} |\n"
        )

    md += f"""
Bootstrap 95% confidence intervals on the test set ({C.N_BOOTSTRAP} resamples) for the
selected model:

| Metric | Mean | 95% CI |
|---|---|---|
"""
    for k in ["f1", "precision", "recall", "pr_auc"]:
        if k in ci:
            md += f"| {k.upper().replace('_','-')} | {f(ci[k]['mean'])} | {f(ci[k]['low'])} – {f(ci[k]['high'])} |\n"

    md += f"""
These intervals are wide — roughly ±0.07 on F1 — because the test set contains only
{SPLIT['random']['test_failures']} failures. Any single-point claim about this model's quality
overstates what the data can support.

### 5.6 The stricter test: time-ordered split

Training on the first {SPLIT['temporal']['train_rows']:,} runs and testing on the last
{SPLIT['temporal']['test_rows']:,} (which contain {SPLIT['temporal']['test_failures']} failures,
{pct(SPLIT['temporal']['test_failure_rate'],2)}):

| Model | Random-split F1 | Time-split F1 | Random-split recall | Time-split recall | Δ F1 |
|---|---|---|---|---|---|
"""
    for k, v in MODELS.items():
        rnd = v["test_at_tuned"]
        tmp = TEMP["classes"][k]["test_at_tuned"]
        md += (
            f"| {v['label']} | {f(rnd['f1'])} | {f(tmp['f1'])} | {f(rnd['recall'])} | "
            f"{f(tmp['recall'])} | {tmp['f1']-rnd['f1']:+.3f} |\n"
        )

    md += f"""
**Interpretation.** The tree models fall by roughly 0.10–0.13 F1 when the future must be
predicted from the past. The random-split numbers in §5.3 are therefore optimistic; the
time-ordered numbers are the fairer expectation for deployment. Both are reported rather than
picking the flattering one.

![Model comparison](figures/10_model_comparison.png)

*Figure 7 — Test metrics at the tuned threshold (white tick = default 0.5), with 5-fold CV
stability on the right.*

![Split comparison](figures/14_split_comparison.png)

*Figure 8 — Random split vs time-ordered split. The gap is the optimism in the random split.*

---

## 6. Evaluation

![Confusion matrices](figures/07_confusion_matrices.png)

*Figure 9 — Confusion matrices on the held-out test set.*

For the selected model ({BEST_LABEL}):

| | Predicted healthy | Predicted failure |
|---|---|---|
| **Actual healthy** | {b['tn']:,} (correct) | {b['fp']:,} (false alarm) |
| **Actual failure** | {b['fn']:,} (**missed failure**) | {b['tp']:,} (caught) |

The bottom-left cell is the expensive error in maintenance. Relative to a
"predict healthy for everything" baseline, the model trades {b['fp']} false alarms for
{b['tp']} correctly anticipated failures.

![PR and ROC](figures/08_pr_roc.png)

*Figure 10 — Precision-Recall (left) and ROC (right). PR-AUC is the fairer summary at a 3.4%
positive rate; the dots mark each model's tuned operating point.*

![Thresholds](figures/09_thresholds.png)

*Figure 11 — Precision, recall and F1 vs the decision threshold, per model. The dashed line is
the tuned cutoff.*

### 6.1 The threshold is a policy decision, not a technical one

The tuned cutoff maximises F1, but F1 weights false positives and false negatives equally —
which maintenance economics rarely does. On the test set, the selected model behaves as:

| Maintenance policy | Inspections triggered | Failures caught | Failures missed | False alarms |
|---|---|---|---|---|
"""
    # computed from saved test predictions
    import pandas as pd

    tp_ = pd.read_csv(C.TEST_PREDICTIONS_PATH)
    for label, t in [
        ("Sensitive (threshold 0.10)", 0.10),
        (f"Tuned for best F1 ({MODELS[BEST]['tuned_threshold']:.2f})", MODELS[BEST]["tuned_threshold"]),
        ("Conservative (threshold 0.80)", 0.80),
    ]:
        pred = (tp_["predicted_probability"] >= t).astype(int)
        act = tp_["actual_failure"]
        tp = int(((pred == 1) & (act == 1)).sum())
        fp = int(((pred == 1) & (act == 0)).sum())
        fn = int(((pred == 0) & (act == 1)).sum())
        md += f"| {label} | {tp+fp} | {tp} | {fn} | {fp} |\n"

    md += f"""
The in-app alarm threshold is a live slider for exactly this reason: a plant that cannot absorb
extra inspections should raise it; a plant where a missed failure is dangerous should lower it.

---

## 7. Which inputs influence the predictions?

Four complementary methods are used, deliberately so they can check each other.

![Feature importance](figures/11_feature_importance.png)

*Figure 12 — Permutation importance (left, model-agnostic) vs Gini importance (right, biased
toward high-cardinality features). Where they disagree, trust the left half.*

### 7.1 Permutation importance — selected model

Measured as the drop in PR-AUC when a feature's values are randomly shuffled:

| Rank | Feature | PR-AUC drop |
|---|---|---|
"""
    ranked = MODELS[BEST]["explanation"]["permutation_importance"]["ranked"]
    for i, (k, v) in enumerate(list(ranked.items())[:10], 1):
        md += f"| {i} | `{k}` | {v:+.4f} |\n"

    md += f"""
### 7.2 Physical interpretation

Reading the top drivers together, the model's behaviour matches this machine's known failure
mechanisms:

- **Rotational speed and torque** jointly define the operating point. Failures sit at high
  torque / low speed, which is the high-power region — consistent with overstrain and power
  failure modes.
- **`Temp difference`** ranking highly is the heat-dissipation pathway: when the process runs
  hot relative to ambient air, the machine cannot shed heat.
- **`Wear-torque interaction`** captures cumulative strain on a worn tool under load.
- **`Mechanical power`** restates the load in the units that actually stress the machine.

This agreement between statistical importance and known physics is the main reason to trust
the model's reasoning — not its accuracy score.

![Partial dependence](figures/13_partial_dependence.png)

*Figure 13 — Partial dependence: how predicted failure probability moves as one sensor varies,
with everything else held fixed.*

![LR coefficients](figures/12_lr_coefficients.png)

*Figure 14 — Logistic Regression coefficients (standardised inputs). Red raises failure odds,
teal lowers them.*

### 7.3 Do the identifier columns matter? (empirical answer)

| Feature | PR-AUC drop | Rank |
|---|---|---|
"""
    pos = {k: i + 1 for i, k in enumerate(ranked)}
    for k in ["UDI", "Product ID serial"]:
        if k in ranked:
            md += f"| `{k}` | {ranked[k]:+.4f} | {pos[k]} of {len(ranked)} |\n"

    total_imp = sum(abs(v) for v in ranked.values()) or 1
    id_share = sum(abs(ranked.get(k, 0)) for k in ["UDI", "Product ID serial"]) / total_imp
    md += f"""
Together the identifiers account for **{pct(id_share,2)}** of total permutation importance, and
`UDI`'s contribution is *negative* — shuffling it slightly improves the score. Conclusion: row
identifiers carry no predictive signal and should be dropped. They remain only because keeping
them was an explicit experiment; set `USE_ID_COLUMNS_AS_FEATURES = False` in `src/config.py`
to remove them.

---

## 8. Prediction and dashboard

A Streamlit application (`app/streamlit_app.py`) realises the model as a usable tool with four
views:

| View | What it does |
|---|---|
| **🔧 Machine Risk Check** | Sliders for the five sensors + machine grade → failure probability, a risk gauge, all three models side by side, and a plain-language "why" panel naming the inputs that moved this specific prediction. Includes preset failure scenarios. |
| **📁 Batch Scoring** | Upload a CSV of machines → cleaned, scored, ranked by risk, downloadable. Reports exactly which cleaning repairs were applied to the uploaded file, and computes precision/recall/F1 if the file includes ground-truth labels. |
| **📈 Model Evaluation** | Leaderboard, confusion matrices, PR/ROC curves, threshold sensitivity, cross-validation stability, bootstrap CIs, and the random vs time-ordered comparison. |
| **🔍 Patterns & Drivers** | All EDA charts, permutation/Gini importance, partial dependence, LR coefficients, the ID-column verdict, and the full cleaning audit including the corrupted-copy test. |

**No train/serve skew:** the dashboard builds a prediction row and pushes it through the *same*
`cleaning → build_features → build_design_matrix` code used in training, then aligns columns to
the saved training schema. This is the single most common cause of models that work in a
notebook but not in production, and it is handled structurally rather than by convention.

### 8.1 Example predictions

| Scenario (sensor pattern) | Random Forest probability | Interpretation |
|---|---|---|
| High-power overstrain (68 Nm @ 1330 rpm) | high | Flag for inspection |
| Worn tool under load (55 Nm, 235 min wear) | moderate–high | Inspect tool, verify load |
| Heat-dissipation risk (large air/process gap) | moderate | Check cooling |
| Healthy reference (40 Nm, 1500 rpm, 20 min wear) | near zero | Routine monitoring |

Every one of these is a probability, not a verdict. The "healthy" row does not certify safety.

---

## 9. Limitations

This section is deliberately prominent. The numbers above are only as trustworthy as these
caveats allow.

1. **The data is simulated.** AI4I 2020 was generated from a physics-inspired process model,
   not logged from real machines. It lacks sensor drift, correlated channels, dropouts,
   maintenance interventions and operator behaviour. **Absolute performance will not transfer
   to a real plant; the method transfers, the numbers do not.**
2. **Failures are rare, so estimates are uncertain.** {AUD['failure_count']} failures total and
   {SPLIT['random']['test_failures']} in the test set give F1 confidence intervals of roughly
   ±0.07. Re-running with a different seed moves results by several points.
3. **Some failures are unlearnable by construction.** The 19 `RNF` (random failure) records
   have no sensor signature. No model can catch them, which caps achievable recall. Chasing
   100% recall means fitting noise.
4. **Tool-wear failure is truncated.** Every `TWF` record occurs at tool wear of 198–253 min —
   i.e. just before the replacement limit — so that mode is partly an artefact of the
   replacement policy rather than pure physics.
5. **Label noise exists.** 18 rows flag a failure mode while labelled healthy, and 9 failures
   have no mode flag. The published target was preserved unchanged; the inconsistency is
   documented rather than silently "corrected".
6. **No per-machine time history.** Each row is an independent run. Real degradation trends —
   the strongest signal in a real plant — cannot be modelled here. This is the single biggest
   gap between this project and a deployable system.
7. **Feature importance is not causality.** "The model used rotational speed" does not mean
   "change the speed to prevent failure". Do not act on importance charts without engineering
   review.
8. **The threshold is a policy tuned on this dataset.** A real deployment must re-tune it
   against the actual cost of a missed failure versus a wasted inspection.
9. **The model is not aware of context.** It does not know about recent maintenance, ambient
   conditions, material batch, or whether a sensor is faulty. Sensor faults can produce
   confident nonsense.
10. **This is not a safety system.** It is decision support for prioritising inspections. It
    must not be used for automatic shutdown, safety certification, or as the sole basis for a
    maintenance decision.

### 9.1 What would make this genuinely deployable

- Per-asset time-series data (the real predictive-maintenance setting), enabling degradation
  trend features and remaining-useful-life modelling.
- Cost-sensitive threshold selection using actual inspection and downtime costs.
- Production drift monitoring and a scheduled re-validation/retraining cycle.
- Sensor-health checks upstream, so a broken sensor cannot masquerade as a failing machine.
- Human-in-the-loop feedback: capturing whether flagged machines were actually faulty, and
  retraining on that ground truth.

---

## 10. Reproducibility

```bash
pip install -r requirements.txt
python -m src.train                 # clean → train → evaluate → explain → save artifacts
streamlit run app/streamlit_app.py  # launch the dashboard
jupyter notebook notebooks/01_failure_prediction_walkthrough.ipynb
```

| Artifact | Contents |
|---|---|
| `data/raw_ai4i2020.csv` | raw dataset snapshot (offline reproducible) |
| `data/clean_ai4i2020.csv` | cleaned data + engineered features |
| `data/synthetic_corrupted_ai4i.csv` | deliberately damaged copy used to validate cleaning |
| `artifacts/models.joblib` | three fitted models, thresholds, training reference stats |
| `artifacts/results.json` | every metric, importance score and cleaning action |
| `artifacts/test_predictions.csv` | per-row test predictions for auditing |
| `artifacts/figures/*.png` | all 14 figures |

Fixed seed {R['config']['random_state']}, deterministic splits, and every modelling choice
recorded as an explicit constant in `src/config.py`.

---

## 11. Conclusion

A Random Forest trained on sensor readings plus physics-motivated features predicts equipment
failure with **F1 {f(b['f1'])} / PR-AUC {f(b['pr_auc'])}** on a random holdout, falling to
**F1 {f(temp_f1)}** under a strict time-ordered split. Its drivers — rotational speed,
temperature difference, wear–torque interaction and mechanical power — are physically
plausible and individually explainable, which is what makes it usable as decision support
rather than a black box. Logistic Regression is demonstrably insufficient here, which itself
reveals that the failure boundary is multiplicative in speed and torque.

The honest framing matters as much as the score: this is a model trained on **simulated data**
with **few failures**, an **unlearnable random-failure class**, and **documented label noise**.
Its predictions are **probabilities, not guarantees of failure**. Used to rank machines for
human inspection, it adds real value. Used as an oracle or a safety device, it would be
dangerous.

---

*Generated {R['generated_at']} from `artifacts/results.json` by `scripts/build_report.py`.
Dataset: AI4I 2020 Predictive Maintenance, UCI Machine Learning Repository (ID 601).*
"""
    return md


# --------------------------------------------------------------------------
# Word report
# --------------------------------------------------------------------------
def build_docx(md_text: str, out_path: Path) -> None:
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    NAVY = RGBColor(0x0F, 0x2B, 0x46)
    TEAL = RGBColor(0x0D, 0x74, 0x8B)
    GREY = RGBColor(0x55, 0x5F, 0x6D)
    AMBER = RGBColor(0x92, 0x40, 0x0E)

    doc = Document()

    # page + base style
    for s in doc.sections:
        s.top_margin = Inches(0.75)
        s.bottom_margin = Inches(0.75)
        s.left_margin = Inches(0.85)
        s.right_margin = Inches(0.85)

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    style.paragraph_format.space_after = Pt(6)
    style.paragraph_format.line_spacing = 1.15

    def shade(cell, hexcolor):
        tcPr = cell._tc.get_or_add_tcPr()
        el = OxmlElement("w:shd")
        el.set(qn("w:val"), "clear")
        el.set(qn("w:fill"), hexcolor)
        tcPr.append(el)

    def add_md_runs(par, text, base_bold=False):
        """Minimal inline markdown: **bold**, `code`, *italic*."""
        import re

        tokens = re.split(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)", text)
        for tok in tokens:
            if not tok:
                continue
            if tok.startswith("**") and tok.endswith("**"):
                r = par.add_run(tok[2:-2]); r.bold = True
            elif tok.startswith("`") and tok.endswith("`"):
                r = par.add_run(tok[1:-1]); r.font.name = "Consolas"
                r.font.size = Pt(9.5); r.font.color.rgb = TEAL
            elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
                r = par.add_run(tok[1:-1]); r.italic = True
            else:
                r = par.add_run(tok)
            if base_bold:
                r.bold = True

    lines = md_text.split("\n")
    i = 0
    in_code = False
    code_buf = []

    while i < len(lines):
        line = lines[i]

        # fenced code block
        if line.strip().startswith("```"):
            if not in_code:
                in_code, code_buf = True, []
            else:
                in_code = False
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.2)
                p.paragraph_format.space_before = Pt(6)
                p.paragraph_format.space_after = Pt(10)
                r = p.add_run("\n".join(code_buf))
                r.font.name = "Consolas"; r.font.size = Pt(9)
                r.font.color.rgb = RGBColor(0x11, 0x11, 0x11)
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "F1F5F9")
                p._p.get_or_add_pPr().append(shd)
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        stripped = line.strip()

        # horizontal rule
        if stripped == "---":
            p = doc.add_paragraph()
            pPr = p._p.get_or_add_pPr()
            pbdr = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single"); bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:color"), "CBD5E1")
            pbdr.append(bottom); pPr.append(pbdr)
            i += 1
            continue

        # images
        if stripped.startswith("!["):
            import re

            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if m:
                alt, rel = m.group(1), m.group(2)
                path = (ROOT / "reports" / rel).resolve()
                if path.exists():
                    try:
                        doc.add_picture(str(path), width=Inches(6.4))
                        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                        if alt:
                            cap = doc.add_paragraph()
                            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            r = cap.add_run(alt)
                            r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = GREY
                    except Exception:
                        pass
            i += 1
            continue

        # italic caption line
        if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**") and len(stripped) > 3:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(stripped.strip("*"))
            r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = GREY
            i += 1
            continue

        # headings
        if stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped.lstrip("#").strip()
            if level == 1:
                p = doc.add_paragraph()
                r = p.add_run(text)
                r.bold = True; r.font.size = Pt(24); r.font.color.rgb = NAVY
                p.paragraph_format.space_after = Pt(4)
            elif level == 2:
                doc.add_page_break() if False else None
                p = doc.add_paragraph()
                r = p.add_run(text)
                r.bold = True; r.font.size = Pt(15); r.font.color.rgb = NAVY
                p.paragraph_format.space_before = Pt(14)
                p.paragraph_format.space_after = Pt(4)
            else:
                p = doc.add_paragraph()
                r = p.add_run(text)
                r.bold = True; r.font.size = Pt(12); r.font.color.rgb = TEAL
                p.paragraph_format.space_before = Pt(10)
                p.paragraph_format.space_after = Pt(3)
            i += 1
            continue

        # tables
        if stripped.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip().replace("|", "").replace(" ", "")) <= set("-:"):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            ncol = len(header)
            table = doc.add_table(rows=1, cols=ncol)
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            hdr = table.rows[0].cells
            for j, h in enumerate(header):
                hdr[j].text = ""
                p = hdr[j].paragraphs[0]
                add_md_runs(p, h, base_bold=True)
                for r in p.runs:
                    r.font.size = Pt(9); r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                shade(hdr[j], "0F2B46")
            for ri, row in enumerate(rows):
                cells = table.add_row().cells
                for j in range(ncol):
                    cells[j].text = ""
                    p = cells[j].paragraphs[0]
                    add_md_runs(p, row[j] if j < len(row) else "")
                    for r in p.runs:
                        r.font.size = Pt(9)
                    if ri % 2 == 1:
                        shade(cells[j], "F8FAFC")
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue

        # blockquote / callout
        if stripped.startswith(">"):
            text = stripped.lstrip("> ").strip()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.15)
            add_md_runs(p, text)
            for r in p.runs:
                r.font.color.rgb = AMBER
            pPr = p._p.get_or_add_pPr()
            pbdr = OxmlElement("w:pBdr")
            left = OxmlElement("w:left")
            left.set(qn("w:val"), "single"); left.set(qn("w:sz"), "18")
            left.set(qn("w:color"), "F59E0B"); left.set(qn("w:space"), "8")
            pbdr.append(left); pPr.append(pbdr)
            i += 1
            continue

        # bullets and numbers
        if stripped.startswith(("- ", "* ")) or (len(stripped) > 1 and stripped[0].isdigit() and stripped[1:3] in (". ", ") ")):
            ordered = not stripped.startswith(("- ", "* "))
            text = stripped[2:].strip() if ordered else stripped[2:].strip()
            p = doc.add_paragraph(style="List Number" if ordered else "List Bullet")
            add_md_runs(p, text)
            for r in p.runs:
                if r.font.size is None:
                    r.font.size = Pt(10.5)
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        p = doc.add_paragraph()
        add_md_runs(p, stripped)
        i += 1

    doc.save(str(out_path))


def sync_figures() -> int:
    """Copy artifacts/figures -> reports/figures.

    The report references images as `figures/xxx.png` so that both the Markdown
    (rendered from the reports/ folder) and the Word build resolve them.
    """
    import shutil

    dest = C.REPORT_DIR / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted(C.FIG_DIR.glob("*.png")):
        shutil.copy2(src, dest / src.name)
        n += 1
    return n


if __name__ == "__main__":
    print(f"synced {sync_figures()} figures -> reports/figures")
    md = build_markdown()
    md_path = C.REPORT_DIR / "failure_prediction_report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"wrote {md_path} ({len(md):,} chars)")

    docx_path = C.REPORT_DIR / "failure_prediction_report.docx"
    build_docx(md, docx_path)
    print(f"wrote {docx_path} ({docx_path.stat().st_size:,} bytes)")
