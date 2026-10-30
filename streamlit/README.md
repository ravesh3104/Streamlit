# ⚙️ Industrial Equipment Failure Prediction

A machine-learning decision-support system that estimates the **probability** that industrial
equipment will fail, using sensor measurements and operating conditions — with a full,
auditable workflow from raw data to an interactive dashboard.

> ⚠️ **These are probabilities, not guarantees of failure.** The model is trained on a
> *simulated* public dataset. Use it to prioritise human inspection — never as a safety
> device, an automatic shutdown trigger, or proof that a machine is healthy.

---

## Headline results

Measured on a held-out test set (untouched by training or threshold tuning):

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.537 | 0.426 | 0.475 | 0.446 | 0.941 |
| Decision Tree | 0.911 | 0.750 | 0.823 | 0.835 | 0.909 |
| **Random Forest** ⭐ | **0.946** | **0.779** | **0.855** | **0.883** | **0.979** |

Under a stricter **time-ordered** split (train on past runs, predict future ones) the best
model scores **F1 0.727** — the more honest expectation for deployment. Both are reported;
see the report for why.

---

> 📖 **New to this? Follow [`GETTING_STARTED.md`](GETTING_STARTED.md)** — a step-by-step guide
> from installing Python to a running dashboard, including a troubleshooting table.
> Windows and macOS/Linux users can also just double-click `RUN_WINDOWS.bat` / `./RUN_MAC_LINUX.sh`.

## Quick start

```bash
# 1. install
pip install -r requirements.txt
python scripts/check_env.py     # verify everything works (recommended)

# 2. run the whole pipeline (clean → train → evaluate → explain → save artifacts)
python -m src.train          # ~60 seconds
# or: make train

# 3. launch the dashboard
streamlit run app/streamlit_app.py
# or: make app
```

Then open the URL Streamlit prints (usually <http://localhost:8501>).

**Notebook:** `notebooks/01_failure_prediction_walkthrough.ipynb` — the narrative version with
all outputs already saved, so you can read it without running anything.

**Report:** `reports/failure_prediction_report.md` and `.docx`.

### Docker

```bash
docker build -t failure-prediction .
docker run -p 8501:8501 failure-prediction
```

---

## Project structure

```
├── data/
│   ├── raw_ai4i2020.csv               raw dataset snapshot (offline-reproducible)
│   ├── clean_ai4i2020.csv             cleaned data + engineered features
│   └── synthetic_corrupted_ai4i.csv   deliberately damaged copy (cleaning test)
├── src/
│   ├── config.py        every decision as an explicit, documented constant
│   ├── cleaning.py      import → audit → clean → feature engineering
│   ├── modeling.py      design matrices, 3 models, threshold tuning, evaluation, explain
│   ├── plots.py         14 themed figures
│   └── train.py         orchestrates the pipeline and persists artifacts
├── app/
│   └── streamlit_app.py the dashboard (4 views)
├── notebooks/
│   └── 01_failure_prediction_walkthrough.ipynb
├── scripts/
│   ├── build_notebook.py  regenerates the notebook from source
│   └── build_report.py    regenerates the .md + .docx report from results.json
├── artifacts/           models.joblib, results.json, figures/, test_predictions.csv
├── reports/             report (.md/.docx) + figures
├── Dockerfile · Makefile · requirements.txt
```

---

## The workflow

```
data/raw_ai4i2020.csv
         │
         ▼
 ① IMPORT + CLEAN ─────────  audit → type coercion → range validation → dedupe
         │                   → label-integrity check → median impute → IQR flag
         ▼                   (+ validated on a deliberately corrupted copy)
 ② FEATURE ENGINEERING ────  5 physics-motivated features
         │
         ▼
 ③ TRAINING ───────────────  LR / Decision Tree / Random Forest, class-weighted,
         │                   random stratified split + time-ordered split
         ▼
 ④ EVALUATION ─────────────  precision, recall, F1, PR-AUC, MCC, confusion matrices,
         │                   bootstrap CIs, 5-fold CV
         ▼
 ⑤ EXPLANATION ────────────  permutation importance, Gini, LR coefficients, PDP
         │
         ▼
 ⑥ PREDICTION + DASHBOARD ──  single machine, batch CSV scoring, live threshold slider
```

---

## The dashboard

| View | What it does |
|---|---|
| **🔧 Machine Risk Check** | Five sensor sliders + machine grade → failure probability, a risk gauge, all three models side by side, and a "why" panel naming the inputs that moved *this* prediction. Preset failure scenarios included. |
| **📁 Batch Scoring** | Upload a CSV → cleaned, scored, ranked by risk, downloadable. Shows every cleaning repair applied to your file, and scores precision/recall/F1 if the file includes ground-truth labels. |
| **📈 Model Evaluation** | Leaderboard, confusion matrices, PR/ROC, threshold sensitivity, CV stability, bootstrap CIs, random vs time-ordered split. |
| **🔍 Patterns & Drivers** | All EDA charts, permutation/Gini importance, partial dependence, LR coefficients, the identifier-column verdict, cleaning audit. |

Views are switched with pill buttons at the top, and **only the selected view is executed** —
the evaluation and patterns views load ~1.2 MB of charts, so rendering them lazily keeps the
first paint fast (especially through a proxy or tunnel).

The alarm threshold is a **live slider** — because a missed failure and a wasted inspection do
not cost the same, and that trade-off belongs to the maintenance team, not the model.

---

## Dataset

**AI4I 2020 Predictive Maintenance** — UCI Machine Learning Repository, ID 601.
10,000 independent machine runs, 5 sensors, machine grade, binary failure label and 5
failure-mode flags. Bundled at `data/raw_ai4i2020.csv`.

- **Refreshing the raw data:** `python -m src.download` (or `make data`).

### Two important data traps, both handled

1. **Leakage.** The five columns `TWF, HDF, PWF, OSF, RNF` are failure-mode flags that only
   exist *because* a failure occurred. Feeding them in as inputs yields a meaningless ~100%
   model. **They are excluded from every model input** and used only for description.
2. **Label noise.** Those flags do not perfectly reconstruct the target — 18 rows flag a mode
   while labelled healthy, and 9 failures have no flag. This is reported as a documented
   limitation; the published labels are deliberately **not** rewritten.

---

## Which inputs drive the predictions?

Ranked by permutation importance (drop in PR-AUC when shuffled) for the Random Forest:

| Rank | Feature | Why it makes physical sense |
|---|---|---|
| 1 | `Rotational speed [rpm]` | Defines the operating point together with torque |
| 2 | `Temp difference [K]` | Large process−air gap ⇒ heat cannot be dissipated |
| 3 | `Wear-torque interaction` | Cumulative strain: worn tool still under load |
| 4 | `Mechanical power [kW]` | Torque × angular velocity = the real load on the machine |

The model's reasoning aligns with this machine's known failure mechanisms, which is the main
reason to trust it — more so than its accuracy score.

**The identifier columns (`UDI`, `Product ID`) were kept as inputs as an explicit experiment.**
Verdict: they are noise (combined 1.3% of permutation importance; `UDI`'s effect is *negative*).
Set `USE_ID_COLUMNS_AS_FEATURES = False` in `src/config.py` to drop them.

---

## Limitations — read before trusting any number

1. **Simulated data.** AI4I 2020 is generated, not logged from real machines. It has no sensor
   drift, no correlated channels, no dropouts, no maintenance interventions. **The method
   transfers to a real plant; the absolute numbers do not.**
2. **Few failures.** 339 total, 68 in the test set → F1 95% CIs of roughly ±0.07. A different
   seed moves results by several points.
3. **Some failures are unlearnable.** The 19 `RNF` (random failure) records have no sensor
   signature. No model can catch them; that caps achievable recall.
4. **Truncated tool-wear failures.** Every `TWF` record sits at 198–253 min of tool wear — just
   before the replacement limit — so that mode is partly an artefact of the replacement policy.
5. **Label noise.** See above: 27 rows are internally inconsistent.
6. **No per-machine time history.** Rows are independent runs, so real degradation trends — the
   strongest signal in a real plant — cannot be modelled. This is the biggest gap to a
   deployable system.
7. **Importance ≠ causality.** "The model used speed" does not mean "change the speed".
8. **The threshold is policy**, tuned on this dataset. Re-tune it against real maintenance costs.
9. **Not a safety system.** Decision support for prioritising inspections only.

See `reports/failure_prediction_report.md` §9 for the full discussion.

---

## Configuration

Every modelling decision lives in `src/config.py` as a named constant, including:

| Flag | Default | Effect |
|---|---|---|
| `USE_ID_COLUMNS_AS_FEATURES` | `True` | keep `UDI`/`Product ID` as inputs (the experiment above) |
| `ADD_ENGINEERED_FEATURES` | `True` | the 5 physics-motivated features |
| `RUN_RANDOM_SPLIT` / `RUN_TEMPORAL_SPLIT` | `True` / `True` | both validation views |
| `IQR_MULTIPLIER` | `3.0` | outlier *flagging* threshold (never deletion) |
| `CV_FOLDS`, `TEST_SIZE`, `RANDOM_STATE` | `5`, `0.20`, `42` | reproducibility |

---

## Reproducing every artifact

```bash
make all       # data → train → report → notebook
make train     # clean, train, evaluate, explain, save
make report    # regenerate reports/*.md and *.docx from results.json
make notebook  # rebuild + execute the notebook
make app       # launch the dashboard
make clean     # remove generated artifacts
```

---

## Credits

Dataset: *AI4I 2020 Predictive Maintenance Dataset*, UCI Machine Learning Repository (ID 601).
Built with pandas, NumPy, scikit-learn, Matplotlib, Seaborn and Streamlit.
