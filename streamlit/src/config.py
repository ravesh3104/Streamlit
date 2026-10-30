"""Central configuration for the Industrial Equipment Failure Prediction project.

Every modelling decision that the user chose during the design interview is
recorded here as an explicit, documented flag. Change a flag here and the
whole pipeline (cleaning -> training -> dashboard) follows.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
FIG_DIR = ARTIFACT_DIR / "figures"
REPORT_DIR = ROOT / "reports"

RAW_CSV = DATA_DIR / "raw_ai4i2020.csv"
CORRUPTED_CSV = DATA_DIR / "synthetic_corrupted_ai4i.csv"
CLEAN_CSV = DATA_DIR / "clean_ai4i2020.csv"

MODELS_PATH = ARTIFACT_DIR / "models.joblib"
RESULTS_PATH = ARTIFACT_DIR / "results.json"
CLEANING_LOG_PATH = ARTIFACT_DIR / "cleaning_log.json"
TEST_PREDICTIONS_PATH = ARTIFACT_DIR / "test_predictions.csv"

for _d in (DATA_DIR, ARTIFACT_DIR, FIG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset shape
# --------------------------------------------------------------------------
TARGET = "Machine failure"

# Failure-mode indicator columns. These are linear combinations of the target
# (TWF + HDF + PWF + OSF + RNF == Machine failure for every single row).
# They are DATA LEAKAGE if used as model inputs -> kept for EDA only.
FAILURE_MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Identifier columns.
ID_COLS = ["UDI", "Product ID"]
# DECISION: keep both as model inputs so the report can *demonstrate* that
# they carry no signal (they should rank at the bottom of every importance
# chart). Flip to False to exclude them - nothing else needs to change.
USE_ID_COLUMNS_AS_FEATURES = True

# Raw numeric sensor columns as delivered by UCI.
SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
CATEGORICAL_COL = "Type"

# DECISION: physics-motivated engineered features (see features.py).
ADD_ENGINEERED_FEATURES = True
ENGINEERED_COLS = [
    "Temp difference [K]",
    "Mechanical power [kW]",
    "Wear-torque interaction",
    "Speed-torque ratio",
    "Wear rate [min per kW]",
]

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
# DECISION: robust, reusable pipeline that is validated on a deliberately
# corrupted copy of the dataset (they/"corrupted-copy test").
VALID_RANGES = {
    # column: (hard minimum, hard maximum) - values outside are treated as
    # impossible sensor readings and set to missing before imputation.
    "Air temperature [K]": (250.0, 350.0),
    "Process temperature [K]": (250.0, 400.0),
    "Rotational speed [rpm]": (0.0, 5000.0),
    "Torque [Nm]": (0.0, 200.0),
    "Tool wear [min]": (0.0, 500.0),
}
IQR_MULTIPLIER = 3.0  # flag-only; we never silently delete extreme-but-real rows

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
# DECISION: report BOTH a random stratified split and a time-ordered split,
# because the random split can flatter a model on this dataset.
RUN_RANDOM_SPLIT = True
RUN_TEMPORAL_SPLIT = True
N_BOOTSTRAP = 300          # bootstrap resamples for confidence intervals
THRESHOLD_GRID = [round(x, 3) for x in __import__("numpy").arange(0.05, 0.951, 0.005)]

N_JOBS = -1

# --------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------
DISCLAIMER = (
    "Predictions are probabilistic estimates from a model trained on the "
    "simulated AI4I 2020 dataset. They are NOT guaranteed equipment failures. "
    "Use them only to prioritise human inspection and maintenance."
)
