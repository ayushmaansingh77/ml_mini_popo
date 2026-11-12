"""
config.py — CreditGuard Central Configuration
All constants, paths, model hyperparameters, and naming conventions live here.
Changing a value here propagates to train.py, core.py, and app.py automatically.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent          # fraud_app/
DATA_DIR = BASE_DIR.parent                # ml_minipopo/ (where creditcard.csv lives)
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

DATA_PATH = DATA_DIR / "creditcard.csv"
TEST_DATA_PATH = ARTIFACTS_DIR / "test_data.parquet"
TEST_SCORES_PATH = ARTIFACTS_DIR / "test_scores.parquet"
VAL_SCORES_PATH = ARTIFACTS_DIR / "validation_scores.parquet"
LEADERBOARD_PATH = ARTIFACTS_DIR / "leaderboard.csv"
SHAP_BG_PATH = ARTIFACTS_DIR / "shap_background.parquet"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20   # 80/20 stratified train-test split
VAL_SIZE = 0.25    # 25% of train → validation set for threshold tuning
                   # (= 20% of full dataset)

# ---------------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------------
TARGET_COL = "Class"
SCALE_COLS = ["Time", "Amount"]                              # columns to StandardScale
V_COLS = [f"V{i}" for i in range(1, 29)]                    # V1 .. V28 (PCA components)
FEATURE_COLS = SCALE_COLS + V_COLS                           # 30 input features
# After ColumnTransformer the output order is:
# [Time(scaled), Amount(scaled), V1, V2, ..., V28]  (same order as FEATURE_COLS)

# ---------------------------------------------------------------------------
# Model & strategy definitions
# ---------------------------------------------------------------------------
MODELS = ["Logistic Regression", "Random Forest", "XGBoost"]
STRATEGIES = ["Baseline", "Class weighting", "SMOTE"]

# Slugs for file naming
MODEL_SLUG = {
    "Logistic Regression": "lr",
    "Random Forest":       "rf",
    "XGBoost":             "xgb",
}
STRATEGY_SLUG = {
    "Baseline":        "baseline",
    "Class weighting": "classweighting",
    "SMOTE":           "smote",
}

def model_path(model_name: str, strategy_name: str) -> Path:
    """Return the joblib file path for a fitted pipeline."""
    slug = f"{MODEL_SLUG[model_name]}_{STRATEGY_SLUG[strategy_name]}"
    return MODELS_DIR / f"{slug}.pkl"

# ---------------------------------------------------------------------------
# Hyperparameters — sourced directly from both notebooks (identical)
# ---------------------------------------------------------------------------
# Logistic Regression
LR_PARAMS = dict(max_iter=1000, random_state=SEED)
LR_PARAMS_WEIGHTED = dict(**LR_PARAMS, class_weight="balanced")

# Random Forest
RF_PARAMS = dict(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED)
RF_PARAMS_WEIGHTED = dict(**RF_PARAMS, class_weight="balanced_subsample")

# XGBoost (scale_pos_weight is computed at training time from y_train)
XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    tree_method="hist", n_jobs=-1, random_state=SEED,
    eval_metric="logloss",
)

# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------
SMOTE_PARAMS = dict(random_state=SEED)

# ---------------------------------------------------------------------------
# SHAP background sample size
# ---------------------------------------------------------------------------
SHAP_BG_SIZE = 2_000   # random test-set rows + all fraud rows (deduplicated)

# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------
THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
THRESHOLD_STEP = 0.01

# ---------------------------------------------------------------------------
# Leaderboard default threshold
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# UI defaults
# ---------------------------------------------------------------------------
DEFAULT_COST_FN = 500   # $ cost of one missed fraud
DEFAULT_COST_FP = 10    # $ cost of one false alarm

