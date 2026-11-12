"""
train.py — CreditGuard Offline Training Script
Run once: python train.py

What this script does:
1. Loads creditcard.csv
2. Creates stratified 80/20 train/test split (seed 42)
3. Creates an additional 75/25 val split from train (for threshold tuning)
4. Trains all 9 model/strategy pipelines
5. Saves each fitted pipeline to artifacts/models/
6. Saves test probabilities, validation probabilities, test features/labels
7. Saves leaderboard.csv (metrics at threshold 0.5)
8. Saves SHAP background data
9. Saves metadata.json

IMPORTANT DATA RULES (see inspection report):
- Scaler fitted on training data only (via ColumnTransformer inside Pipeline)
- SMOTE applied to training data only (inside Pipeline)
- scale_pos_weight computed from y_train only (not full y)
- Validation set only used for threshold tuning — never for model fitting
- Test set never used for fitting or threshold selection
"""

import json
import time
import warnings
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef,
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import project config
# ---------------------------------------------------------------------------
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from config import (
    DATA_PATH, ARTIFACTS_DIR, MODELS_DIR,
    TEST_DATA_PATH, TEST_SCORES_PATH, VAL_SCORES_PATH,
    LEADERBOARD_PATH, SHAP_BG_PATH, METADATA_PATH,
    SEED, TEST_SIZE, VAL_SIZE,
    SCALE_COLS, FEATURE_COLS, TARGET_COL,
    MODELS, STRATEGIES,
    LR_PARAMS, LR_PARAMS_WEIGHTED,
    RF_PARAMS, RF_PARAMS_WEIGHTED,
    XGB_PARAMS, SMOTE_PARAMS,
    SHAP_BG_SIZE, DEFAULT_THRESHOLD,
    model_path,
)
from core import score_col_name

# ---------------------------------------------------------------------------
# Ensure artifact directories exist
# ---------------------------------------------------------------------------
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: build preprocessing step
# ---------------------------------------------------------------------------
def prep():
    """
    ColumnTransformer: StandardScale Time & Amount, passthrough V1-V28.
    Output feature order: [Time, Amount, V1, V2, ..., V28] = FEATURE_COLS.
    Source: fraud_extended.ipynb
    """
    return ColumnTransformer(
        [("sc", StandardScaler(), SCALE_COLS)],
        remainder="passthrough",
    )


# ---------------------------------------------------------------------------
# Helper: build classifier
# ---------------------------------------------------------------------------
def make_clf(model_name: str, weighted: bool, spw: float):
    """
    Return an unfitted classifier with exact hyperparameters from notebooks.
    weighted=True applies class_weight / scale_pos_weight.
    spw = scale_pos_weight computed from y_train (not full y).
    """
    if model_name == "Logistic Regression":
        params = LR_PARAMS_WEIGHTED if weighted else LR_PARAMS
        return LogisticRegression(**params)

    if model_name == "Random Forest":
        params = RF_PARAMS_WEIGHTED if weighted else RF_PARAMS
        return RandomForestClassifier(**params)

    # XGBoost
    xgb_params = dict(XGB_PARAMS)
    xgb_params["scale_pos_weight"] = spw if weighted else 1
    return XGBClassifier(**xgb_params)


# ---------------------------------------------------------------------------
# Helper: build full pipeline
# ---------------------------------------------------------------------------
def build_pipeline(model_name: str, strategy: str, spw: float) -> Pipeline:
    """
    Construct the imblearn Pipeline for one (model, strategy) combination.

    Strategies:
      Baseline       → prep + clf (no resampling, no class weight)
      Class weighting → prep + clf (with class_weight / scale_pos_weight)
      SMOTE          → prep + SMOTE + clf (no class weight)

    Source: fraud_extended.ipynb build() function
    """
    weighted = (strategy == "Class weighting")
    clf = make_clf(model_name, weighted, spw)
    steps = [("prep", prep())]
    if strategy == "SMOTE":
        steps.append(("smote", SMOTE(**SMOTE_PARAMS)))
    steps.append(("clf", clf))
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Helper: evaluate at threshold 0.5
# ---------------------------------------------------------------------------
def evaluate_at_half(y_true, proba) -> dict:
    """Full metric dict at threshold 0.5."""
    pred = (np.asarray(proba) >= DEFAULT_THRESHOLD).astype(int)
    y_true = np.asarray(y_true)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "Accuracy":       float(accuracy_score(y_true, pred)),
            "Precision":      float(precision_score(y_true, pred, zero_division=0)),
            "Recall":         float(recall_score(y_true, pred, zero_division=0)),
            "F1":             float(f1_score(y_true, pred, zero_division=0)),
            "MCC":            float(matthews_corrcoef(y_true, pred)),
            "ROC_AUC":        float(roc_auc_score(y_true, proba)),
            "PR_AUC":         float(average_precision_score(y_true, proba)),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        }


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def main():
    total_start = time.time()

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("CreditGuard — Training Script")
    print("=" * 65)
    print(f"\n[1/6] Loading {DATA_PATH} ...", end=" ", flush=True)

    if not DATA_PATH.exists():
        print(f"\nERROR: {DATA_PATH} not found.")
        print("Place creditcard.csv in the ml_minipopo/ directory.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=TARGET_COL)
    y = df[TARGET_COL]

    n_total  = len(df)
    n_fraud  = int(y.sum())
    n_legit  = n_total - n_fraud
    fraud_pct = 100 * n_fraud / n_total

    print("done.")
    print(f"    Rows: {n_total:,} | Fraud: {n_fraud} ({fraud_pct:.4f}%) | Legit: {n_legit:,}")

    # -----------------------------------------------------------------------
    # 2. Stratified train / test split  (80/20, seed 42)
    # -----------------------------------------------------------------------
    print(f"\n[2/6] Splitting data (test={TEST_SIZE}, seed={SEED}) ...", end=" ", flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    # Validation split from TRAIN only (for threshold tuning — never touches test)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=SEED
    )

    print("done.")
    print(f"    Train : {len(X_train):,} rows (fraud {int(y_train.sum())})")
    print(f"    Fit   : {len(X_fit):,} rows  (fraud {int(y_fit.sum())})")
    print(f"    Val   : {len(X_val):,} rows  (fraud {int(y_val.sum())})")
    print(f"    Test  : {len(X_test):,} rows  (fraud {int(y_test.sum())})")

    # scale_pos_weight — computed from y_TRAIN only (corrects leakage in extended notebook)
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"    scale_pos_weight (from y_train): {spw:.2f}")

    # -----------------------------------------------------------------------
    # 3. Train all 9 pipelines
    # -----------------------------------------------------------------------
    print(f"\n[3/6] Training 9 pipelines ...\n")
    print(f"{'Model':<22} {'Strategy':<18} {'F1':>6} {'PR-AUC':>8} {'Time':>8}")
    print("-" * 65)

    leaderboard_rows = []
    test_scores = {}   # col_name → test probabilities (on X_test, full train model)
    val_scores  = {}   # col_name → val  probabilities (on X_val,  fit-subset model)

    for strategy in STRATEGIES:
        for model_name in MODELS:
            col = score_col_name(model_name, strategy)

            # --- Train on FULL X_train (for leaderboard + test probabilities) ---
            pipe_full = build_pipeline(model_name, strategy, spw)
            t0 = time.time()
            pipe_full.fit(X_train, y_train)
            train_time = time.time() - t0

            proba_test = pipe_full.predict_proba(X_test)[:, 1]
            metrics = evaluate_at_half(y_test, proba_test)
            metrics["Model"]        = model_name
            metrics["Strategy"]     = strategy
            metrics["Train_time_s"] = round(train_time, 2)
            leaderboard_rows.append(metrics)
            test_scores[col] = proba_test

            # Save fitted pipeline
            out_path = model_path(model_name, strategy)
            joblib.dump(pipe_full, out_path)

            # --- Train on X_fit (for validation probabilities — threshold tuning) ---
            pipe_fit = build_pipeline(model_name, strategy, spw)
            pipe_fit.fit(X_fit, y_fit)
            proba_val = pipe_fit.predict_proba(X_val)[:, 1]
            val_scores[col] = proba_val

            print(
                f"  {model_name:<20} {strategy:<18} "
                f"{metrics['F1']:>6.3f} {metrics['PR_AUC']:>8.3f} "
                f"{train_time:>6.1f}s"
            )

    # -----------------------------------------------------------------------
    # 4. Save artifacts
    # -----------------------------------------------------------------------
    print(f"\n[4/6] Saving artifacts ...", end=" ", flush=True)

    # Leaderboard CSV
    lb_df = pd.DataFrame(leaderboard_rows)
    col_order = [
        "Model", "Strategy", "Accuracy", "Precision", "Recall",
        "F1", "MCC", "ROC_AUC", "PR_AUC",
        "TP", "FP", "FN", "TN", "Train_time_s",
    ]
    lb_df = lb_df[col_order]
    lb_df.round(6).to_csv(LEADERBOARD_PATH, index=False)

    # Test features (raw, unscaled) and labels
    X_test_save = X_test.copy().reset_index(drop=True)
    X_test_save.to_parquet(TEST_DATA_PATH, index=False)

    # Test labels saved inside test_data parquet as _label column
    test_label_df = X_test_save.copy()
    test_label_df["_label"] = y_test.values
    test_label_df.to_parquet(TEST_DATA_PATH, index=False)

    # Test scores (probabilities on held-out test set, full-train model)
    ts_df = pd.DataFrame(test_scores)
    ts_df["_label"] = y_test.values
    ts_df.to_parquet(TEST_SCORES_PATH, index=False)

    # Validation scores (probabilities from fit-subset model, on val set)
    vs_df = pd.DataFrame(val_scores)
    vs_df["_label"] = y_val.values
    vs_df.to_parquet(VAL_SCORES_PATH, index=False)

    print("done.")

    # -----------------------------------------------------------------------
    # 5. SHAP background data
    # -----------------------------------------------------------------------
    print(f"[5/6] Building SHAP background ...", end=" ", flush=True)

    # Use XGBoost + Class weighting pipeline (best single-model SHAP source)
    # Apply only the prep step to X_test to get transformed features
    xgb_pipe = joblib.load(model_path("XGBoost", "Class weighting"))
    prep_step = xgb_pipe.named_steps["prep"]

    # Transform test set
    X_test_tf = prep_step.transform(X_test.reset_index(drop=True))

    # Sample 2000 rows + all fraud rows (matches fraud_extended.ipynb approach)
    rng = np.random.default_rng(SEED)
    n_sample = min(SHAP_BG_SIZE, len(X_test_tf))
    sample_idx = rng.choice(len(X_test_tf), size=n_sample, replace=False)
    fraud_idx = np.where(y_test.values == 1)[0]
    all_idx = np.unique(np.concatenate([sample_idx, fraud_idx]))

    bg_array = X_test_tf[all_idx]
    bg_df = pd.DataFrame(bg_array, columns=FEATURE_COLS)
    bg_df["_label"] = y_test.values[all_idx]
    bg_df.to_parquet(SHAP_BG_PATH, index=False)

    print(f"done. ({len(bg_df)} background rows)")

    # -----------------------------------------------------------------------
    # 6. Metadata
    # -----------------------------------------------------------------------
    print(f"[6/6] Saving metadata ...", end=" ", flush=True)

    import sklearn, imblearn, xgboost, shap as shap_lib
    metadata = {
        "trained_at":      datetime.now().isoformat(),
        "total_rows":      n_total,
        "fraud_rows":      n_fraud,
        "legit_rows":      n_legit,
        "train_rows":      len(X_train),
        "fit_rows":        len(X_fit),
        "val_rows":        len(X_val),
        "test_rows":       len(X_test),
        "scale_pos_weight": spw,
        "seed":            SEED,
        "test_size":       TEST_SIZE,
        "val_size":        VAL_SIZE,
        "library_versions": {
            "numpy":          np.__version__,
            "pandas":         pd.__version__,
            "sklearn":        sklearn.__version__,
            "imblearn":       imblearn.__version__,
            "xgboost":        xgboost.__version__,
            "shap":           shap_lib.__version__,
        },
        "models":     MODELS,
        "strategies": STRATEGIES,
        "feature_cols": FEATURE_COLS,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("done.")

    total_time = time.time() - total_start
    print(f"\n{'=' * 65}")
    print(f"Training complete in {total_time:.1f}s")
    print(f"Artifacts saved to: {ARTIFACTS_DIR}")
    print(f"{'=' * 65}\n")

    # Print mini leaderboard
    print(lb_df[["Model", "Strategy", "F1", "PR_AUC", "ROC_AUC", "Train_time_s"]]
          .sort_values("F1", ascending=False)
          .to_string(index=False))
    print()


if __name__ == "__main__":
    main()
