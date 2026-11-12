"""
core.py — CreditGuard Reusable Functions
Handles: model loading, prediction, metrics, cost calculations,
         threshold optimization, and SHAP explanations.

No model training happens here. All functions consume pre-trained artifacts.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef,
)

from config import (
    THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP,
    DEFAULT_THRESHOLD, FEATURE_COLS, model_path,
)

# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

_pipeline_cache: dict = {}


def load_pipeline(model_name: str, strategy_name: str):
    """Load a fitted imblearn pipeline from disk. Cached in memory."""
    key = (model_name, strategy_name)
    if key not in _pipeline_cache:
        path = model_path(model_name, strategy_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}\n"
                f"Run `python train.py` first to generate artifacts."
            )
        _pipeline_cache[key] = joblib.load(path)
    return _pipeline_cache[key]


def clear_cache():
    """Clear the in-memory pipeline cache (useful for testing)."""
    _pipeline_cache.clear()


# ---------------------------------------------------------------------------
# 2. Prediction
# ---------------------------------------------------------------------------

def predict_proba(pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Return fraud probability scores for a DataFrame of raw transactions.
    The pipeline's prep step handles scaling internally.
    """
    return pipeline.predict_proba(X)[:, 1]


def predict(pipeline, X: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Return binary predictions (0/1) at the given threshold."""
    proba = predict_proba(pipeline, X)
    return (proba >= threshold).astype(int)


# ---------------------------------------------------------------------------
# 3. Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """
    Compute all evaluation metrics at a given decision threshold.

    Returns a dict with:
    Accuracy, Precision, Recall, F1, MCC, ROC_AUC, PR_AUC, TP, FP, FN, TN.

    Sources:
    - Accuracy: fraud.ipynb
    - MCC: fraud_extended.ipynb
    - All others: both notebooks
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)

    # Guard against edge cases (all predictions same class)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "Accuracy":  float(accuracy_score(y_true, pred)),
            "Precision": float(precision_score(y_true, pred, zero_division=0)),
            "Recall":    float(recall_score(y_true, pred, zero_division=0)),
            "F1":        float(f1_score(y_true, pred, zero_division=0)),
            "MCC":       float(matthews_corrcoef(y_true, pred)),
            "ROC_AUC":   float(roc_auc_score(y_true, proba)),
            "PR_AUC":    float(average_precision_score(y_true, proba)),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        }


def confusion_matrix_df(y_true, proba, threshold=DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Return a 2×2 confusion matrix as a labelled DataFrame."""
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=pd.Index(["Actual Legit", "Actual Fraud"], name=""),
        columns=pd.Index(["Predicted Legit", "Predicted Fraud"], name=""),
    )


# ---------------------------------------------------------------------------
# 4. Cost function
# ---------------------------------------------------------------------------

def total_cost(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    threshold: float,
    cost_fn: float,
    cost_fp: float,
) -> float:
    """
    Compute total financial cost at a given threshold.

    Formula: (FN × cost_fn) + (FP × cost_fp)

    Args:
        y_true:    Ground-truth labels (0/1).
        proba:     Predicted fraud probabilities.
        threshold: Decision threshold.
        cost_fn:   Cost per missed fraud (False Negative).
        cost_fp:   Cost per false alarm (False Positive).

    Returns:
        Total cost (float).
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return float(fn * cost_fn + fp * cost_fp)


def cost_sweep(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> pd.DataFrame:
    """
    Sweep decision thresholds from THRESHOLD_MIN to THRESHOLD_MAX and
    return a DataFrame with columns: threshold, cost, FN, FP.

    Used to draw the cost-vs-threshold chart in Tab 2.
    """
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    thresholds = np.round(thresholds, 4)
    rows = []
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(proba)
    for t in thresholds:
        pred = (p_arr >= t).astype(int)
        cm = confusion_matrix(y_arr, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rows.append({
            "threshold": float(t),
            "cost":      float(fn * cost_fn + fp * cost_fp),
            "FN":        int(fn),
            "FP":        int(fp),
            "TP":        int(tp),
            "TN":        int(tn),
        })
    return pd.DataFrame(rows)


def best_threshold(
    y_val: np.ndarray | pd.Series,
    proba_val: np.ndarray,
    cost_fn: float,
    cost_fp: float,
) -> float:
    """
    Find the threshold that minimises total cost on VALIDATION data.

    IMPORTANT: This function must only ever be called with validation
    probabilities, never test probabilities. Leakage prevention.

    Returns:
        Optimal threshold (float, rounded to 4 dp).
    """
    df = cost_sweep(y_val, proba_val, cost_fn, cost_fp)
    best_idx = df["cost"].idxmin()
    return float(df.loc[best_idx, "threshold"])


# ---------------------------------------------------------------------------
# 5. SHAP explanations
# ---------------------------------------------------------------------------

def _get_transformed_input(pipeline, X_raw: pd.DataFrame) -> np.ndarray:
    """
    Apply the pipeline's 'prep' (ColumnTransformer) step to raw features.
    Returns a numpy array in the same feature order the classifier sees.

    For SMOTE pipelines, the SMOTE step is skipped (it's a training-only step).
    """
    prep_step = pipeline.named_steps["prep"]
    return prep_step.transform(X_raw)


def get_shap_values(
    pipeline,
    X_raw: pd.DataFrame,
    background_data: np.ndarray | None = None,
    model_name: str = "",
) -> tuple[np.ndarray, list[str]]:
    """
    Compute SHAP values for a single transaction or small batch.

    Args:
        pipeline:        Fitted imblearn pipeline.
        X_raw:           Raw (unscaled) input DataFrame.
        background_data: Pre-transformed background rows for LinearExplainer.
        model_name:      "Logistic Regression", "Random Forest", or "XGBoost".

    Returns:
        (shap_values_1d, feature_names) for the first row of X_raw.
        shap_values_1d: 1D array of SHAP values (one per feature), positive = toward fraud.
        feature_names: list of feature names matching shap_values order.
    """
    import shap

    clf = pipeline.named_steps["clf"]
    X_transformed = _get_transformed_input(pipeline, X_raw)

    if model_name == "Logistic Regression":
        if background_data is None:
            raise ValueError("background_data required for LinearExplainer.")
        explainer = shap.LinearExplainer(clf, background_data)
        sv = explainer.shap_values(X_transformed)
        # LinearExplainer returns (n_samples, n_features) for binary — class 1 contributions
        if isinstance(sv, list):
            sv = sv[1]
        shap_row = np.asarray(sv[0])
    else:
        # TreeExplainer for Random Forest and XGBoost
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_transformed)

        # Handle different SHAP output shapes across versions:
        # SHAP < 0.42: RF returns list [class0_sv, class1_sv] each (n_samples, n_features)
        # SHAP >= 0.42: RF may return (n_samples, n_features, n_classes) or (n_samples, n_features)
        # XGBoost binary → (n_samples, n_features) — class 1 contributions directly
        if isinstance(sv, list):
            # Old-style list: take class 1 (fraud)
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            # (n_samples, n_features, n_classes) — take class 1
            shap_row = sv[0, :, 1]
        elif sv.ndim == 2:
            # (n_samples, n_features) — single sample is first row
            # But check if it's (n_features, n_classes) for a single sample
            if sv.shape[0] == X_transformed.shape[1]:
                # Likely (n_features, n_classes) for single sample — take class 1
                shap_row = sv[:, 1] if sv.shape[1] == 2 else sv[:, 0]
            else:
                shap_row = sv[0]
        elif sv.ndim == 1:
            shap_row = sv
        else:
            shap_row = sv.ravel()[:X_transformed.shape[1]]

    return shap_row, FEATURE_COLS


def top_shap_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    n: int = 8,
) -> pd.DataFrame:
    """
    Return the top-n features by absolute SHAP value as a DataFrame.

    Columns: feature, value (SHAP contribution), abs_value, direction.
    direction: "Toward Fraud" (positive) or "Toward Legitimate" (negative).
    """
    df = pd.DataFrame({
        "feature":   feature_names,
        "shap_value": shap_values,
        "abs_value":  np.abs(shap_values),
    })
    df["direction"] = df["shap_value"].apply(
        lambda v: "Toward Fraud" if v > 0 else "Toward Legitimate"
    )
    return df.nlargest(n, "abs_value").reset_index(drop=True)


def shap_plain_english(top_df: pd.DataFrame, fraud_prob: float, threshold: float) -> str:
    """
    Generate a plain-English explanation of the top SHAP contributions.
    """
    decision = "FRAUD DETECTED" if fraud_prob >= threshold else "LEGITIMATE TRANSACTION"
    lines = [
        f"**Decision:** {decision} (probability {fraud_prob:.1%}, threshold {threshold:.2f})",
        "",
        "**Top contributing features:**",
    ]
    for _, row in top_df.iterrows():
        direction_word = "increases" if row["direction"] == "Toward Fraud" else "decreases"
        lines.append(
            f"- **{row['feature']}**: {direction_word} fraud probability "
            f"(SHAP contribution: {row['shap_value']:+.4f})"
        )
    lines += [
        "",
        "_SHAP explains what the model learned from historical data. "
        "It describes model behaviour, not causation._",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Leaderboard utilities
# ---------------------------------------------------------------------------

def compute_leaderboard_costs(
    test_scores: pd.DataFrame,
    y_test: pd.Series,
    cost_fn: float,
    cost_fp: float,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.Series:
    """
    Given a DataFrame of test probability columns (one per model/strategy combo),
    compute the total cost for each configuration at the given threshold.

    Returns a Series indexed by column name.
    """
    results = {}
    for col in test_scores.columns:
        proba = test_scores[col].values
        c = total_cost(y_test.values, proba, threshold, cost_fn, cost_fp)
        results[col] = c
    return pd.Series(results)


def score_col_name(model_name: str, strategy_name: str) -> str:
    """Canonical column name for test/validation score DataFrames."""
    return f"{model_name}|{strategy_name}"


def parse_score_col(col: str) -> tuple[str, str]:
    """Split 'Model|Strategy' column name back into (model, strategy)."""
    parts = col.split("|", 1)
    return parts[0], parts[1]

