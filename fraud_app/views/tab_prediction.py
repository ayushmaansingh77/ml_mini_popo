"""
views/tab_prediction.py — Tab 3: Transaction Prediction + SHAP Explainability
Allows manual entry, CSV upload, or test-set sampling.
Generates fraud probability, decision outcome, and SHAP local attribution chart.
"""

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    MODELS, STRATEGIES, FEATURE_COLS, SCALE_COLS, SEED, DEFAULT_THRESHOLD,
)
from core import (
    load_pipeline, predict_proba, get_shap_values, top_shap_features, shap_plain_english,
)
from ui.components import render_prediction_badge, render_shap_disclaimer
from ui.plots import plot_shap_contributions


def render_prediction_tab(
    X_test_df: pd.DataFrame,
    y_test: pd.Series,
    shap_background: np.ndarray,
):
    """Render Tab 3: Transaction Prediction + SHAP."""
    st.markdown("### 🔍 Transaction Prediction + SHAP Explainability")

    # --- Model & Strategy Selection ---
    p3c1, p3c2 = st.columns(2)
    with p3c1:
        t3_model = st.selectbox("Model", MODELS, key="t3_model")
    with p3c2:
        t3_strat = st.selectbox("Strategy", STRATEGIES, key="t3_strat")

    t3_thr = st.slider(
        "Decision Threshold",
        min_value=0.01,
        max_value=0.99,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        key="t3_thr",
        help="Transactions with fraud probability ≥ threshold are classified as fraud.",
    )

    st.divider()

    # --- Input Method Selector ---
    input_method = st.radio(
        "Input method",
        ["Manual entry", "Upload CSV", "Use test-set sample"],
        horizontal=True,
        key="t3_input",
    )

    X_pred = None

    # ---- 1. Manual Entry ----
    if input_method == "Manual entry":
        st.markdown("**Enter transaction values** (V1–V28 are PCA components; Time in seconds; Amount in USD)")
        with st.expander("Transaction features", expanded=True):
            cols_per_row = 5
            feature_values = {}
            field_rows = [
                FEATURE_COLS[i : i + cols_per_row]
                for i in range(0, len(FEATURE_COLS), cols_per_row)
            ]
            for row_fields in field_rows:
                fcols = st.columns(len(row_fields))
                for fc, fname in zip(fcols, row_fields):
                    default_val = 50.0 if fname == "Amount" else 0.0
                    feature_values[fname] = fc.number_input(
                        fname,
                        value=default_val,
                        format="%.4f" if fname not in SCALE_COLS else "%.2f",
                        key=f"manual_{fname}",
                    )
        X_pred = pd.DataFrame([feature_values])

    # ---- 2. Upload CSV ----
    elif input_method == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload CSV (must contain features: " + ", ".join(FEATURE_COLS) + ")",
            type="csv",
            key="t3_csv",
        )
        if uploaded:
            try:
                X_upload = pd.read_csv(uploaded)
                if "Class" in X_upload.columns:
                    X_upload = X_upload.drop(columns=["Class"])
                missing_cols = [c for c in FEATURE_COLS if c not in X_upload.columns]
                extra_cols = [c for c in X_upload.columns if c not in FEATURE_COLS]
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    if extra_cols:
                        st.warning(f"Extra columns ignored: {', '.join(extra_cols)}")
                    X_pred = X_upload[FEATURE_COLS].head(50)
                    st.success(f"Loaded {len(X_pred)} transaction(s).")
            except Exception as e:
                st.error(f"Could not parse uploaded CSV: {e}")

    # ---- 3. Test-Set Sample ----
    else:
        n_sample = st.slider("Number of test samples", 1, 20, 5, key="t3_n_sample")
        sample_type = st.radio(
            "Sample type",
            ["Random", "Known fraud", "Known legitimate"],
            horizontal=True,
            key="t3_sample_type",
        )
        if sample_type == "Known fraud":
            pool = X_test_df[y_test.values == 1]
        elif sample_type == "Known legitimate":
            pool = X_test_df[y_test.values == 0]
        else:
            pool = X_test_df
        X_pred = pool.sample(min(n_sample, len(pool)), random_state=SEED).reset_index(drop=True)
        st.info(f"Sampled {len(X_pred)} transaction(s) from held-out test data.")

    # --- Run Prediction & Generate SHAP ---
    if X_pred is not None and len(X_pred) > 0:
        st.divider()
        if st.button("🔍 Predict & Explain", type="primary", key="t3_predict"):
            with st.spinner("Loading model and generating prediction..."):
                try:
                    pipeline = load_pipeline(t3_model, t3_strat)
                    probas = predict_proba(pipeline, X_pred)
                    preds = (probas >= t3_thr).astype(int)
                except FileNotFoundError as e:
                    st.error(str(e))
                    st.stop()

            # Render individual predictions & SHAP plots
            for i in range(len(X_pred)):
                p = probas[i]
                pred_label = preds[i]

                st.markdown(f"---\n#### Transaction {i + 1}")
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Fraud Probability", f"{p:.4f}")
                r2.metric("Threshold", f"{t3_thr:.2f}")
                r3.metric("Model", t3_model)
                r4.metric("Strategy", t3_strat)

                render_prediction_badge(is_fraud=(pred_label == 1))

                # SHAP local attribution
                with st.spinner("Computing SHAP local attribution..."):
                    try:
                        shap_vals, feat_names = get_shap_values(
                            pipeline,
                            X_pred.iloc[[i]],
                            background_data=shap_background,
                            model_name=t3_model,
                        )
                        top_df = top_shap_features(shap_vals, feat_names, n=8)

                        shap_chart_png = plot_shap_contributions(
                            top_df=top_df,
                            x_sample_row=X_pred.iloc[i],
                            title=f"Top 8 SHAP Features — {t3_model} + {t3_strat}",
                        )
                        st.image(shap_chart_png)

                        # Plain-English Explanation
                        explanation = shap_plain_english(top_df, p, t3_thr)
                        with st.expander("📖 Plain-English Explanation", expanded=True):
                            st.markdown(explanation)

                    except Exception as e:
                        st.warning(f"SHAP explanation unavailable: {e}")

                if i >= 4:
                    st.info("Displaying first 5 results. Upload fewer rows for full individual SHAP breakdown.")
                    break

        render_shap_disclaimer()

