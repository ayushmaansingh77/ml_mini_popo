"""
views/tab_cost_sensitive.py — Tab 2: Cost-Sensitive Fraud Detection
Finds the minimum financial cost decision threshold on validation data,
then verifies performance on held-out test data.
"""

import pandas as pd
import streamlit as st

from config import (
    MODELS, STRATEGIES, DEFAULT_THRESHOLD, DEFAULT_COST_FN, DEFAULT_COST_FP,
)
from core import (
    evaluate, confusion_matrix_df, total_cost, cost_sweep, best_threshold, score_col_name,
)
from ui.components import render_cost_tradeoff_disclaimer
from ui.plots import (
    plot_cost_vs_threshold, plot_error_tradeoff, plot_confusion_matrix,
)


def render_cost_tab(
    val_scores_df: pd.DataFrame,
    test_scores_df: pd.DataFrame,
    y_val: pd.Series,
    y_test: pd.Series,
):
    """Render Tab 2: Cost-Sensitive Mode."""
    st.markdown("### 💰 Cost-Sensitive Fraud Detection")
    st.markdown(
        "Select a model and cost structure. The system finds the decision threshold "
        "that minimises total financial cost on the **validation set**, then reports "
        "performance on the **held-out test set**."
    )

    # --- Inputs ---
    t2c1, t2c2, t2c3, t2c4 = st.columns(4)
    with t2c1:
        t2_model = st.selectbox("Model", MODELS, key="t2_model")
    with t2c2:
        t2_strat = st.selectbox("Strategy", STRATEGIES, key="t2_strat")
    with t2c3:
        t2_fn = st.number_input(
            "Cost: missed fraud ($)",
            min_value=1,
            value=DEFAULT_COST_FN,
            step=50,
            key="t2_fn",
        )
    with t2c4:
        t2_fp = st.number_input(
            "Cost: false alarm ($)",
            min_value=1,
            value=DEFAULT_COST_FP,
            step=5,
            key="t2_fp",
        )

    # Pull pre-computed probabilities
    col_key = score_col_name(t2_model, t2_strat)
    proba_val = val_scores_df[col_key].values
    proba_test = test_scores_df[col_key].values

    # Threshold optimization performed on VALIDATION data only (zero test leakage)
    opt_thr = best_threshold(y_val, proba_val, t2_fn, t2_fp)

    # Evaluate at both thresholds on held-out TEST data
    metrics_def = evaluate(y_test, proba_test, DEFAULT_THRESHOLD)
    metrics_opt = evaluate(y_test, proba_test, opt_thr)
    cost_def = total_cost(y_test, proba_test, DEFAULT_THRESHOLD, t2_fn, t2_fp)
    cost_opt = total_cost(y_test, proba_test, opt_thr, t2_fn, t2_fp)

    st.divider()

    # --- Threshold Comparison Table ---
    st.markdown("#### Threshold Comparison (Test Set Performance)")
    hdr, col_def, col_opt = st.columns([1.2, 1, 1])
    hdr.markdown("**Metric**")
    col_def.markdown(f"**Default (0.50)**")
    col_opt.markdown(f"**Optimal ({opt_thr:.2f})**")

    for metric in ["Precision", "Recall", "F1", "FP", "FN"]:
        v_def = metrics_def[metric]
        v_opt = metrics_opt[metric]
        fmt = "{:.3f}" if metric not in ("FP", "FN") else "{:,}"
        hdr.markdown(f"`{metric}`")
        col_def.markdown(fmt.format(v_def))
        better = (v_opt > v_def) if metric not in ("FP", "FN") else (v_opt < v_def)
        col_opt.markdown(f"{'🟢' if better else '🔴'} **{fmt.format(v_opt)}**")

    # Total financial cost row
    hdr.markdown("**Total Cost ($)**")
    col_def.markdown(f"${cost_def:,.0f}")
    col_opt.markdown(f"{'🟢' if cost_opt < cost_def else '🔴'} **${cost_opt:,.0f}**")

    st.divider()

    # --- Cost vs Threshold Chart ---
    st.markdown("#### Cost vs Decision Threshold")
    st.caption("Computed on validation data. Optimal threshold marked in gold.")

    sweep_df = cost_sweep(y_val, proba_val, t2_fn, t2_fp)
    cost_chart_png = plot_cost_vs_threshold(
        sweep_df=sweep_df,
        opt_thr=opt_thr,
        default_thr=DEFAULT_THRESHOLD,
        title=f"{t2_model} + {t2_strat} — Cost vs Threshold (Validation)",
    )
    st.image(cost_chart_png)

    # FN & FP Tradeoff Curve
    error_chart_png = plot_error_tradeoff(sweep_df, opt_thr)
    st.image(error_chart_png)

    st.divider()

    # --- Confusion Matrix at Optimal Threshold ---
    st.markdown(f"#### Confusion Matrix at Optimal Threshold ({opt_thr:.2f}) — Test Set")
    cm_t2 = confusion_matrix_df(y_test, proba_test, opt_thr)
    cm_png = plot_confusion_matrix(
        cm_df=cm_t2,
        title=f"{t2_model} + {t2_strat} (threshold={opt_thr:.2f})",
    )
    st.image(cm_png)

    # Regulatory & Methodological Disclaimer
    render_cost_tradeoff_disclaimer()

