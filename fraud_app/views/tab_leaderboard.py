"""
views/tab_leaderboard.py — Tab 1: Model Leaderboard
Renders 9-configuration leaderboard, cost-sensitive re-ranking,
metric comparison bar chart, and confusion matrix.
"""

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    MODELS, STRATEGIES, DEFAULT_THRESHOLD, DEFAULT_COST_FN, DEFAULT_COST_FP,
)
from core import (
    evaluate, confusion_matrix_df, compute_leaderboard_costs, score_col_name,
)
from ui.components import render_best_model_banner, render_metric_card
from ui.plots import plot_metric_comparison, plot_confusion_matrix


def render_leaderboard_tab(
    leaderboard_df: pd.DataFrame,
    test_scores_df: pd.DataFrame,
    y_test: pd.Series,
    metadata: dict,
):
    """Render Tab 1: Model Leaderboard."""
    st.markdown("### 📊 Model Leaderboard")
    st.caption(
        f"Trained on {metadata.get('train_rows', 227845):,} rows · "
        f"Evaluated on {metadata.get('test_rows', 56962):,} held-out test rows · "
        f"Seed {metadata.get('seed', 42)}"
    )

    # --- Cost Inputs ---
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        lb_cost_fn = st.number_input(
            "Cost per missed fraud ($)",
            min_value=0,
            value=DEFAULT_COST_FN,
            step=50,
            key="lb_fn",
            help="Financial loss when a fraudulent transaction is not detected (False Negative).",
        )
    with c2:
        lb_cost_fp = st.number_input(
            "Cost per false alarm ($)",
            min_value=0,
            value=DEFAULT_COST_FP,
            step=5,
            key="lb_fp",
            help="Investigation cost when a legitimate transaction is flagged (False Positive).",
        )

    st.divider()

    # --- Build display leaderboard with live costs ---
    lb = leaderboard_df.copy()
    cost_series = compute_leaderboard_costs(
        test_scores_df, y_test, lb_cost_fn, lb_cost_fp
    )
    lb["Total_Cost"] = lb.apply(
        lambda r: cost_series.get(score_col_name(r["Model"], r["Strategy"]), np.nan),
        axis=1,
    ).round(0).astype(int)

    lb_display = lb[[
        "Model", "Strategy", "Precision", "Recall", "F1",
        "PR_AUC", "ROC_AUC", "Accuracy", "Train_time_s", "Total_Cost",
    ]].copy()
    lb_display.rename(columns={
        "PR_AUC": "PR-AUC",
        "ROC_AUC": "ROC-AUC",
        "Train_time_s": "Train(s)",
        "Total_Cost": "Cost($)",
    }, inplace=True)

    # Best model by lowest cost
    best_idx = lb["Total_Cost"].idxmin()
    best_row = lb.iloc[best_idx]

    render_best_model_banner(
        model_name=best_row["Model"],
        strategy_name=best_row["Strategy"],
        cost=best_row["Total_Cost"],
        f1=best_row["F1"],
        pr_auc=best_row["PR_AUC"],
    )

    # --- Sort control & Table ---
    sort_col = st.selectbox(
        "Sort leaderboard by",
        ["Cost($)", "F1", "PR-AUC", "ROC-AUC", "Recall", "Precision"],
        index=0,
        key="lb_sort",
    )
    lb_display_sorted = lb_display.sort_values(
        sort_col,
        ascending=(sort_col == "Cost($)"),
    ).reset_index(drop=True)

    def highlight_best(row):
        if row["Model"] == best_row["Model"] and row["Strategy"] == best_row["Strategy"]:
            return ["background-color: #0d2b1e; color: #4caf8a; font-weight: bold"] * len(row)
        return [""] * len(row)

    styled_df = lb_display_sorted.style.apply(highlight_best, axis=1).format({
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1": "{:.3f}",
        "PR-AUC": "{:.3f}",
        "ROC-AUC": "{:.3f}",
        "Accuracy": "{:.4f}",
        "Train(s)": "{:.1f}",
        "Cost($)": "{:,.0f}",
    })

    st.dataframe(styled_df, width="stretch", height=360)

    st.divider()

    # --- Metric Comparison Bar Chart ---
    st.markdown("#### Metric Comparison")
    metric_choice = st.selectbox(
        "Metric to visualise",
        ["F1", "PR-AUC", "ROC-AUC", "Recall", "Precision", "Accuracy"],
        key="lb_metric_chart",
    )
    metric_map = {"PR-AUC": "PR_AUC", "ROC-AUC": "ROC_AUC"}
    raw_col = metric_map.get(metric_choice, metric_choice)

    chart_png = plot_metric_comparison(
        lb_df=lb,
        metric_col=raw_col,
        metric_label=metric_choice,
        best_model=best_row["Model"],
        best_strategy=best_row["Strategy"],
    )
    st.image(chart_png)

    st.divider()

    # --- Confusion Matrix for Selected Model ---
    st.markdown("#### Confusion Matrix")
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        cm_model = st.selectbox("Model", MODELS, key="lb_cm_model")
    with cm_col2:
        cm_strat = st.selectbox("Strategy", STRATEGIES, key="lb_cm_strat")

    cm_proba = test_scores_df[score_col_name(cm_model, cm_strat)].values
    cm_df = confusion_matrix_df(y_test, cm_proba, DEFAULT_THRESHOLD)
    cm_png = plot_confusion_matrix(
        cm_df=cm_df,
        title=f"{cm_model} + {cm_strat}\n(threshold = {DEFAULT_THRESHOLD})",
    )
    st.image(cm_png)

    # Summary Cards
    sel_metrics = evaluate(y_test, cm_proba)
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    for col_ui, label, key in [
        (mc1, "F1", "F1"),
        (mc2, "Precision", "Precision"),
        (mc3, "Recall", "Recall"),
        (mc4, "PR-AUC", "PR_AUC"),
        (mc5, "ROC-AUC", "ROC_AUC"),
    ]:
        render_metric_card(col_ui, label, sel_metrics[key])

