"""
ui/plots.py — Matplotlib Visualization Engine
Pure charting functions decoupled from Streamlit rendering.
Can be tested independently without launching a web server.
"""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from ui.styles import DARK_BG, GRID_COL, TEXT_COL, FRAUD_COL, LEGIT_COL, GOLD_COL


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Convert a Matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf.read()


def style_fig(fig: plt.Figure, ax_or_axes):
    """Apply CreditGuard dark navy theme to figure and axis/axes."""
    fig.patch.set_facecolor(DARK_BG)
    axes = [ax_or_axes] if hasattr(ax_or_axes, "set_facecolor") else ax_or_axes
    for ax in axes:
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=8)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        if ax.get_title():
            ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)


def plot_metric_comparison(
    lb_df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    best_model: str,
    best_strategy: str,
) -> bytes:
    """Render horizontal/vertical bar chart comparing a metric across all 9 configurations."""
    lb_sorted = lb_df.sort_values(metric_col, ascending=False)
    labels = [
        f"{r['Model'].split()[0][:2]}-{r['Strategy'][:4]}"
        for _, r in lb_sorted.iterrows()
    ]
    colors = [
        GOLD_COL if (r["Model"] == best_model and r["Strategy"] == best_strategy)
        else LEGIT_COL
        for _, r in lb_sorted.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    bars = ax.bar(labels, lb_sorted[metric_col].values, color=colors, width=0.6)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} — All Configurations (gold = lowest cost)")
    max_val = lb_sorted[metric_col].max()
    ax.set_ylim(0, min(1.05, max_val * 1.12 if max_val > 0 else 1.0))

    for bar, val in zip(bars, lb_sorted[metric_col].values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color=TEXT_COL,
        )

    ax.tick_params(axis="x", rotation=35, labelsize=7.5)
    style_fig(fig, ax)
    return fig_to_bytes(fig)


def plot_confusion_matrix(cm_df: pd.DataFrame, title: str) -> bytes:
    """Render a 2x2 confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    ax.imshow(cm_df.values, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Legit", "Predicted Fraud"], color=TEXT_COL, fontsize=9)
    ax.set_yticklabels(["Actual Legit", "Actual Fraud"], color=TEXT_COL, fontsize=9)

    max_val = cm_df.values.max()
    for i in range(2):
        for j in range(2):
            val = cm_df.values[i, j]
            ax.text(
                j,
                i,
                f"{val:,}",
                ha="center",
                va="center",
                color="white" if val > max_val / 2 else TEXT_COL,
                fontsize=12,
                fontweight="bold",
            )

    ax.set_title(title, color=TEXT_COL, fontsize=9)
    style_fig(fig, ax)
    ax.grid(False)
    return fig_to_bytes(fig)


def plot_cost_vs_threshold(
    sweep_df: pd.DataFrame,
    opt_thr: float,
    default_thr: float,
    title: str,
) -> bytes:
    """Render cost versus decision threshold curve."""
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(sweep_df["threshold"], sweep_df["cost"], color=LEGIT_COL, linewidth=2, label="Total cost")
    ax.axvline(opt_thr, color=GOLD_COL, linewidth=1.8, linestyle="--", label=f"Optimal threshold ({opt_thr:.2f})")
    ax.axvline(default_thr, color="#888888", linewidth=1, linestyle=":", label=f"Default ({default_thr:.2f})")

    min_cost_val = sweep_df.loc[sweep_df["threshold"].sub(opt_thr).abs().idxmin(), "cost"]
    ax.scatter([opt_thr], [min_cost_val], color=GOLD_COL, s=80, zorder=5)

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title(title)
    ax.legend(fontsize=8, facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    style_fig(fig, ax)
    return fig_to_bytes(fig)


def plot_error_tradeoff(sweep_df: pd.DataFrame, opt_thr: float) -> bytes:
    """Render Missed Fraud (FN) vs False Alarms (FP) count curves."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(sweep_df["threshold"], sweep_df["FN"], color=FRAUD_COL, linewidth=1.8, label="Missed Fraud (FN)")
    ax.plot(sweep_df["threshold"], sweep_df["FP"], color=LEGIT_COL, linewidth=1.8, label="False Alarms (FP)")
    ax.axvline(opt_thr, color=GOLD_COL, linewidth=1.5, linestyle="--", label=f"Optimal ({opt_thr:.2f})")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Count")
    ax.set_title("FN & FP vs Threshold (Validation)")
    ax.legend(fontsize=8, facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)
    style_fig(fig, ax)
    return fig_to_bytes(fig)


def plot_shap_contributions(top_df: pd.DataFrame, x_sample_row: pd.Series, title: str) -> bytes:
    """Render horizontal bar chart for top-8 SHAP feature contributions."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bar_colors = [
        FRAUD_COL if d == "Toward Fraud" else LEGIT_COL
        for d in top_df["direction"]
    ]

    labels = [
        f"{r['feature']} = {x_sample_row[r['feature']]:.3f}"
        for _, r in top_df.iterrows()
    ]

    ax.barh(labels, top_df["shap_value"].values, color=bar_colors, height=0.6)
    ax.axvline(0, color=TEXT_COL, linewidth=0.8)
    ax.set_xlabel("SHAP Contribution (positive → fraud)")
    ax.set_title(title, color=TEXT_COL)

    red_patch = mpatches.Patch(color=FRAUD_COL, label="Toward Fraud")
    blue_patch = mpatches.Patch(color=LEGIT_COL, label="Toward Legitimate")
    ax.legend(handles=[red_patch, blue_patch], fontsize=7, facecolor=DARK_BG, edgecolor=GRID_COL, labelcolor=TEXT_COL)

    style_fig(fig, ax)
    return fig_to_bytes(fig)

