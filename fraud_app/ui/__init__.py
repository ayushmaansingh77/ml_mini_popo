"""
ui package for CreditGuard
Provides design system, reusable UI components, and plotting engines.
"""
from ui.styles import apply_custom_css, DARK_BG, GRID_COL, TEXT_COL, FRAUD_COL, LEGIT_COL, GOLD_COL
from ui.components import (
    render_header,
    render_footer,
    render_metric_card,
    render_best_model_banner,
    render_prediction_badge,
    render_cost_tradeoff_disclaimer,
    render_shap_disclaimer,
)
from ui.plots import (
    fig_to_bytes,
    plot_metric_comparison,
    plot_confusion_matrix,
    plot_cost_vs_threshold,
    plot_error_tradeoff,
    plot_shap_contributions,
)

