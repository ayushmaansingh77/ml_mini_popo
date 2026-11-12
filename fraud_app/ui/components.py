"""
ui/components.py — Reusable Visual HTML Components
Decouples raw HTML string creation from application view logic.
"""

import streamlit as st


def render_header():
    """Render main top banner header."""
    st.markdown("""
    <div class="cg-header">
      <div class="cg-title">🛡️ <span>Credit</span>Guard</div>
      <div class="cg-subtitle">Explainable AI-Powered Credit Card Fraud Detection</div>
    </div>
    """, unsafe_allow_html=True)


def render_footer(metadata: dict):
    """Render footer with model training metadata."""
    st.divider()
    trained_at = metadata.get("trained_at", "unknown")[:19].replace("T", " ")
    total_rows = metadata.get("total_rows", 284807)
    fraud_rows = metadata.get("fraud_rows", 492)
    fraud_rate = (100 * fraud_rows / total_rows) if total_rows else 0.1727

    st.caption(
        f"CreditGuard · Models trained {trained_at} · "
        f"Dataset: {total_rows:,} transactions · "
        f"Fraud rate: {fraud_rate:.4f}%"
    )


def render_metric_card(col, label: str, value: float | str, fmt: str = "{:.3f}"):
    """Render a styled metric card in a Streamlit column."""
    if isinstance(value, (int, float)):
        num = float(value)
        clr = "green" if num >= 0.8 else ("amber" if num >= 0.5 else "red")
        formatted_val = fmt.format(num)
    else:
        clr = ""
        formatted_val = str(value)

    col.markdown(f"""
    <div class="metric-card">
      <div class="label">{label}</div>
      <div class="value {clr}">{formatted_val}</div>
    </div>
    """, unsafe_allow_html=True)


def render_best_model_banner(model_name: str, strategy_name: str, cost: float, f1: float, pr_auc: float):
    """Render green highlight banner indicating current lowest cost model."""
    st.markdown(f"""
    <div class="best-banner">
      ✅ <strong>Best for current cost settings:</strong>
      &nbsp;<strong>{model_name}</strong> + <strong>{strategy_name}</strong>
      &nbsp;·&nbsp; Cost = <strong>${int(cost):,}</strong>
      &nbsp;·&nbsp; F1 = <strong>{f1:.3f}</strong>
      &nbsp;·&nbsp; PR-AUC = <strong>{pr_auc:.3f}</strong>
    </div>
    """, unsafe_allow_html=True)


def render_prediction_badge(is_fraud: bool):
    """Render visual status badge for fraud vs legitimate classification."""
    if is_fraud:
        st.markdown('<div class="fraud-badge">⚠️ FRAUD DETECTED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="legit-badge">✅ LEGITIMATE TRANSACTION</div>', unsafe_allow_html=True)


def render_cost_tradeoff_disclaimer():
    """Render disclaimer explaining validation-derived threshold optimization."""
    st.markdown("""
    <div class="disclaimer">
    ⚖️ <strong>Trade-off:</strong>
    Lower thresholds catch more fraudulent transactions but increase false alarms.
    Higher thresholds reduce false alarms but may miss more fraudulent transactions.
    The optimal threshold minimises the weighted combination of both error types.
    <br><br>
    📋 <em>The optimal threshold is selected using validation data. Final performance is
    evaluated on held-out test data. Real-world deployment requires representative
    validation data and periodic threshold recalibration.</em>
    </div>
    """, unsafe_allow_html=True)


def render_shap_disclaimer():
    """Render disclaimer clarifying SHAP interpretability boundaries."""
    st.markdown("""
    <div class="disclaimer">
    🔬 <strong>Disclaimer:</strong>
    SHAP values explain what features drove this model's prediction.
    They describe learned correlations in historical data — not causation.
    V1–V28 are anonymous PCA components; their SHAP values indicate model-level
    influence, not interpretable real-world attributes. This tool is for
    research and educational purposes only.
    </div>
    """, unsafe_allow_html=True)

