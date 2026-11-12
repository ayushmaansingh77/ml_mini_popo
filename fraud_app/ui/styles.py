"""
ui/styles.py — Design System, Themes, and CSS Injection
Provides cohesive modern dark navy styling for CreditGuard dashboard.
"""

import streamlit as st

# Color Palette Constants
DARK_BG = "#0d1b2a"
PANEL_BG = "#101e2e"
GRID_COL = "#1e3a5f"
TEXT_COL = "#c8d8e8"
MUTED_COL = "#8aabb8"
LABEL_COL = "#6b8fa8"

FRAUD_COL = "#e05252"
LEGIT_COL = "#3a9fd8"
SUCCESS_COL = "#4caf8a"
GOLD_COL = "#f0a500"


def apply_custom_css():
    """Inject global CSS into the Streamlit application."""
    st.markdown(f"""
    <style>
    /* ---- Global Typography ---- */
    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    /* ---- Header Container ---- */
    .cg-header {{
        background: linear-gradient(135deg, {DARK_BG} 0%, #1b2838 60%, #152638 100%);
        padding: 2rem 2.5rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid {GRID_COL};
    }}
    .cg-title {{
        font-size: 2.3rem;
        font-weight: 800;
        color: #e8f4fd;
        margin: 0;
        letter-spacing: -0.5px;
    }}
    .cg-title span {{ color: {LEGIT_COL}; }}
    .cg-subtitle {{
        font-size: 0.95rem;
        color: {MUTED_COL};
        margin-top: 0.35rem;
    }}

    /* ---- Metric Card ---- */
    .metric-card {{
        background: {PANEL_BG};
        border: 1px solid {GRID_COL};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }}
    .metric-card .label {{
        font-size: 0.72rem;
        color: {LABEL_COL};
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }}
    .metric-card .value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: #e8f4fd;
    }}
    .metric-card .value.green {{ color: {SUCCESS_COL}; }}
    .metric-card .value.amber {{ color: {GOLD_COL}; }}
    .metric-card .value.red   {{ color: {FRAUD_COL}; }}

    /* ---- Banners & Status Badges ---- */
    .best-banner {{
        background: linear-gradient(90deg, #0d2b1e, #0d3326);
        border: 1px solid #1a7a4a;
        border-left: 4px solid {SUCCESS_COL};
        border-radius: 8px;
        padding: 0.9rem 1.3rem;
        margin: 1rem 0;
    }}
    .fraud-badge {{
        background: #3d1010;
        border: 1px solid #c0392b;
        color: {FRAUD_COL};
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }}
    .legit-badge {{
        background: #0d2b1e;
        border: 1px solid #1a7a4a;
        color: {SUCCESS_COL};
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }}

    /* ---- Disclaimers ---- */
    .disclaimer {{
        background: {DARK_BG};
        border: 1px solid {GRID_COL};
        border-left: 3px solid {LEGIT_COL};
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        font-size: 0.82rem;
        color: {MUTED_COL};
        margin-top: 1.2rem;
        line-height: 1.5;
    }}
    </style>
    """, unsafe_allow_html=True)

