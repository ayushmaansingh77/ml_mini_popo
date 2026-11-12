"""
app.py — CreditGuard Streamlit Application
Orchestrator entry point.
Run: streamlit run app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure fraud_app/ is resolvable in sys.path
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))

from config import (
    LEADERBOARD_PATH, TEST_SCORES_PATH, VAL_SCORES_PATH,
    TEST_DATA_PATH, SHAP_BG_PATH, METADATA_PATH,
    MODELS, STRATEGIES, model_path,
)
from ui.styles import apply_custom_css
from ui.components import render_header, render_footer
from views import (
    render_leaderboard_tab,
    render_cost_tab,
    render_prediction_tab,
)

# ---------------------------------------------------------------------------
# 1. Streamlit Page Configuration & Design System
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_custom_css()
render_header()

# ---------------------------------------------------------------------------
# 2. Cached Artifact Loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_leaderboard() -> pd.DataFrame:
    return pd.read_csv(LEADERBOARD_PATH)

@st.cache_data
def load_test_scores() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(TEST_SCORES_PATH)
    return df.drop(columns=["_label"]), df["_label"]

@st.cache_data
def load_val_scores() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(VAL_SCORES_PATH)
    return df.drop(columns=["_label"]), df["_label"]

@st.cache_data
def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(TEST_DATA_PATH)
    return df.drop(columns=["_label"]), df["_label"]

@st.cache_data
def load_shap_background():
    df = pd.read_parquet(SHAP_BG_PATH)
    return df.drop(columns=["_label"]).values

@st.cache_data
def load_metadata() -> dict:
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def verify_artifacts() -> bool:
    required = [LEADERBOARD_PATH, TEST_SCORES_PATH, VAL_SCORES_PATH,
                TEST_DATA_PATH, SHAP_BG_PATH, METADATA_PATH]
    missing = [str(p) for p in required if not p.exists()]
    for model in MODELS:
        for strategy in STRATEGIES:
            p = model_path(model, strategy)
            if not p.exists():
                missing.append(str(p))

    if missing:
        st.error(
            "**Missing Artifacts.** Run `python train.py` first to generate models and scores:\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
        )
        return False
    return True

if not verify_artifacts():
    st.stop()

# Load frozen artifacts
leaderboard_df = load_leaderboard()
test_scores_df, y_test = load_test_scores()
val_scores_df, y_val = load_val_scores()
X_test_df, _ = load_test_data()
shap_background = load_shap_background()
metadata = load_metadata()

# ---------------------------------------------------------------------------
# 3. Render Modular Views
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Model Leaderboard",
    "💰 Cost-Sensitive Mode",
    "🔍 Transaction Prediction + SHAP",
])

with tab1:
    render_leaderboard_tab(
        leaderboard_df=leaderboard_df,
        test_scores_df=test_scores_df,
        y_test=y_test,
        metadata=metadata,
    )

with tab2:
    render_cost_tab(
        val_scores_df=val_scores_df,
        test_scores_df=test_scores_df,
        y_val=y_val,
        y_test=y_test,
    )

with tab3:
    render_prediction_tab(
        X_test_df=X_test_df,
        y_test=y_test,
        shap_background=shap_background,
    )

# ---------------------------------------------------------------------------
# 4. Footer
# ---------------------------------------------------------------------------
render_footer(metadata=metadata)
