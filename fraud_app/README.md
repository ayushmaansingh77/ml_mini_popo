# CreditGuard — Explainable AI for Credit Card Fraud Detection

> **Academic ML project** · BTech Computer Science · End-to-end fraud detection platform  
> Built with Scikit-learn · XGBoost · imbalanced-learn · SHAP · Streamlit

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Research Implementation](#existing-research)
4. [Models & Strategies](#models--strategies)
5. [Architecture](#architecture)
6. [Training Methodology](#training-methodology)
7. [Installation](#installation)
8. [Running the Application](#running-the-application)
9. [Application Features](#application-features)
10. [Cost-Sensitive Mode](#cost-sensitive-mode)
11. [SHAP Explainability](#shap-explainability)
12. [Limitations](#limitations)
13. [Future Improvements](#future-improvements)

---

## Problem Statement

Credit card fraud detection is an **extreme class-imbalance problem**: in this dataset only 0.17% of transactions are fraudulent. Standard accuracy is useless — a model predicting "legitimate" for every transaction achieves 99.83% accuracy while detecting zero fraud.

Three compounding challenges:

| Challenge | Impact |
|---|---|
| Class imbalance (1:577 ratio) | Standard models ignore the minority class |
| Asymmetric costs | Missing a fraud costs far more than a false alarm |
| Threshold choice | Default threshold 0.5 is arbitrary; optimal threshold depends on business costs |

CreditGuard addresses all three: it compares imbalance strategies, models business costs directly, and uses SHAP to explain individual predictions.

---

## Dataset

**Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — ULB Machine Learning Group

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.1727%) |
| Legitimate | 284,315 |
| Features | 30 (Time, Amount, V1–V28) |
| Missing values | 0 |
| Time span | 48 hours (September 2013) |

**Features:**
- `Time` — Seconds since the first transaction in the dataset
- `Amount` — Transaction amount in euros
- `V1–V28` — Principal Components of the original (confidential) features; privacy-preserving PCA transformation applied by the bank

**Preprocessing:** Only `Time` and `Amount` are scaled (StandardScaler). V1–V28 are already PCA-transformed and on comparable scales.

---

## Existing Research

This application is built on two research notebooks:

| Notebook | Contents |
|---|---|
| `fraud.ipynb` | Baseline implementation: 9 model/strategy combinations, evaluation at threshold 0.5, PR curves, ROC curves, feature importance |
| `fraud_extended.ipynb` | Extended research: proper sklearn Pipelines, 5-fold CV, additional resampling strategies, F1-based threshold tuning, RandomizedSearchCV for XGBoost, SHAP summary plots |

Both notebooks are preserved unmodified. The application synthesises the best elements of each.

---

## Models & Strategies

### Models

| Model | Type | Key parameter |
|---|---|---|
| **Logistic Regression** | Linear | `max_iter=1000, class_weight="balanced"` (weighted) |
| **Random Forest** | Bagging ensemble | `n_estimators=100, max_depth=12, class_weight="balanced_subsample"` (weighted) |
| **XGBoost** | Gradient boosting | `n_estimators=200, max_depth=6, lr=0.1, scale_pos_weight≈577` (weighted) |

### Strategies

| Strategy | Description |
|---|---|
| **Baseline** | Train on original imbalanced data, no adjustments |
| **Class weighting** | Modify loss function to penalise minority class misclassification more heavily |
| **SMOTE** | Synthetic Minority Oversampling — interpolate new fraud samples in feature space |

**9 total configurations** (3 models × 3 strategies).

---

## Architecture

```
ml_minipopo/
├── fraud.ipynb               ← Original research (preserved)
├── fraud_extended.ipynb      ← Extended research (preserved)
├── creditcard.csv            ← Dataset
│
└── fraud_app/
    ├── config.py             ← All constants, paths, hyperparameters
    ├── train.py              ← Offline training script (run once)
    ├── core.py               ← Reusable prediction, metrics, cost, SHAP functions
    ├── app.py                ← Streamlit dashboard orchestrator (~119 lines)
    ├── requirements.txt
    ├── README.md
    ├── viva.md               ← Viva & interview preparation guide
    ├── context.md            ← Implementation single-source-of-truth
    │
    ├── ui/                   ← Decoupled UI Design System & Plotting
    │   ├── __init__.py
    │   ├── styles.py         ← CSS, palette, typography
    │   ├── components.py     ← Reusable HTML widgets (cards, badges, banners)
    │   └── plots.py          ← Matplotlib plotting engine (unit-testable)
    │
    ├── views/                ← Independent Tab Controllers
    │   ├── __init__.py
    │   ├── tab_leaderboard.py    ← Tab 1 view & interactions
    │   ├── tab_cost_sensitive.py ← Tab 2 view & threshold curves
    │   └── tab_prediction.py     ← Tab 3 view & SHAP attribution
    │
    └── artifacts/
        ├── models/           ← 9 fitted pipelines (.pkl)
        ├── test_data.parquet ← Held-out test features + labels
        ├── test_scores.parquet  ← Test-set probabilities (9 columns)
        ├── validation_scores.parquet ← Validation probabilities (9 columns)
        ├── leaderboard.csv   ← Metrics at threshold 0.5
        ├── shap_background.parquet  ← Pre-transformed SHAP background rows
        └── metadata.json     ← Training timestamp, library versions, splits
```

**Key design decision:** Training is completely separated from the application. `train.py` runs once and saves all artifacts. `app.py` only loads frozen artifacts — no training happens at runtime.

---

## Training Methodology

### Data Splits

```
Full dataset (284,807 rows)
  └── Stratified 80/20 split (seed=42)
        ├── Test set  (20% = ~56,962 rows)   ← NEVER used for training or tuning
        └── Train set (80% = ~227,845 rows)
              └── Stratified 75/25 split (seed=42)
                    ├── Val set  (25% of train = ~56,961 rows) ← threshold selection only
                    └── Fit set  (75% of train = ~170,884 rows) ← threshold-tuning model
```

The **leaderboard models** train on the full 80% training set.  
The **threshold-tuning models** train on the 60% fit subset, with the validation set kept truly unseen.

### Pipeline

```python
Pipeline([
    ("prep",  ColumnTransformer([("sc", StandardScaler(), ["Time","Amount"])],
                                 remainder="passthrough")),
    ("smote", SMOTE(random_state=42)),   # only for SMOTE strategy
    ("clf",   <model>)
])
```

### Data Leakage Prevention

| Rule | Implementation |
|---|---|
| Scaler fit on training data only | `ColumnTransformer` fitted inside `Pipeline.fit()` |
| SMOTE applied to training data only | SMOTE step inside pipeline — never reaches test/val |
| `scale_pos_weight` from `y_train` only | Computed after split, not from full dataset |
| Threshold selected on validation data only | `best_threshold()` accepts validation probabilities |
| Test set never used for fitting or selection | Test probabilities only used for final evaluation |

---

## Installation

```bash
# Clone or navigate to the project directory
cd c:\Users\AYUSHMAAN\Desktop\ml_minipopo\fraud_app

# Install dependencies
pip install -r requirements.txt
```

**Requires:** Python 3.10+, creditcard.csv in the parent directory.

---

## Running the Application

### Step 1 — Train models (one time only)

```bash
python train.py
```

Expected runtime: 5–20 minutes depending on hardware (Random Forest + SMOTE is the slowest configuration).

Artifacts are saved to `fraud_app/artifacts/`.

### Step 2 — Launch the application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Application Features

### Tab 1 — Model Leaderboard

- Sortable table of all 9 model/strategy configurations
- Metrics: Precision, Recall, F1, PR-AUC, ROC-AUC, Accuracy, Training time, Total cost
- **Live cost recalculation** — enter business cost values to see which configuration minimises total financial cost
- Best-configuration banner (highlighted)
- Metric comparison bar chart (any metric)
- Confusion matrix for any selected configuration

### Tab 2 — Cost-Sensitive Mode

- Select model, strategy, and cost values
- System finds the **optimal decision threshold** that minimises `FN × cost_fn + FP × cost_fp` on the validation set
- Cost vs threshold chart with minimum-cost marker
- FN & FP vs threshold chart (shows the trade-off)
- Confusion matrix at the optimal threshold
- Performance comparison: default threshold 0.5 vs optimal threshold

### Tab 3 — Transaction Prediction + SHAP

- **Input options:** Manual entry, CSV upload, or test-set sample (random / known fraud / known legitimate)
- Fraud probability and binary classification (FRAUD / LEGITIMATE)
- Adjustable decision threshold
- SHAP horizontal bar chart (top 8 features, red = toward fraud, blue = toward legitimate)
- Plain-English explanation of the prediction

---

## Cost-Sensitive Mode

### Cost Function

```
Total Cost = (FN × cost_of_missed_fraud) + (FP × cost_of_false_alarm)
```

### Threshold Optimization Algorithm

1. Train model on fit subset (60% of data)
2. Generate predicted probabilities on validation set (20% of data, never seen by model)
3. Sweep thresholds from 0.01 to 0.99 in steps of 0.01
4. Compute total cost at each threshold
5. Select threshold with **minimum total cost**
6. Report performance at selected threshold on the held-out test set (20% of data)

### Trade-off

> Lower thresholds catch more fraud (higher Recall) but increase false alarms (lower Precision).  
> Higher thresholds reduce false alarms but miss more fraud.  
> The optimal threshold balances both according to actual business costs.

> **Disclaimer:** The optimal threshold is selected using validation data. Final performance is evaluated on held-out test data. Real-world deployment requires representative validation data and periodic threshold recalibration.

---

## SHAP Explainability

### What is SHAP?

SHAP (SHapley Additive exPlanations) assigns each feature a contribution value for a specific prediction. The contributions sum to the difference between the prediction and the model's average prediction.

### Implementation

| Model | Explainer | Notes |
|---|---|---|
| Random Forest | `shap.TreeExplainer` | Exact tree-traversal algorithm |
| XGBoost | `shap.TreeExplainer` | Exact, very fast |
| Logistic Regression | `shap.LinearExplainer` | Uses model coefficients + feature covariance |

The pipeline's `prep` step is applied to the input before computing SHAP values, ensuring feature ordering matches what the model received during training.

### Display

- Horizontal bar chart: top 8 features by absolute SHAP contribution
- 🔴 Red bars → push prediction toward fraud
- 🔵 Blue bars → push prediction toward legitimate
- Each bar labelled with feature name and actual value

### Important Limitation

> SHAP explains model behaviour on historical data. It does not establish causation. V1–V28 are anonymous PCA components — their SHAP values indicate model-level influence, not real-world meaning.

---

## Limitations

1. **Static model:** Trained on 2013 European transaction data. Fraud patterns change over time (concept drift).
2. **Anonymous features:** V1–V28 are PCA components — SHAP explanations are model-level, not humanly interpretable in business terms.
3. **Fixed costs:** The cost function uses user-supplied constants. Real fraud costs are dynamic (transaction-specific).
4. **No uncertainty quantification:** Probabilities are point estimates. Borderline cases lack confidence intervals.
5. **No real-time scoring:** This is a research/demo platform, not a production API.
6. **Threshold stability:** The optimal threshold can vary with dataset size and fraud rate. Recalibration is needed periodically.

---

## Future Improvements

- **Tuned XGBoost** configuration using the best RandomizedSearchCV parameters found in `fraud_extended.ipynb` (`n_estimators=300, max_depth=8, lr=0.05, subsample=0.7`)
- **Calibrated probabilities** using `CalibratedClassifierCV` for better-calibrated uncertainty estimates
- **LightGBM** as a fourth model for comparison
- **Concept drift monitoring** dashboard
- **Batch CSV prediction** with downloadable results
- **Model versioning** with MLflow
- **REST API** (FastAPI) for real-time transaction scoring
- **Automated retraining** pipeline on new labeled data

