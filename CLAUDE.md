# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup (use Python 3.11 or 3.12 — NOT 3.14, pyarrow wheels unavailable)
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run dashboard (opens http://localhost:8501)
streamlit run app.py

# Retrain model (regenerates upgrade_model.pkl + model_metadata.json)
python train_model.py
```

No test suite, linter, or build step exists.

## Critical Constraint: scikit-learn version lock

`scikit-learn==1.7.2` in `requirements.txt` is pinned to the version that trained
`upgrade_model.pkl`. Changing it may break model loading (joblib unpickle). If you
bump it, retrain with `train_model.py`.

## Architecture

Two-file system sharing one feature contract:

- `train_model.py` — trains a `GradientBoostingClassifier` inside an sklearn
  `Pipeline` (ColumnTransformer: StandardScaler on numeric + OneHotEncoder on
  categorical). Saves `upgrade_model.pkl` and writes `model_metadata.json`
  (holdout metrics + threshold tradeoffs).
- `app.py` — Streamlit + Plotly dashboard, 5 tabs: Overview, Who Upgrades (EDA),
  Drivers, Predict & Target, Model Health. Loads the pkl, scores customers, adds
  a business decision layer (segments, actions, priority, campaign value), and
  re-runs the holdout eval live for transparency.

### Feature contract — keep in sync across both files

`add_engineered_features()` is DUPLICATED in `train_model.py` and `app.py` and
MUST stay identical. It derives 5 features from raw columns:
`Spend_Efficiency`, `Complaint_Rate`, `Cuisine_Diversity`,
`Delivery_Pain_Index`, `Tip_to_Order_Ratio`. `RAW_FEATURES` and
`MODEL_FEATURES` lists also exist in both files and must match. Editing one
side without the other silently breaks scoring or training.

### Data

`food_app_customer_data.csv` — 2,000 rows. Target = `Membership_upgrade`.
1,800 labeled rows (used for training, dropna on target); 200 unlabeled rows
(scored for prioritization only, never trained on).

### Business decision layer (app.py only)

After the model produces `Upgrade_Probability`, app derives:
- `assign_segment()` → High intent / Persuadable / Service recovery /
  Experience risk / Low priority (rules combine probability + App_Rating +
  complaints + Spending_Score).
- `recommend_action()` → campaign action per segment (Platinum members excluded).
- `priority_score()` → ranking blend: probability 0.65 + value 0.20 +
  frequency 0.15 − service penalty.

`app.py` re-runs the holdout evaluation live in `evaluate_model()` (same
80/20 stratified split, random_state=42) for the Model Health tab.

## Model positioning — data ceiling, not a code problem

ROC-AUC caps at ~0.70 for EVERY model tried (GB, HistGB, LogReg). The data has
real signal in only ~2 behaviors: upgraders have LOWER Spending_Score (~39 vs 56)
and LOWER Purchase_Frequency (~6.4 vs 8.9) — both negatively correlated. All other
features are near-zero noise. So a fancier algorithm won't help; better data would.
Treat the model as a directional ranking aid built around threshold tradeoffs and
campaign lift, NOT autopilot. The counterintuitive finding (light/casual users
upgrade, not heavy spenders) is the headline analytics result.

## Config gotcha

`.streamlit/config.toml` must be saved WITHOUT a UTF-8 BOM, or Streamlit throws
`TomlDecodeError: Found invalid character in key name: '['`. Don't edit it with
tools that add a BOM. There is no root `config.toml` — Streamlit ignores it.
