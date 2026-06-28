# Premium Membership Upgrade Growth Dashboard

A Streamlit business analytics app for identifying food-delivery customers who are most likely to upgrade to a premium membership and turning those predictions into campaign actions.

The solution helps growth and marketing teams prioritize customers, recommend the right upgrade action, and estimate the expected value of a premium membership campaign.

## Python Version Requirement

Use Python 3.11 or 3.12 for this project. Python 3.14 is not recommended because the Streamlit dependency stack includes `pyarrow`, and compatible wheels may not be available for Python 3.14 yet.

On Windows, prefer `py -3.12` in the setup commands so the virtual environment is created with Python 3.12 instead of your system default Python.

## Business Problem

Food-delivery platforms often know which customers order frequently, spend well, use discounts, or show signs of service frustration, but they still need a practical way to decide:

- Which customers should be targeted for a premium upgrade campaign?
- Which users need service recovery before an upgrade pitch?
- How many customers should the team contact?
- What is the expected value under different campaign assumptions?

This app adds a business decision layer on top of a machine-learning model trained on historical organic upgrade behavior.

## What The App Does

- Scores customers using the saved `upgrade_model.pkl` pipeline and ranks them for outreach.
- Builds engineered behavior features used during model training.
- Ranks customers by upgrade probability, spending behavior, purchase frequency, and service risk.
- Segments customers into campaign-ready groups:
  - High upgrade intent
  - Persuadable
  - Service recovery first
  - Experience risk
  - Low priority
- Recommends an action for each customer.
- Estimates expected campaign value using adjustable assumptions for membership value, offer cost, and campaign capacity.
- Provides a customer lookup view for reviewing individual recommendations.

## Business Assumptions

The available data does not provide exact premium membership economics, so the dashboard uses adjustable scenario inputs:

- Estimated annual value per upgrade
- Average campaign cost per targeted user
- Campaign capacity

The target variable represents organic upgrade behavior. That means the model predicts similarity to customers who upgraded naturally. It does not prove a customer will respond to a paid offer. A real campaign should validate this with lift tracking or A/B testing.

## Model Summary

The project now uses a Gradient Boosting model selected from local benchmarking against Logistic Regression, Random Forest, ExtraTrees, and Gradient Boosting candidates. The saved model is a scikit-learn pipeline that includes preprocessing and classification.

From the local holdout benchmark:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Top-Decile Lift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous Random Forest | 0.817 | 0.865 | 0.344 | 0.492 | 0.710 | 0.603 | 3.44x |
| Improved Gradient Boosting | 0.831 | 0.971 | 0.355 | 0.520 | 0.700 | 0.611 | 3.66x |

The improved Gradient Boosting model is still precision-led and recall-constrained, but it improves PR-AUC, F1, precision, and top-decile campaign lift versus the previous Random Forest. It is useful for ranking and prioritization, but it should not be presented as a fully mature production model. For a business campaign, the threshold and targeting size should be chosen based on expected value, lift, and campaign capacity, not accuracy alone.

## Model Quality Positioning

The model should be treated as a directional targeting model. It has useful signal and stronger high-priority ranking after the Gradient Boosting update, but recall is still weak at the default threshold. The product experience is therefore designed around campaign ranking, threshold tradeoffs, and business validation rather than blind automated decisions.

Recommended next steps for improving the ML layer:

- Optimize for top-N campaign lift and expected profit.
- Compare Logistic Regression, Random Forest, Gradient Boosting, and calibrated models.
- Tune thresholds based on marketing capacity and acceptable precision/recall tradeoff.
- Track actual campaign response to move from organic upgrade prediction toward offer-response modeling.
## Data

The project uses `food_app_customer_data.csv` with 2,000 customer rows. There are 1,800 labeled rows for upgrade modeling and 200 rows with missing upgrade labels.

Main feature groups:

- Demographics: age, gender, annual income
- Purchase behavior: purchase frequency, spending score, average order value
- Preferences: preferred cuisine, discount usage, cuisines tried
- Service experience: app rating, delivery time, complaints, tips

Engineered features:

- Spend efficiency
- Complaint rate
- Cuisine diversity
- Delivery pain index
- Tip-to-order ratio

## Project Structure

```text
membership-upgrade-predictor/
├── app.py                       # Streamlit business dashboard
├── upgrade_model.pkl            # Improved Gradient Boosting model pipeline
├── model_metadata.json          # Holdout metrics and threshold tradeoffs
├── train_model.py                # Reproducible model training script
├── food_app_customer_data.csv    # Customer dataset
├── Phase 1+2+3.ipynb            # EDA, feature engineering, model training
├── BITSPilani_TeamMuggles.pdf   # Supporting presentation artifact
├── requirements.txt             # Python dependencies
├── images/                      # App assets
└── .streamlit/config.toml       # Streamlit theme config
```

## Run Locally

```bash
# Use Python 3.11 or 3.12 for best dependency compatibility
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Reproducibility Note

`upgrade_model.pkl` should be loaded with a compatible scikit-learn version. The dependency file pins the expected model-loading stack.

## Business Value

The dashboard is designed as an operating tool for three decisions:

- Prioritize the highest-value customers for premium upgrade outreach.
- Avoid pitching upgrades to customers who first need service recovery.
- Compare campaign scenarios using expected upgrades and expected profit.



