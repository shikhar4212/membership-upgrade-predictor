# 🍔 Membership Upgrade Predictor

A **Streamlit** dashboard that predicts the probability of a food-delivery customer
upgrading to a **premium membership**, based on their demographics, ordering behaviour,
spending patterns, and service experience.

Enter a customer's details (or pick a ready-made persona) and the trained model returns
an upgrade probability with a high / moderate / low recommendation.

---

## ✨ Features

- Interactive form for 14 customer attributes (age, income, order value, ratings, etc.)
- Three built-in test personas (likely / unsure / unlikely to upgrade)
- Engineered features computed live (spend efficiency, complaint rate, delivery pain index, …)
- Probability output with colour-coded likelihood tiers
- Clean sidebar with project description and team cards

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — web UI
- **scikit-learn** — trained classification model
- **pandas / numpy** — data handling
- **joblib** — model serialization

---

## 🚀 Run Locally

```bash
# 1. Clone
git clone https://github.com/shikhar4212/membership-upgrade-predictor.git
cd membership-upgrade-predictor

# 2. (Recommended) create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

The app opens at `http://localhost:8501`.

> **Note:** `upgrade_model.pkl` must load with the same scikit-learn version it was
> trained on. If you hit a model-load error, set the matching version in `requirements.txt`.

---

## 📂 Project Structure

```
membership-upgrade-predictor/
├── app.py                  # Streamlit dashboard
├── upgrade_model.pkl       # Trained model
├── food_app_customer_data.csv   # Dataset
├── Phase 1+2+3.ipynb       # EDA + feature engineering + model training
├── requirements.txt
├── images/                 # Persona & team avatars
└── .streamlit/             # Theme config
```

---

## 🧠 How It Works

1. User inputs raw customer attributes through the dashboard.
2. The app derives additional features (spend efficiency, complaint rate, cuisine
   diversity, delivery pain index, tip-to-order ratio, weekend order ratio).
3. The full feature row is passed to the trained scikit-learn model.
4. `predict_proba` returns the upgrade probability, shown with a recommendation tier.

---

## 👥 Team

Built by **Team Muggles** — MBA Business Analytics, BITS Pilani:

- [Shikhar Panthari](https://www.linkedin.com/in/shikhar-panthari/)
- [Abhinav Singh](https://www.linkedin.com/in/abhinav-singh-bits/)
- [Ankit Nandi](https://www.linkedin.com/in/ankit-nandi-b53ab71a1/)
