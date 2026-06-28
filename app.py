"""Premium Membership Upgrade - Data Analytics Dashboard.

A business analytics app over food-delivery customer data. It explains WHO upgrades
to premium and WHY, scores every customer with a trained model, and turns those
scores into a prioritized, value-estimated campaign target list.

Honest framing: the dataset carries real signal in only a couple of behaviors, so
the model is a directional ranking aid (ROC-AUC ~0.70), not an autopilot.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "upgrade_model.pkl"
DATA_PATH = BASE_DIR / "food_app_customer_data.csv"
METADATA_PATH = BASE_DIR / "model_metadata.json"

RAW_FEATURES = [
    "Age",
    "Gender",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency",
    "Avg_Order_Value",
    "Preferred_Cuisine",
    "Weekend_Order_Ratio",
    "App_Rating",
    "Avg_Delivery_Tips",
    "Discount_Usage_Freq",
    "Total_Cuisines_Tried",
    "Avg_Delivery_Time",
    "Last_Month_Complaints",
]

ENGINEERED_FEATURES = [
    "Spend_Efficiency",
    "Complaint_Rate",
    "Cuisine_Diversity",
    "Delivery_Pain_Index",
    "Tip_to_Order_Ratio",
]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES

# Numeric behaviors used for the EDA / driver views.
NUMERIC_BEHAVIORS = [
    "Age",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency",
    "Avg_Order_Value",
    "Weekend_Order_Ratio",
    "App_Rating",
    "Avg_Delivery_Tips",
    "Total_Cuisines_Tried",
    "Avg_Delivery_Time",
    "Last_Month_Complaints",
]

# Brand + narrative from the BITSPilani_TeamMuggles deck (Insightify 6.0).
PLATFORM = "CraveConnect"
TEAM = "The Muggles"
TEAM_MEMBERS = ["Ankit Nandi", "Shikhar Panthari", "Abhinav Singh"]

PITCHABLE_SEGMENTS = ["High intent", "Persuadable"]

# One locked accent (deep violet, from the CraveConnect deck) + neutrals.
# Semantic colors used ONLY for true status (risk / action), never decoration.
ACCENT = "#4c1d95"
ACCENT_SOFT = "#7c3aed"
INK = "#1c1917"
MUTED = "#78716c"
LINE = "#e7e5e4"
SEGMENT_COLORS = {
    "High intent": "#4c1d95",
    "Persuadable": "#7c3aed",
    "Service recovery": "#b45309",
    "Experience risk": "#b91c1c",
    "Low priority": "#a8a29e",
}
PLOTLY_FONT = "Outfit, system-ui, sans-serif"

MODEL_WARNING = (
    "Use this model as a targeting aid, not an autopilot. The data carries strong "
    "signal in only a few behaviors, so ROC-AUC tops out around 0.70. Rank with it, "
    "then validate the campaign with conversion tracking or an A/B test."
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
LOGGER = logging.getLogger("membership_upgrade_dashboard")


# --------------------------------------------------------------------------- #
# Data + model loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_model():
    LOGGER.info("Loading model from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_customer_data():
    LOGGER.info("Loading customer data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    LOGGER.info("Loaded %s rows, %s columns", df.shape[0], df.shape[1])
    return df


def add_engineered_features(df):
    """Create the behavioral features used by the trained model.

    Kept identical to train_model.py - editing one side without the other
    silently breaks scoring.
    """
    scored = df.copy()
    scored["Spend_Efficiency"] = scored["Spending_Score"] / (scored["Purchase_Frequency"] + 1)
    scored["Complaint_Rate"] = scored["Last_Month_Complaints"] / (scored["Purchase_Frequency"] + 1)
    scored["Cuisine_Diversity"] = scored["Total_Cuisines_Tried"] / 8
    scored["Delivery_Pain_Index"] = (scored["Avg_Delivery_Time"] / 60) + scored["Last_Month_Complaints"]
    scored["Tip_to_Order_Ratio"] = scored["Avg_Delivery_Tips"] / (scored["Avg_Order_Value"] + 1)
    return scored


# --------------------------------------------------------------------------- #
# Business decision layer
# --------------------------------------------------------------------------- #
def assign_segment(row):
    """Convert model + service signals into action-oriented campaign groups."""
    probability = row["Upgrade_Probability"]
    rating = row["App_Rating"]
    complaints = row["Last_Month_Complaints"]
    spending = row["Spending_Score"]

    if spending >= 70 and rating <= 2:
        return "Service recovery"
    if probability >= 0.65:
        return "High intent"
    if probability >= 0.35:
        return "Persuadable"
    if complaints > 0 or rating <= 2:
        return "Experience risk"
    return "Low priority"


def recommend_action(row):
    segment = row["Segment"]
    discount = row["Discount_Usage_Freq"]
    level = row.get("Membership_Level", "")

    if level == "Platinum":
        return "Already premium; exclude from upgrade push"
    if segment == "High intent":
        return "Offer premium trial plus annual plan CTA"
    if segment == "Persuadable" and discount == "High":
        return "Limited-time discount on premium membership"
    if segment == "Persuadable":
        return "Personalized value message with premium benefits"
    if segment == "Service recovery":
        return "Resolve service issue before upgrade pitch"
    if segment == "Experience risk":
        return "Retention check-in; no upgrade pitch yet"
    return "Do not target in this campaign"


def priority_score(row):
    """Rank customers by upgrade signal, customer value, and service risk."""
    service_penalty = 0.18 if row["Segment"] in ["Service recovery", "Experience risk"] else 0
    value_signal = min(row["Spending_Score"] / 100, 1)
    frequency_signal = min(row["Purchase_Frequency"] / 15, 1)
    return (
        row["Upgrade_Probability"] * 0.65
        + value_signal * 0.20
        + frequency_signal * 0.15
        - service_penalty
    )


@st.cache_data(show_spinner=False)
def score_customers(_model, df):
    """Score customer rows and attach business recommendations.

    `_model` is underscore-prefixed so Streamlit does not try to hash it.
    """
    model_input = add_engineered_features(df[RAW_FEATURES])
    probabilities = _model.predict_proba(model_input[MODEL_FEATURES])[:, 1]

    scored = df.copy()
    scored["Upgrade_Probability"] = probabilities
    scored["Segment"] = scored.apply(assign_segment, axis=1)
    scored["Recommended_Action"] = scored.apply(recommend_action, axis=1)
    scored["Priority_Score"] = scored.apply(priority_score, axis=1)
    return scored.sort_values("Priority_Score", ascending=False)


def estimate_campaign_value(scored_df, target_count, membership_value, offer_cost):
    targets = scored_df.head(target_count).copy()
    targets["Expected_Upgrades"] = targets["Upgrade_Probability"]
    targets["Expected_Revenue"] = targets["Upgrade_Probability"] * membership_value
    targets["Expected_Offer_Cost"] = offer_cost
    targets["Expected_Profit"] = targets["Expected_Revenue"] - targets["Expected_Offer_Cost"]
    return targets


# --------------------------------------------------------------------------- #
# Model evaluation (recreated holdout for transparency)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def evaluate_model(_model, df, threshold):
    labeled = df.dropna(subset=["Membership_upgrade"]).copy()
    labeled = add_engineered_features(labeled)
    x = labeled[MODEL_FEATURES]
    y = labeled["Membership_upgrade"].astype(int)
    _, x_test, _, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    probabilities = _model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "PR-AUC": average_precision_score(y_test, probabilities),
    }

    threshold_rows = []
    for t in [0.20, 0.35, 0.50, 0.65]:
        pred = (probabilities >= t).astype(int)
        threshold_rows.append(
            {
                "Threshold": t,
                "Targeted Users": int(pred.sum()),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
            }
        )

    order = np.argsort(-probabilities)
    base_rate = float(y_test.mean())
    deciles = []
    for d in range(1, 11):
        cut = max(int(len(order) * d / 10), 1)
        idx = order[:cut]
        rate = float(y_test.iloc[idx].mean())
        deciles.append({"Top %": d * 10, "Upgrade Rate": rate, "Lift": rate / base_rate if base_rate else 0})

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    return metrics, pd.DataFrame(threshold_rows), pd.DataFrame(deciles), base_rate, (fpr, tpr)


def feature_correlations(df):
    labeled = df.dropna(subset=["Membership_upgrade"]).copy()
    labeled = add_engineered_features(labeled)
    cols = NUMERIC_BEHAVIORS + ENGINEERED_FEATURES
    corr = labeled[cols + ["Membership_upgrade"]].corr()["Membership_upgrade"].drop("Membership_upgrade")
    return corr.sort_values()


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def format_currency(value):
    return f"${value:,.0f}"


def style_fig(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family=PLOTLY_FONT, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        colorway=[ACCENT, ACCENT_SOFT, "#a8a29e", "#b45309", "#b91c1c", "#0f766e"],
    )
    fig.update_xaxes(gridcolor=LINE, zerolinecolor="#d6d3d1", linecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor="#d6d3d1", linecolor=LINE)
    return fig


def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');
        :root {
            --bg:#f7f7f5; --surface:#ffffff; --line:#e7e5e4; --line-strong:#d6d3d1;
            --ink:#1c1917; --muted:#78716c; --accent:#4c1d95; --accent-soft:#f5f3ff;
            --warn:#b45309; --warn-soft:#fffbeb; --good:#15803d; --good-soft:#f0fdf4;
            --radius:10px;
        }
        html, body, [class*="css"], .stApp { font-family:'Outfit', system-ui, sans-serif; }
        .stApp { background: var(--bg); color: var(--ink); }
        .block-container { padding-top:1.4rem; padding-bottom:2.5rem; max-width:1440px; }
        h1,h2,h3,h4,h5,h6 { color:var(--ink); letter-spacing:-0.02em; font-weight:700; }
        p,label,span,li { color:var(--ink); }

        /* Sidebar: neutral, not baby-blue */
        [data-testid="stSidebar"] { background:var(--surface); border-right:1px solid var(--line); }
        [data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span { color:var(--ink); }

        /* Metric: hairline card, mono number, no candy top-border */
        div[data-testid="stMetric"] {
            border:1px solid var(--line); border-radius:var(--radius);
            padding:14px 16px; background:var(--surface);
            box-shadow:0 1px 2px rgba(28,25,23,.04);
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color:var(--muted); font-weight:600; font-size:.72rem;
            text-transform:uppercase; letter-spacing:.06em;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color:var(--ink); font-weight:700; font-family:'JetBrains Mono', monospace;
            letter-spacing:-.01em;
        }

        /* Tabs: clean underline, no folder chrome */
        .stTabs [data-baseweb="tab-list"] { gap:4px; border-bottom:1px solid var(--line-strong); }
        .stTabs [data-baseweb="tab"] {
            background:transparent; border:0; border-bottom:2px solid transparent;
            color:var(--muted); font-weight:600; padding:10px 14px; border-radius:0;
        }
        .stTabs [aria-selected="true"] { color:var(--accent); border-bottom-color:var(--accent); }

        /* Buttons: single accent, tactile press */
        .stButton > button, .stDownloadButton > button {
            background:var(--accent); color:#fff; border:1px solid var(--accent);
            border-radius:var(--radius); font-weight:600; transition:transform .08s ease, filter .15s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { filter:brightness(1.1); color:#fff; }
        .stButton > button:active, .stDownloadButton > button:active { transform:translateY(1px); }

        /* Unified callouts: subtle tint + left hairline, one shape */
        .note, .risk, .insight {
            border:1px solid var(--line); border-left:3px solid var(--accent); border-radius:var(--radius);
            background:var(--accent-soft); padding:.8rem 1rem; margin:.5rem 0 1rem;
            color:var(--ink); font-weight:500; line-height:1.5;
        }
        .risk { border-left-color:var(--warn); background:var(--warn-soft); }
        .insight { border-left-color:var(--good); background:var(--good-soft); }

        [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:var(--radius); }
        hr { border-color:var(--line); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        f"""
        <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;padding-top:.9rem;line-height:1.35;">
          <span style="font-size:2rem;font-weight:800;color:#4c1d95;letter-spacing:-.03em;">{PLATFORM}</span>
          <span style="font-size:1.4rem;font-weight:600;color:#1c1917;">Premium Membership Growth Analytics</span>
        </div>
        <div style="color:#78716c;font-weight:500;margin-top:2px;">
          Insightify 6.0 &nbsp;·&nbsp; {TEAM} &nbsp;·&nbsp; {", ".join(TEAM_MEMBERS)}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write(
        f"Who upgrades to {PLATFORM} premium and why - then turn the model scores into two "
        "concrete growth plays: grow premium adoption, and win back high-value unhappy customers."
    )
    st.markdown(f'<div class="risk"><strong>Model note:</strong> {MODEL_WARNING}</div>', unsafe_allow_html=True)


def persona_card(name, tagline, body, strategy_title, strategy_body):
    st.markdown(
        f"""
        <div style="border:1px solid #e7e5e4;border-left:3px solid #4c1d95;border-radius:10px;
                    background:#fff;padding:1rem 1.2rem;box-shadow:0 1px 2px rgba(28,25,23,.04);">
          <div style="font-size:1.15rem;font-weight:700;color:#4c1d95;letter-spacing:-.01em;">Persona - {name}</div>
          <div style="color:#78716c;font-weight:600;margin:2px 0 8px;">{tagline}</div>
          <div style="color:#1c1917;line-height:1.55;">{body}</div>
          <div style="margin-top:10px;font-weight:700;color:#15803d;">{strategy_title}</div>
          <div style="color:#1c1917;line-height:1.55;">{strategy_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Tab: Overview
# --------------------------------------------------------------------------- #
def render_overview(df, scored_df):
    labeled = df.dropna(subset=["Membership_upgrade"])
    upgrade_rate = labeled["Membership_upgrade"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Labeled", f"{len(labeled):,}")
    c3.metric("Organic Upgrade Rate", f"{upgrade_rate:.1%}")
    c4.metric("High-intent Customers", f"{(scored_df['Segment'] == 'High intent').sum():,}")
    c5.metric("Avg Upgrade Probability", f"{scored_df['Upgrade_Probability'].mean():.1%}")

    st.markdown(
        f'<div class="note"><strong>Problem statement.</strong> {PLATFORM} wants to grow premium '
        "membership revenue. Demographics (age, income) barely differ across tiers, so the answer "
        "is in <em>behavior</em>. This dashboard finds the behavioral signals, scores every customer, "
        "and frames two growth strategies built on real personas.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="insight"><strong>Headline finding:</strong> Upgraders are NOT the heaviest '
        "spenders. Customers who upgrade have lower Spending_Score (~39 vs 56) and lower "
        "Purchase_Frequency (~6.4 vs 8.9). Premium appeals to lighter, more deliberate users - "
        "target casual customers, not your top spenders.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Upgrade Mix")
        counts = labeled["Membership_upgrade"].map({0: "Did not upgrade", 1: "Upgraded"}).value_counts()
        fig = px.pie(values=counts.values, names=counts.index, hole=0.55,
                     color_discrete_sequence=["#a8a29e", "#4c1d95"])
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    with right:
        st.subheader("Campaign Segments (all customers)")
        seg = scored_df["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
        fig = px.bar(seg, x="Customers", y="Segment", orientation="h", color="Segment",
                     color_discrete_map=SEGMENT_COLORS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab: EDA / Who Upgrades
# --------------------------------------------------------------------------- #
def render_eda(df):
    st.subheader("Who Upgrades? Behavior Comparison")
    st.write("Compare upgraders vs non-upgraders across each behavior. Big gaps = real signal.")

    labeled = df.dropna(subset=["Membership_upgrade"]).copy()
    labeled["Upgrade"] = labeled["Membership_upgrade"].map({0: "Did not upgrade", 1: "Upgraded"})

    means = labeled.groupby("Upgrade")[NUMERIC_BEHAVIORS].mean().T
    means["Difference %"] = ((means["Upgraded"] - means["Did not upgrade"]) / means["Did not upgrade"] * 100)
    means = means.reindex(means["Difference %"].abs().sort_values(ascending=False).index)

    fig = px.bar(means.reset_index(), x="Difference %", y="index", orientation="h",
                 color="Difference %", color_continuous_scale="RdBu", range_color=[-35, 35])
    fig.update_layout(yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    st.caption("Negative = upgraders score lower on this behavior. Spending and frequency dominate.")

    st.subheader("Distribution Explorer")
    feature = st.selectbox("Behavior", NUMERIC_BEHAVIORS, index=NUMERIC_BEHAVIORS.index("Spending_Score"))
    fig = px.histogram(labeled, x=feature, color="Upgrade", barmode="overlay", nbins=40,
                       color_discrete_map={"Did not upgrade": "#a8a29e", "Upgraded": "#4c1d95"},
                       marginal="box")
    st.plotly_chart(style_fig(fig, 420), use_container_width=True)

    st.subheader("Upgrade Rate by Category")
    cat = st.selectbox("Category", ["Membership_Level", "Gender", "Preferred_Cuisine", "Discount_Usage_Freq"])
    rate = labeled.groupby(cat)["Membership_upgrade"].agg(["mean", "count"]).reset_index()
    rate.columns = [cat, "Upgrade Rate", "Customers"]
    rate = rate.sort_values("Upgrade Rate", ascending=False)
    fig = px.bar(rate, x=cat, y="Upgrade Rate", text=rate["Upgrade Rate"].map(lambda v: f"{v:.0%}"),
                 color="Upgrade Rate", color_continuous_scale="Blues")
    fig.update_layout(coloraxis_showscale=False, yaxis_tickformat=".0%")
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)


# --------------------------------------------------------------------------- #
# Tab: Drivers
# --------------------------------------------------------------------------- #
def render_drivers(df, model):
    st.subheader("What Drives an Upgrade?")
    st.write("Correlation of each behavior with the upgrade outcome. Near-zero = noise.")

    corr = feature_correlations(df)
    fig = px.bar(x=corr.values, y=corr.index, orientation="h",
                 color=corr.values, color_continuous_scale="RdBu", range_color=[-0.3, 0.3])
    fig.update_layout(xaxis_title="Correlation with upgrade", yaxis_title="", coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 460), use_container_width=True)

    st.markdown(
        '<div class="note"><strong>Read this honestly:</strong> Only Purchase_Frequency and '
        "Spending_Score (and the engineered Spend_Efficiency) carry meaningful signal - both "
        "negative. The rest hover near zero, which is exactly why no model beats ~0.70 ROC-AUC on "
        "this data. The fix is better data, not a fancier algorithm.</div>",
        unsafe_allow_html=True,
    )

    # Model-side importance, if the pipeline exposes it.
    try:
        importances = model.named_steps["model"].feature_importances_
        feat_names = model.named_steps["preprocessor"].get_feature_names_out()
        imp = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        imp = imp.sort_values("Importance", ascending=False).head(12)
        imp["Feature"] = imp["Feature"].str.replace("num__", "").str.replace("cat__", "")
        st.subheader("Model Feature Importance (top 12)")
        fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                     color_discrete_sequence=["#4c1d95"])
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
    except Exception:
        st.info("Model does not expose feature importances.")


# --------------------------------------------------------------------------- #
# Tab: Predict & Target
# --------------------------------------------------------------------------- #
def filter_targets(scored_df, controls):
    targets = scored_df.copy()
    targets = targets[targets["Upgrade_Probability"] >= controls["min_probability"]]
    if controls["included_segments"]:
        targets = targets[targets["Segment"].isin(controls["included_segments"])]
    if controls["exclude_platinum"] and "Membership_Level" in targets.columns:
        targets = targets[targets["Membership_Level"] != "Platinum"]
    return targets.sort_values("Priority_Score", ascending=False)


def render_campaign(scored_df, controls):
    st.subheader("Strategy I - Grow Premium Adoption")
    persona_card(
        "Rahul - The Occasional Optimizer",
        "Light, low-frequency user · modest spender · deliberate about value",
        "Corrected from the data: the customers our model ranks highest for upgrade are NOT the "
        "power spenders. They order rarely (~3 orders/month) and spend modestly (Spending Score ~18 "
        "vs ~52 for the average customer). For an occasional user, a per-order delivery fee stings - "
        "so premium that removes that fee is exactly what makes their infrequent orders worth it.",
        "The play: \"Premium Pays for Itself\"",
        "Lead with simple math - premium pays back in just a few orders via free/reduced delivery and "
        "no surge - plus a low-commitment trial. Pitch value-per-order, not volume perks. Use the "
        "target list below to reach every Occasional Optimizer at scale.",
    )
    st.write("")

    eligible = filter_targets(scored_df, controls)
    campaign = estimate_campaign_value(
        eligible, controls["campaign_capacity"], controls["membership_value"], controls["offer_cost"]
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Qualified Pool", f"{len(eligible):,}")
    k2.metric("Selected Targets", f"{len(campaign):,}")
    k3.metric("Avg Probability", f"{(campaign['Upgrade_Probability'].mean() if len(campaign) else 0):.1%}")
    k4.metric("Expected Upgrades", f"{campaign['Expected_Upgrades'].sum():,.1f}")
    k5.metric("Expected Profit", format_currency(campaign["Expected_Profit"].sum()))

    st.markdown(
        '<div class="note">Workflow: tune campaign rules in the sidebar, review the segment + action '
        "mix, export the list, then track real conversion after outreach.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Segment Mix (eligible)")
        seg = eligible["Segment"].value_counts().rename_axis("Segment").reset_index(name="Customers")
        if len(seg):
            fig = px.bar(seg, x="Segment", y="Customers", color="Segment", color_discrete_map=SEGMENT_COLORS)
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
        else:
            st.info("No customers match the current rules.")
    with right:
        st.subheader("Action Mix (selected)")
        act = campaign["Recommended_Action"].value_counts().rename_axis("Action").reset_index(name="Customers")
        if len(act):
            fig = px.bar(act, x="Customers", y="Action", orientation="h", color_discrete_sequence=["#4c1d95"])
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
        else:
            st.info("No selected targets. Relax filters or raise capacity.")

    st.subheader("Campaign Target List")
    cols = [
        "CustomerID", "Name", "Membership_Level", "Segment", "Recommended_Action",
        "Upgrade_Probability", "Priority_Score", "Purchase_Frequency", "Avg_Order_Value",
        "Spending_Score", "App_Rating", "Discount_Usage_Freq", "Expected_Profit",
    ]
    if len(campaign):
        st.dataframe(
            campaign[cols].style.format(
                {"Upgrade_Probability": "{:.1%}", "Priority_Score": "{:.2f}",
                 "Avg_Order_Value": "${:,.0f}", "Expected_Profit": "${:,.0f}"}
            ),
            use_container_width=True, hide_index=True,
        )
        csv = campaign[cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download target list", data=csv,
                           file_name="premium_upgrade_campaign_targets.csv", mime="text/csv", type="primary")
    else:
        st.warning("No targets match the current rules. Lower probability or add Persuadable users.")

    st.subheader("Customer Lookup")
    term = st.text_input("Search by customer ID or name")
    view = scored_df
    if term:
        mask = (view["CustomerID"].str.contains(term, case=False, na=False)
                | view["Name"].str.contains(term, case=False, na=False))
        view = view[mask]
    look_cols = ["CustomerID", "Name", "Membership_Level", "Segment",
                 "Upgrade_Probability", "Recommended_Action", "Priority_Score"]
    st.dataframe(
        view[look_cols].head(50).style.format({"Upgrade_Probability": "{:.1%}", "Priority_Score": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )


# --------------------------------------------------------------------------- #
# Tab: Strategy II - Win Back (Neha / High-Value Low-Rating)
# --------------------------------------------------------------------------- #
def render_winback(scored_df, controls):
    st.subheader("Strategy II - Win Back High-Value, Unhappy Customers")
    persona_card(
        "Neha",
        "High spender · frequent orderer · low app rating · churn risk",
        "Neha spends heavily and orders often, but late deliveries and cold food dropped her rating. "
        "She is high-value yet dissatisfied - the costliest customer to lose silently.",
        "The play: \"Win Back the Food Lover\"",
        "Treat her like a VIP foodie, not a complaint ticket: proactive service recovery, priority "
        "support, and a goodwill perk BEFORE any upgrade pitch.",
    )
    st.write("")

    hv = scored_df[(scored_df["Spending_Score"] > 70) & (scored_df["App_Rating"] < 3)].copy()
    other = scored_df[~((scored_df["Spending_Score"] > 70) & (scored_df["App_Rating"] < 3))]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High-Value / Low-Rating", f"{len(hv):,}")
    k2.metric("Share of base", f"{len(hv) / len(scored_df):.1%}")
    k3.metric("Avg Spending Score", f"{hv['Spending_Score'].mean():.0f}")
    k4.metric("vs Rest (Spending)", f"{other['Spending_Score'].mean():.0f}")

    st.markdown(
        f'<div class="risk"><strong>Why it matters:</strong> only ~{len(hv) / len(scored_df):.1%} of '
        "customers, but they spend far above average. Losing them quietly costs more than any single "
        "failed upgrade. Recover service first, monetize later.</div>",
        unsafe_allow_html=True,
    )

    compare = pd.DataFrame(
        {
            "Metric": ["Purchase Frequency", "Spending Score", "Avg Order Value"],
            "High-Value/Low-Rating": [hv["Purchase_Frequency"].mean(), hv["Spending_Score"].mean(), hv["Avg_Order_Value"].mean()],
            "Other Customers": [other["Purchase_Frequency"].mean(), other["Spending_Score"].mean(), other["Avg_Order_Value"].mean()],
        }
    )
    melt = compare.melt(id_vars="Metric", var_name="Group", value_name="Value")
    fig = px.bar(melt, x="Metric", y="Value", color="Group", barmode="group",
                 color_discrete_map={"High-Value/Low-Rating": "#b91c1c", "Other Customers": "#a8a29e"})
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)

    st.subheader("Win-Back Target List")
    cols = ["CustomerID", "Name", "Membership_Level", "Spending_Score", "App_Rating",
            "Last_Month_Complaints", "Purchase_Frequency", "Avg_Order_Value", "Recommended_Action"]
    cols = [c for c in cols if c in hv.columns]
    if len(hv):
        view = hv.sort_values("Spending_Score", ascending=False)[cols]
        st.dataframe(
            view.style.format({"Avg_Order_Value": "${:,.0f}"}),
            use_container_width=True, hide_index=True,
        )
        csv = view.to_csv(index=False).encode("utf-8")
        st.download_button("Download win-back list", data=csv,
                           file_name="craveconnect_winback_targets.csv", mime="text/csv", type="primary")
    else:
        st.info("No high-value/low-rating customers in the current data.")


# --------------------------------------------------------------------------- #
# Tab: Model Health
# --------------------------------------------------------------------------- #
def render_model_health(model, df, threshold):
    st.subheader("Model Health")
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        st.caption(f"Saved model: {meta.get('model_name', 'scikit-learn pipeline')} - {meta.get('selection_reason', '')}")
    st.markdown(f'<div class="risk"><strong>Honest read:</strong> {MODEL_WARNING}</div>', unsafe_allow_html=True)

    metrics, threshold_df, decile_df, base_rate, roc = evaluate_model(model, df, threshold)

    m = st.columns(6)
    m[0].metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
    m[1].metric("PR-AUC", f"{metrics['PR-AUC']:.3f}")
    m[2].metric(f"Precision @{threshold:.2f}", f"{metrics['Precision']:.3f}")
    m[3].metric(f"Recall @{threshold:.2f}", f"{metrics['Recall']:.3f}")
    m[4].metric(f"F1 @{threshold:.2f}", f"{metrics['F1']:.3f}")
    m[5].metric("Top-Decile Lift", f"{decile_df.iloc[0]['Lift']:.1f}x")

    left, right = st.columns(2)
    with left:
        st.subheader("ROC Curve")
        fpr, tpr = roc
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Model", line=dict(color="#4c1d95", width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                 line=dict(color="#a8a29e", dash="dash")))
        fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    with right:
        st.subheader("Cumulative Lift by Decile")
        fig = px.line(decile_df, x="Top %", y="Lift", markers=True, color_discrete_sequence=["#15803d"])
        fig.add_hline(y=1.0, line_dash="dash", line_color="#a8a29e")
        st.plotly_chart(style_fig(fig, 360), use_container_width=True)
    st.caption("Lift chart is the real campaign value: contact the top deciles to hit upgraders far above base rate.")

    st.subheader("Threshold Tradeoff")
    st.dataframe(
        threshold_df.style.format({"Threshold": "{:.2f}", "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

    st.subheader("Recommended Next Steps")
    st.markdown(
        """
        - **Target by lift / top-N, not a fixed 0.50 cutoff** - pick how many to contact, then take the top of the ranking.
        - **Collect better signal** - the data ceiling is ~0.70 AUC; richer behavioral data would help more than any algorithm swap.
        - **Optimize for campaign profit or recall** at acceptable precision, not plain accuracy.
        - **Track real campaign response** - organic upgrade prediction is not the same as offer-response modeling.
        """
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar(scored_df):
    st.sidebar.title("Campaign Setup")
    st.sidebar.caption("Set business assumptions and targeting rules.")

    membership_value = st.sidebar.number_input("Annual value per upgrade", 0, value=1200, step=50)
    offer_cost = st.sidebar.number_input("Cost per contacted user", 0, value=150, step=25)
    campaign_capacity = st.sidebar.number_input(
        "Campaign capacity", 1, max(len(scored_df), 1), value=min(100, len(scored_df)), step=10
    )
    min_probability = st.sidebar.slider("Minimum upgrade probability", 0.0, 1.0, 0.30, 0.05)
    included_segments = st.sidebar.multiselect(
        "Segments to include", sorted(scored_df["Segment"].unique()), default=PITCHABLE_SEGMENTS
    )
    exclude_platinum = st.sidebar.checkbox("Exclude current Platinum customers", value=True)

    st.sidebar.markdown("---")
    threshold = st.sidebar.slider("Model decision threshold (Model Health tab)", 0.20, 0.80, 0.50, 0.05)

    return {
        "membership_value": membership_value,
        "offer_cost": offer_cost,
        "campaign_capacity": int(campaign_capacity),
        "min_probability": min_probability,
        "included_segments": included_segments,
        "exclude_platinum": exclude_platinum,
        "threshold": threshold,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="Premium Upgrade Analytics", layout="wide")
    inject_css()
    render_header()

    try:
        model = load_model()
        df = load_customer_data()
    except Exception as exc:
        LOGGER.exception("Startup failed")
        st.error(f"Could not load project assets: {exc}")
        st.stop()

    try:
        scoring_columns = RAW_FEATURES + ["CustomerID", "Name", "Membership_Level", "Membership_upgrade"]
        scored_df = score_customers(model, df[scoring_columns])
    except Exception as exc:
        LOGGER.exception("Scoring failed")
        st.error(f"Scoring failed: {exc}")
        st.stop()

    controls = render_sidebar(scored_df)

    tabs = st.tabs([
        "Overview",
        "Problem & EDA",
        "Predictive Modelling",
        "Strategy I - Grow Premium",
        "Strategy II - Win Back",
    ])
    with tabs[0]:
        render_overview(df, scored_df)
    with tabs[1]:
        render_eda(df)
    with tabs[2]:
        render_drivers(df, model)
        st.divider()
        render_model_health(model, df, controls["threshold"])
    with tabs[3]:
        render_campaign(scored_df, controls)
    with tabs[4]:
        render_winback(scored_df, controls)

    st.divider()
    st.caption(f"{PLATFORM} · Insightify 6.0 · {TEAM} - {', '.join(TEAM_MEMBERS)}")


if __name__ == "__main__":
    main()
