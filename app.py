import streamlit as st
import pandas as pd
import joblib
def team_card(image, name, role, school, link):

    col1, col2 = st.sidebar.columns([1,3])

    with col1:
        st.image(image, width=55)

    with col2:
        st.markdown(
            f"""
<a href="{link}" target="_blank" style="text-decoration:none;color:inherit;">
<div class="team-text">
<div class="team-name">{name}</div>
<div class="team-role">{role}</div>
<div class="team-role">{school}</div>
</div>
</a>
""",
            unsafe_allow_html=True
        )

st.set_page_config(
    page_title="Customer Upgrade Dashboard",
    layout="wide"
)
st.markdown("""
<style>
/* Remove top padding from sidebar */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem;
            
}
            section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem;
    margin-top: -20px;
}

/* Optional: reduce overall sidebar padding */
section[data-testid="stSidebar"] .block-container {
    padding-top: 0rem;
}


.team-card{
    display:flex;
    align-items:center;
    gap:14px;
    padding:12px;
    border-radius:12px;
    background:#e1c3ff;
    margin-bottom:12px;
    transition:0.2s;
}

.team-card:hover{
    background:#8c52ff;
}

.team-img{
    border-radius:50%;
}

.team-text{
    line-height:1.2;
}

.team-name{
    font-weight:600;
    font-size:15px;
}

.team-role{
    font-size:13px;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<style>

.member-card{
    display:flex;
    align-items:center;
    border-radius:10px;
    background:#e1c3ff;
    text-decoration:none;
    color:black;
    transition:0.2s;
    margin-bottom:12px;

                    
}


.member-card:hover{
    background:#8c52ff;
    color:white;
}

.member-img{
    width:60px;
    height:60px;
    border-radius:50%;
    object-fit:cover;
}

.member-text{
    line-height:1.3;
}

.member-name{
    font-weight:600;
    font-size:16px;
}

.member-role{
    font-size:13px;
}

</style>


""", unsafe_allow_html=True)
model = joblib.load("upgrade_model.pkl")

st.title("Customer Upgrade Prediction Dashboard")

st.write("Enter customer behaviour details to estimate the probability of upgrading to a premium membership.")

# ------------------------
# CUSTOMER PROFILE
# ------------------------

st.subheader("Customer Profile")

st.caption("Basic demographic information about the customer.")
col1, col2 = st.sidebar.columns([1,2])

st.sidebar.markdown("### So what is this for?")

st.sidebar.markdown("This dashboard analyzes customer behavior and service experience to estimate the probability that a user will upgrade to a premium membership tier. By combining demographic information, ordering patterns, spending behavior, and service feedback, the model identifies customers who are most likely to benefit from targeted upgrade offers.")


st.sidebar.markdown("### Made By")

team_card(
    "images/shikhar.svg",
    "Shikhar Panthari",
    "MBA Business Analytics",
    "BITS Pilani",
    "https://www.linkedin.com/in/shikhar-panthari/"
)

team_card(
    "images/abhinav.svg",
    "Abhinav Singh",
    "MBA Business Analytics",
    "BITS Pilani",
    "https://www.linkedin.com/in/abhinav-singh-bits/"
)

team_card(
    "images/ankit.svg",
    "Ankit Nandi",
    "MBA Business Analytics",
    "BITS Pilani",
    "https://www.linkedin.com/in/ankit-nandi-b53ab71a1/"
)
presets = {
    "Custom Input": None,

    "Neha-Will Likely Upgrade": {
        "age": 42,
        "gender": "Female",
        "income": 59599,
        "purchase_freq":1.8,
        "spending_score":20,
        "order_value":59.91,
        "weekend_orders":12,
        "rating":4,
        "delivery_time":46.3,
        "complaints":0,
        "tips":8.25,
        "cuisine":"Pizza",
        "discount_usage":"High",
        "cuisines_tried":5,     
    },

    "Rahul-is unsure about upgrading": {
        "age": 33,
        "gender": "Male",
        "income": 50550,
        "purchase_freq":5.1,
        "spending_score":39,
        "order_value":60.53,
        "weekend_orders":420,
        "rating":3,
        "delivery_time":53.5,
        "complaints":0,
        "tips":5.22,
        "cuisine":"Burger",
        "discount_usage":"Medium",
        "cuisines_tried":6, 
    },

    "Raj-Doesn't want to upgrade": {
        "age": 38,
        "gender": "Male",
        "income": 51739,
        "purchase_freq":5.2,
        "spending_score":59,
        "order_value":75.03,
        "weekend_orders":15,
        "rating":5,
        "delivery_time":31.3,
        "complaints":0,
        "tips":4.18,
        "cuisine":"Mexican",
        "discount_usage":"Medium",
        "cuisines_tried":2,
   
    }
}


preset = st.selectbox(
    "Select Test Customer Profile",
    list(presets.keys())
)
col1, col2 = st.columns([2,1])

with col2:
    if preset == "Neha-Will Likely Upgrade":
        st.image("images/neha.svg", width=200)

        st.markdown("""
        **Neha – Will Likely Upgrade**

Neha is a 42-year-old working professional who regularly orders food online due to her busy schedule. With an annual income of around ₹60K, she values convenience and affordability, often choosing delivery over cooking on weekdays.

She orders food almost twice a week and usually spends about ₹60 per order, typically preferring comfort foods like pizza. Neha frequently looks for discounts and promotional offers, which strongly influence her ordering decisions.

Despite her moderate spending habits, Neha is a reliable and satisfied customer. She maintains a high app rating (4/5), rarely complains about orders, and even leaves delivery tips as appreciation for good service.
        """)

    elif preset == "Rahul-is unsure about upgrading":

        st.image("images/rahul.svg", width=200)

        st.markdown("""

Rahul is a 33-year-old young professional who relies heavily on food delivery due to his busy work routine. With an annual income of around ₹50K, he prioritizes convenience and quick meals, frequently ordering food instead of cooking at home.

He places orders more than five times a week, showing very high engagement with the platform. Rahul usually spends about ₹60 per order, often opting for fast, comfort food like burgers. While he occasionally uses discounts, his ordering behavior is less price-sensitive compared to some other users.

Rahul’s app rating is moderate (3/5), and his delivery experience has been fairly consistent with no complaint history, although slightly longer delivery times suggest occasional service delays. Despite this, he continues to order frequently, indicating strong platform dependency.

        """)
    elif preset == "Raj-Doesn't want to upgrade":
        st.image("images/raj.svg", width=200)


        st.markdown("""

Raj is a 38-year-old professional who frequently relies on food delivery as part of his daily routine. With an annual income of around ₹51K, he values quality and convenience when ordering meals.

He orders food over five times a week, indicating very high engagement with the platform. Arjun tends to spend more per order (around ₹75) compared to typical users, often choosing flavorful cuisines like Mexican. While he occasionally uses discounts, his decisions are driven more by taste and experience rather than price sensitivity.

Arjun consistently rates the app very highly (5/5), reflecting strong satisfaction with the service. His orders are delivered relatively quickly, and he has never raised complaints, suggesting a smooth and reliable experience.

        """)

with col1:
    if presets[preset]:
        st.session_state.age = presets[preset]["age"]
        st.session_state.gender = presets[preset]["gender"]
        st.session_state.income = presets[preset]["income"]
        st.session_state.purchase_freq = presets[preset]["purchase_freq"]
        st.session_state.spending_score = presets[preset]["spending_score"]
        st.session_state.order_value = presets[preset]["order_value"]
        st.session_state.weekend_orders = presets[preset]["weekend_orders"]
        st.session_state.rating = presets[preset]["rating"]
        st.session_state.delivery_time = presets[preset]["delivery_time"]
        st.session_state.complaints = presets[preset]["complaints"]
        st.session_state.tips = presets[preset]["tips"]
        st.session_state.cuisine = presets[preset]["cuisine"]
        st.session_state.discount_usage = presets[preset]["discount_usage"]
        st.session_state.cuisines_tried = presets[preset]["cuisines_tried"]

        

    age = st.number_input(
        "Age",
        min_value=12,
        max_value=90,
        step=1,
        format="%d",
        help="Age of the customer in years.",
        key = "age"
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        help="Gender of the customer.",
        key = "gender"
    )

    income = st.slider(
        "Annual Income",
        0,100000,
        help="Yearly Income of the customer.",
        key = "income"
    )

    # ------------------------
    # PURCHASE BEHAVIOR
    # ------------------------

    st.subheader("Purchase Behaviour")

    st.caption("Metrics describing how frequently and how much the customer orders.")

    purchase_freq = st.number_input(
        "Purchase Frequency",
        min_value=0.1,
        step = 0.1,
        help="Number of orders placed by the customer within a given time period.",
        key = "purchase_freq"
    )

    spending_score = st.slider(
        "Spending Score",
        0,100,
        help="A score representing the customer's spending behaviour and engagement.",
        key = "spending_score"

    )

    order_value = st.number_input(
        "Average Order Value",
        help="Average value of orders placed by the customer.",
        key = "order_value"
    )

    weekend_orders = st.number_input(
        "Weekend_Order_Ratio",
        min_value=0,
        step=1,
        help="Proportion of orders placed during weekends.",
        key = "weekend_orders"
    )
    # ------------------------
    # SERVICE EXPERIENCE
    # ------------------------

    st.subheader("Service Experience")

    st.caption("Customer satisfaction and delivery experience metrics.")

    rating = st.slider(
        "App Rating",
        1.0,5.0,
        help="Average rating the customer gives to the app experience.",
        key = "rating"
    )

    delivery_time = st.number_input(
        "Average Delivery Time",
        help="Average time (in minutes) taken for orders to be delivered.",
        key = "delivery_time"
    )

    complaints = st.number_input(
        "Last Month Complaints",
        help="Number of complaints made by the customer in the last month.",
        key = "complaints"
    )

    tips = st.number_input(
        "Average Delivery Tips",
        help="Average tip amount given to delivery personnel.",
        key = "tips"
    )

    # ------------------------
    # CUSTOMER PREFERENCES
    # ------------------------

    st.subheader("Customer Preferences")

    st.caption("Information about cuisine choices and discount behaviour.")

    cuisine = st.selectbox(
        "Preferred Cuisine",
        ["Indian","Chinese","Italian","Mexican","Burger","Pizza","Healthy","Thai"],
        help="Cuisine category most frequently ordered by the customer.",
        key = "cuisine"
    )

    discount_usage = st.selectbox(
        "Discount Usage Frequency",
        ["Low","Medium","High"],
        help="How frequently the customer uses discounts or promotional offers.",
        key = "discount_usage"
    )

    cuisines_tried = st.number_input(
        "Total Cuisines Tried",
        min_value=1,
        step=1,
        format="%d",
        help="Number of different cuisine categories the customer has tried.",
        key = "cuisines_tried"
    )

    # -----------------------
    # PREDICTION
    # -----------------------

    if st.button("Predict Upgrade Probability"):
        

        # Derived Features

        spend_eff = spending_score / (purchase_freq + 1)

        complaint_rate = complaints / (purchase_freq + 1)

        cuisine_div = cuisines_tried / 8

        delivery_pain = (delivery_time / 60) + complaints

        tip_ratio = tips / (order_value + 1)

        weekend_ratio = weekend_orders/purchase_freq*12

        # Create dataframe with ALL required features
        

        new_customer = pd.DataFrame({
            'Age':[age],
            'Gender':[gender],
            'Annual_Income':[income],
            'Spending_Score':[spending_score],
            'Purchase_Frequency':[purchase_freq],
            'Avg_Order_Value':[order_value],
            'Preferred_Cuisine':[cuisine],
            'Weekend_Order_Ratio':[weekend_ratio],
            'App_Rating':[rating],
            'Avg_Delivery_Tips':[tips],
            'Discount_Usage_Freq':[discount_usage],
            'Total_Cuisines_Tried':[cuisines_tried],
            'Avg_Delivery_Time':[delivery_time],
            'Last_Month_Complaints':[complaints],
            'Spend_Efficiency':[spend_eff],
            'Complaint_Rate':[complaint_rate],
            'Cuisine_Diversity':[cuisine_div],
            'Delivery_Pain_Index':[delivery_pain],
            'Tip_to_Order_Ratio':[tip_ratio]

        })
        st.write("Dashboard Input Row:")
        st.write(new_customer)
        prob = model.predict_proba(new_customer)[0][1]    
        st.subheader("Prediction Result")

        st.metric("Upgrade Probability", f"{round(prob*100,2)}%")

        if prob > 0.7:
            st.success("High upgrade likelihood")

        elif prob > 0.4:
            st.warning("Moderate likelihood")

        else:
            st.error("Low likelihood")
