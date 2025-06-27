import streamlit as st
import pandas as pd
import joblib
import json

# Load model and features
model = joblib.load("models/best_stack.pkl")
with open("models/feature_columns.json") as f:
    feature_cols = json.load(f)

st.set_page_config(page_title="E-Commerce Cancellation Predictor", layout="centered")
st.title("🛍️ E-Commerce Cancellation Predictor")
st.markdown("Predict whether a transaction will likely be **cancelled** or **completed**.")

# Sidebar inputs
st.sidebar.header("Customer & Order Info")
recency = st.sidebar.slider("Recency (days since last purchase)", 0, 1000, 200)
frequency = st.sidebar.slider("Frequency (total orders)", 1, 50, 5)
monetary = st.sidebar.number_input("Monetary (total spend)", 0.0, 10000.0, 500.0)
avg_order_value = st.sidebar.number_input("Avg Order Value", 0.0, 1000.0, 100.0)
purchase_std = st.sidebar.number_input("Purchase Std Dev", 0.0, 1000.0, 50.0)
cancel_rate = st.sidebar.slider("Cancel Rate (historical)", 0.0, 1.0, 0.05)

st.sidebar.header("Invoice Info")
hour = st.sidebar.slider("Hour of Day", 0, 23, 14)
weekday = st.sidebar.slider("Weekday (0=Mon)", 0, 6, 2)
month = st.sidebar.slider("Month", 1, 12, 6)
product_diversity = st.sidebar.slider("Product Diversity", 1, 50, 5)
abs_quantity = st.sidebar.slider("Absolute Quantity", 1, 100, 5)
is_neg_qty = st.sidebar.selectbox("Negative Quantity", [0, 1])
is_high_value = st.sidebar.selectbox("High Value Order", [0, 1])

# Dummy encoded countries and categories
st.sidebar.header("Extras")
countries = [col for col in feature_cols if col.startswith("Country_")]
country_selected = st.sidebar.selectbox("Country", [c.replace("Country_", "") for c in countries])
cats = [col for col in feature_cols if col.startswith("MainCategory_")]
cat_selected = st.sidebar.selectbox("Main Product Category", [c.replace("MainCategory_", "") for c in cats])

# Create input dataframe
input_data = {
    "Recency": recency,
    "Frequency": frequency,
    "Monetary": monetary,
    "Hour": hour,
    "Weekday": weekday,
    "Month": month,
    "ProductDiversity": product_diversity,
    "AvgOrderValue": avg_order_value,
    "IsHighValue": is_high_value,
    "PurchaseStd": purchase_std,
    "CancelRate": cancel_rate,
    "AbsQuantity": abs_quantity,
    "IsNegativeQuantity": is_neg_qty
}

# Add dummy columns
for col in feature_cols:
    if col not in input_data:
        if col.startswith("Country_"):
            input_data[col] = 1 if col == f"Country_{country_selected}" else 0
        elif col.startswith("MainCategory_"):
            input_data[col] = 1 if col == f"MainCategory_{cat_selected}" else 0

# Convert to DataFrame
input_df = pd.DataFrame([input_data])[feature_cols]

# Prediction
if st.button("Predict Cancellation"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]
    
    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"❌ Likely to be Cancelled (probability: {prob:.2f})")
    else:
        st.success(f"✅ Likely to be Completed (probability: {1 - prob:.2f})")
