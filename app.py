import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# ---------------- Page Config ----------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")

# ---------------- Load Dataset ----------------
df = pd.read_csv("customer_churn.csv")

# ---------------- Load Models ----------------
with open("model.pkl", "rb") as f:
    ml_model = pickle.load(f)
'''
with open("ann_model.pkl", "rb") as file:
    ann_model = pickle.load(file)
'''

# ---------------- Load Scaler ----------------
scaler = None
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    pass

feature_names = ml_model.feature_names_in_

# ---------------- Dataset Preview ----------------
with st.expander("📂 View Dataset"):
    st.dataframe(df.head(100), use_container_width=True)

# ---------------- Input Form ----------------
st.subheader("📝 Enter Customer Information")

inputs = {}
FEATURE_COLS = [c for c in df.columns if c not in ["Churn", "customerID", "TotalCharges"]]

with st.form("churn_form"):
    cols = st.columns(3)
    i = 0

    for col in FEATURE_COLS:
        with cols[i % 3]:
            if col.lower() == "gender":
                inputs[col] = st.selectbox(col, ["Female", "Male"])
            elif col == "SeniorCitizen":
                inputs[col] = st.selectbox(col, [0, 1])
            elif col == "tenure":
                inputs[col] = st.number_input(col, 0, 72, 12)
            elif col == "InternetService":
                inputs[col] = st.selectbox(col, ["DSL", "Fiber optic", "No"])
            elif col == "Contract":
                inputs[col] = st.selectbox(col, ["Month-to-month", "One year", "Two year"])
            elif col == "PaymentMethod":
                inputs[col] = st.selectbox(
                    col,
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ]
                )
            elif df[col].dtype == "object":
                inputs[col] = st.selectbox(col, ["No", "Yes"])
            else:
                inputs[col] = st.number_input(
                    col,
                    float(df[col].min()),
                    float(df[col].max()),
                    float(df[col].mean())
                )
        i += 1

    submitted_ml = st.form_submit_button("🔮 Predict with ML Model")
    submitted_ann = st.form_submit_button("🧠 Predict with ANN Model")

# ---------------- Input Processing Function ----------------
def prepare_input_df(inputs, feature_names, scaler):
    inputs["TotalCharges"] = inputs["tenure"] * inputs["MonthlyCharges"]

    input_df = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    for col in numeric_cols:
        if col in input_df.columns:
            input_df[col] = inputs[col]

    if "gender_Male" in input_df.columns:
        input_df["gender_Male"] = 1 if inputs["gender"] == "Male" else 0

    for key in ["InternetService", "Contract", "PaymentMethod"]:
        dummy = f"{key}_{inputs[key]}"
        if dummy in input_df.columns:
            input_df[dummy] = 1

    for col, val in inputs.items():
        if val in ["Yes", "No"]:
            dummy = f"{col}_{val}"
            if dummy in input_df.columns:
                input_df[dummy] = 1

    if scaler is not None:
        input_df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
            input_df[["tenure", "MonthlyCharges", "TotalCharges"]]
        )

    return input_df

# ---------------- ML Model Prediction ----------------
if submitted_ml:
    input_df = prepare_input_df(inputs, feature_names, scaler)
    prob = ml_model.predict_proba(input_df)[0][1]

    st.subheader("📌 ML Model Prediction")
    if prob > 0.5:
        st.error(f"🚨 Customer Likely to Churn — {prob*100:.2f}%")
    else:
        st.success(f"✅ Customer Not Likely to Churn — {(1-prob)*100:.2f}%")

# ---------------- ANN Model Prediction ----------------
if submitted_ann:
    input_df = prepare_input_df(inputs, feature_names, scaler)
    prob = ann_model.predict(input_df)[0][0]

    st.subheader("🧠 ANN Model Prediction")
    if prob > 0.5:
        st.error(f"🚨 Customer Likely to Churn — {prob*100:.2f}%")
    else:
        st.success(f"✅ Customer Not Likely to Churn — {(1-prob)*100:.2f}%")

# ---------------- Full Dataset Prediction ----------------
st.subheader("🌐 Predict Churn for Entire Dataset")

if st.button("Predict All Customers (ML Model)"):
    df_all = df.drop(columns=["customerID"])
    df_all["TotalCharges"] = df_all["tenure"] * df_all["MonthlyCharges"]

    df_encoded = pd.get_dummies(
        df_all,
        columns=["gender", "InternetService", "Contract", "PaymentMethod"]
    )

    for col in feature_names:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[feature_names]

    if scaler is not None:
        df_encoded[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
            df_encoded[["tenure", "MonthlyCharges", "TotalCharges"]]
        )

    df_encoded["Churn_Probability"] = ml_model.predict_proba(df_encoded)[:, 1]
    df_encoded["Prediction"] = df_encoded["Churn_Probability"].apply(lambda x: "Churn" if x > 0.5 else "Stay")

    st.dataframe(df_encoded.head(100), use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df_encoded["Churn_Probability"], bins=30, kde=True, ax=ax)
    ax.set_title("Churn Probability Distribution")
    st.pyplot(fig)
