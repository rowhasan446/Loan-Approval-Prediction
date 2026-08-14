import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="CreditWise Loan Predictor", page_icon="💰", layout="centered")

@st.cache_resource
def load_trained_model():
    model_path = os.path.join(os.path.dirname(__file__), "loan_approval_pipeline.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    # Train pipeline on the dataset if pickled file is not present
    data_path = os.path.join(os.path.dirname(__file__), "loan_approval_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["Loan_Approved"])
    if "Applicant_ID" in df.columns:
        df = df.drop(columns=["Applicant_ID"])

    X = df.drop(columns=["Loan_Approved"])
    y = df["Loan_Approved"].map({"Yes": 1, "No": 0})

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    pipeline.fit(X, y)
    return pipeline

# Load model
pipeline = load_trained_model()

st.title("💰 CreditWise Loan Approval Predictor")
st.markdown("### Will your loan be **Approved** or **Rejected**?")
st.caption("Fill in the applicant details below to predict loan approval using our trained machine learning model.")

# --- Input Fields ---
st.subheader("💵 Financial Information")
col1, col2 = st.columns(2)

with col1:
    applicant_income = st.number_input("Applicant Income ($/month)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-applicant Income ($/month)", min_value=0, value=2000, step=500)
    savings = st.number_input("Total Savings ($)", min_value=0, value=10000, step=1000)
    collateral_value = st.number_input("Collateral Value ($)", min_value=0, value=20000, step=1000)

with col2:
    loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=15000, step=1000)
    loan_term = st.number_input("Loan Term (Months)", min_value=6, max_value=360, value=36, step=6)
    dti_ratio = st.slider("DTI Ratio (Debt-to-Income, %)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

st.subheader("👤 Applicant & Credit Profile")
col3, col4 = st.columns(2)

with col3:
    credit_score = st.slider("Credit Score", 300, 850, 680)
    age = st.slider("Age", 18, 80, 35)
    dependents = st.slider("Number of Dependents", 0, 5, 1)
    existing_loans = st.slider("Existing Loans", 0, 6, 1)

with col4:
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"])
    employer_category = st.selectbox("Employer Category", ["Private", "Government", "MNC", "Business", "Unemployed"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

col5, col6 = st.columns(2)
with col5:
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Personal", "Education", "Business"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
with col6:
    gender = st.selectbox("Gender", ["Male", "Female"])

st.markdown("---")

# --- Predict Button ---
if st.button("🔍 Predict Loan Approval", type="primary", use_container_width=True):
    # Construct input dataframe matching exact model feature names
    input_data = pd.DataFrame([{
        'Applicant_Income': applicant_income,
        'Coapplicant_Income': coapplicant_income,
        'Age': age,
        'Dependents': dependents,
        'Credit_Score': credit_score,
        'Existing_Loans': existing_loans,
        'DTI_Ratio': dti_ratio,
        'Loan_Amount': loan_amount,
        'Employment_Status': employment_status,
        'Marital_Status': marital_status,
        'Property_Area': property_area,
        'Education_Level': education,
        'Gender': gender,
        'Savings': savings,
        'Collateral_Value': collateral_value,
        'Loan_Term': loan_term,
        'Loan_Purpose': loan_purpose,
        'Employer_Category': employer_category
    }])

    # Predict probability using real ML pipeline
    probability = float(pipeline.predict_proba(input_data)[0][1])
    prediction = 1 if probability >= 0.5 else 0

    st.subheader("🎯 Prediction Result")

    # Display dynamic probability progress bar
    st.progress(probability, text=f"Approval Probability: {probability:.1%}")

    if prediction == 1:
        st.success(f"✅ **Loan Approved!** (Probability: {probability:.1%})")
    else:
        st.error(f"❌ **Loan Rejected** (Probability of approval: {probability:.1%})")

    # Key profile summary / breakdown
    st.markdown("#### 📋 Profile Overview & Risk Factors")
    factors = []
    if credit_score >= 700:
        factors.append("🟢 **Excellent Credit Score** (> 700)")
    elif credit_score < 580:
        factors.append("🔴 **Low Credit Score** (< 580)")

    if dti_ratio <= 0.36:
        factors.append("🟢 **Healthy Debt-to-Income (DTI) Ratio** (≤ 36%)")
    elif dti_ratio > 0.50:
        factors.append("🔴 **High Debt-to-Income Ratio** (> 50%)")

    total_income = applicant_income + coapplicant_income
    if total_income > 0 and (loan_amount / (total_income * 12)) < 3:
        factors.append("🟢 **Strong Income-to-Loan Ratio**")
    elif total_income > 0 and (loan_amount / (total_income * 12)) > 5:
        factors.append("🔴 **High Loan Amount relative to annual income**")

    if savings >= loan_amount * 0.5:
        factors.append("🟢 **Strong Savings Cushion**")

    if factors:
        for f in factors:
            st.markdown(f"- {f}")
    else:
        st.write("Balanced applicant profile.")

    st.caption("Note: Prediction is generated dynamically using a machine learning model trained on the project dataset.")