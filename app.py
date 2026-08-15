import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="CreditWise | Enterprise Loan Approval Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Ultra-Professional Website Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Navbar / Header Styling */
    .brand-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .brand-header::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.25) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
    }

    .brand-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 6px 14px;
        border-radius: 30px;
        border: 1px solid rgba(96, 165, 250, 0.3);
        margin-bottom: 1rem;
    }

    .brand-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        background: linear-gradient(180deg, #FFFFFF 0%, #CBD5E1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }

    /* Section Cards */
    .input-card {
        background: #FFFFFF;
        padding: 1.8rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 1.5rem;
    }

    .dark-card-title {
        color: #0F172A;
        font-size: 1.25rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #F1F5F9;
    }

    /* Custom Streamlit Button Styling */
    div.stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.85rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        letter-spacing: 0.3px;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 20px -3px rgba(37, 99, 235, 0.45) !important;
    }

    /* Prediction Banner Styling */
    .result-banner-approved {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.25);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .result-banner-rejected {
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.25);
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .result-status {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .result-prob {
        font-size: 1.1rem;
        opacity: 0.95;
    }

    .factor-tag {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .footer {
        text-align: center;
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loader ---
@st.cache_resource
def load_trained_model():
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

# Load model pipeline
pipeline = load_trained_model()

# --- Header Banner ---
st.markdown("""
<div class="brand-header">
    <div class="brand-badge">Enterprise ML Portal v2.4</div>
    <h1 class="brand-title">CreditWise Loan Assessment Engine</h1>
    <p class="brand-subtitle">AI-powered instant credit risk analysis and loan eligibility prediction</p>
</div>
""", unsafe_allow_html=True)

# --- Layout Columns ---
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown('<div class="dark-card-title">💵 Financial Parameters</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        applicant_income = st.number_input("Applicant Income ($/mo)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant Income ($/mo)", min_value=0, value=2000, step=500)
        savings = st.number_input("Total Savings ($)", min_value=0, value=10000, step=1000)
    with c2:
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, value=15000, step=1000)
        loan_term = st.number_input("Loan Tenure (Months)", min_value=6, max_value=360, value=36, step=6)
        collateral_value = st.number_input("Collateral Asset Value ($)", min_value=0, value=20000, step=1000)

    dti_ratio = st.slider("Debt-to-Income (DTI) Ratio (%)", min_value=0.0, max_value=1.0, value=0.35, step=0.01, help="Total monthly debt payments divided by gross monthly income")

    st.markdown('<div class="dark-card-title" style="margin-top: 1.5rem;">👤 Applicant & Credit Profile</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        credit_score = st.slider("Credit Score (FICO)", 300, 850, 680)
        age = st.slider("Applicant Age", 18, 80, 35)
        dependents = st.slider("Dependents Count", 0, 5, 1)
        existing_loans = st.slider("Active Existing Loans", 0, 6, 1)

    with c4:
        employment_status = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Contract", "Unemployed"])
        employer_category = st.selectbox("Employer Sector", ["Private", "Government", "MNC", "Business", "Unemployed"])
        marital_status = st.selectbox("Marital Status", ["Married", "Single"])
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])

    c5, c6 = st.columns(2)
    with c5:
        loan_purpose = st.selectbox("Loan Purpose", ["Home", "Car", "Personal", "Education", "Business"])
        property_area = st.selectbox("Property Zone", ["Urban", "Semiurban", "Rural"])
    with c6:
        gender = st.selectbox("Gender", ["Male", "Female"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Evaluate Eligibility & Predict Approval", use_container_width=True)

with col_right:
    st.markdown('<div class="dark-card-title">📊 Assessment Output & Analytics</div>', unsafe_allow_html=True)
    
    if predict_clicked:
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

        probability = float(pipeline.predict_proba(input_data)[0][1])
        prediction = 1 if probability >= 0.5 else 0

        if prediction == 1:
            st.markdown(f"""
            <div class="result-banner-approved">
                <div class="result-status">✅ LOAN APPROVED</div>
                <div class="result-prob">High Approval Confidence: <strong>{probability:.1%}</strong></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-banner-rejected">
                <div class="result-status">❌ APPLICATION REJECTED</div>
                <div class="result-prob">Approval Probability: <strong>{probability:.1%}</strong></div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(probability)
        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("💡 Key Decision Indicators")
        
        factors = []
        if credit_score >= 700:
            factors.append("🟢 Strong Credit Score (> 700 FICO)")
        elif credit_score < 580:
            factors.append("🔴 Elevated Credit Risk (< 580 FICO)")

        if dti_ratio <= 0.36:
            factors.append("🟢 Healthy Debt-to-Income Ratio (≤ 36%)")
        elif dti_ratio > 0.50:
            factors.append("🔴 High Debt Burden relative to income (> 50%)")

        total_income = applicant_income + coapplicant_income
        if total_income > 0 and (loan_amount / (total_income * 12)) < 3:
            factors.append("🟢 Favorable Income-to-Loan Coverage")
        elif total_income > 0 and (loan_amount / (total_income * 12)) > 5:
            factors.append("🔴 High Leverage (Loan Amount exceeds 5x annual income)")

        if savings >= loan_amount * 0.5:
            factors.append("🟢 Robust Liquidity Cushion")

        if factors:
            for factor in factors:
                st.markdown(f'<div class="factor-tag">{factor}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="factor-tag">ℹ️ Standard balanced risk indicators</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Enter applicant details on the left and click **Evaluate Eligibility & Predict Approval** to view the real-time AI credit assessment.")

        st.markdown("""
        <div style="background:#F8FAFC; padding:1.2rem; border-radius:12px; border:1px dashed #CBD5E1; margin-top:1rem;">
            <h5 style="margin-top:0; color:#475569;">🔒 Security & Intelligence Note</h5>
            <p style="font-size:0.85rem; color:#64748B; margin:0;">
                Predictions are dynamically processed using an automated Scikit-Learn Logistic Regression pipeline trained on loan applicant historical data. 100% confidential and secure.
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    CreditWise Enterprise Financial Portal &bull; Powered by Scikit-Learn Pipeline Architecture &bull; All Rights Reserved
</div>
""", unsafe_allow_html=True)