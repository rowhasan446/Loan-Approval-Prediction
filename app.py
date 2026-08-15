import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
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

# Helper function to convert local image file to base64
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

banner_path = os.path.join(os.path.dirname(__file__), "assets", "banner.png")
icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
banner_b64 = get_base64_of_bin_file(banner_path)
icon_b64 = get_base64_of_bin_file(icon_path)

# --- Custom CSS for Ultra-Professional Crisp White Styling & CSS Animations ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    /* Keyframe Animations */
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(12px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes pulseGlow {{
        0% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.4); }}
        70% {{ box-shadow: 0 0 0 15px rgba(37, 99, 235, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }}
    }}

    @keyframes floatIcon {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-6px); }}
        100% {{ transform: translateY(0px); }}
    }}

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .stApp {{
        background-color: #F8FAFC !important;
    }}

    /* Main Container Padding */
    .main .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1240px;
        animation: fadeIn 0.6s ease-out forwards;
    }}

    /* Header Banner Styling */
    .brand-header {{
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.88) 0%, rgba(30, 41, 59, 0.92) 100%), url('data:image/png;base64,{banner_b64}');
        background-size: cover;
        background-position: center;
        padding: 3rem 3rem;
        border-radius: 24px;
        color: white;
        box-shadow: 0 20px 30px -10px rgba(15, 23, 42, 0.25);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }}

    .brand-badge {{
        display: inline-block;
        background: #2563EB;
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 6px 16px;
        border-radius: 30px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4);
    }}

    .brand-title {{
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        color: #FFFFFF !important;
    }}

    .brand-subtitle {{
        color: #E2E8F0 !important;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }}

    .header-icon-img {{
        width: 110px;
        height: 110px;
        object-fit: contain;
        filter: drop-shadow(0 10px 15px rgba(0,0,0,0.3));
        animation: floatIcon 3.5s ease-in-out infinite;
    }}

    /* Clean White Card Wrappers */
    .white-card {{
        background: #FFFFFF !important;
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.04), 0 8px 10px -6px rgba(15, 23, 42, 0.02);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }}

    .white-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 16px 35px -5px rgba(15, 23, 42, 0.08);
    }}

    .card-title-text {{
        color: #0F172A !important;
        font-size: 1.3rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 1.2rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #F1F5F9;
    }}

    /* Ensure text labels inside inputs are dark & highly legible */
    label, p, div, span {{
        color: #1E293B;
    }}

    .stNumberInput label, .stSlider label, .stSelectbox label {{
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 0.92rem !important;
    }}

    /* Custom Streamlit Button Styling with Animation */
    div.stButton > button {{
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        padding: 0.95rem 2rem !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 10px 20px -3px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.3s ease !important;
        width: 100%;
        letter-spacing: 0.3px;
        animation: pulseGlow 3s infinite;
    }}

    div.stButton > button:hover {{
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 16px 25px -3px rgba(37, 99, 235, 0.5) !important;
    }}

    /* Prediction Banner Animation */
    .result-banner-approved {{
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: #FFFFFF;
        padding: 2.2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.3);
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.5s ease-in-out;
    }}

    .result-banner-rejected {{
        background: linear-gradient(135deg, #DC2626 0%, #EF4444 100%);
        color: #FFFFFF;
        padding: 2.2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(239, 68, 68, 0.3);
        margin-top: 1rem;
        margin-bottom: 1.5rem;
        animation: fadeIn 0.5s ease-in-out;
    }}

    .result-status {{
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
        letter-spacing: -0.5px;
    }}

    .result-prob {{
        font-size: 1.15rem;
        opacity: 0.95;
    }}

    .factor-tag {{
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0;
        padding: 0.85rem 1.1rem;
        border-radius: 12px;
        font-size: 0.95rem;
        font-weight: 600;
        color: #1E293B !important;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        display: flex;
        align-items: center;
        gap: 0.6rem;
        transition: transform 0.2s ease;
    }}

    .factor-tag:hover {{
        transform: translateX(4px);
    }}

    .footer {{
        text-align: center;
        color: #64748B;
        font-size: 0.88rem;
        font-weight: 500;
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
    }}
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

# --- Header Banner with Generated Image Asset ---
header_icon_html = f'<img src="data:image/png;base64,{icon_b64}" class="header-icon-img" alt="Fintech Icon">' if icon_b64 else '💳'

st.markdown(f"""
<div class="brand-header">
    <div>
        <div class="brand-badge">Enterprise ML Portal v2.4</div>
        <h1 class="brand-title">CreditWise Loan Assessment Engine</h1>
        <p class="brand-subtitle">AI-powered instant credit risk analysis & eligibility prediction</p>
    </div>
    <div>
        {header_icon_html}
    </div>
</div>
""", unsafe_allow_html=True)

# --- Layout Columns ---
col_left, col_right = st.columns([1.15, 0.85], gap="large")

with col_left:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title-text">💵 Financial Parameters</div>', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title-text">👤 Applicant & Credit Profile</div>', unsafe_allow_html=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡ Evaluate Eligibility & Predict Approval", use_container_width=True)

with col_right:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title-text">📊 Assessment Analytics</div>', unsafe_allow_html=True)
    
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

        st.markdown("<h4 style='color:#0F172A; font-weight:700;'>💡 Key Decision Indicators</h4>", unsafe_allow_html=True)
        
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
        st.info("👈 Enter applicant details on the left and click **Evaluate Eligibility** to run real-time AI credit scoring.")

        st.markdown("""
        <div style="background:#FFFFFF; padding:1.2rem; border-radius:14px; border:1px solid #E2E8F0; margin-top:1rem; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
            <h5 style="margin-top:0; color:#0F172A; font-weight:700;">🔒 Security & Intelligence Note</h5>
            <p style="font-size:0.88rem; color:#475569; margin:0; line-height:1.5;">
                Predictions are dynamically computed using an automated Scikit-Learn Logistic Regression pipeline trained on historical loan data.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    CreditWise Enterprise Financial Portal &bull; Powered by Scikit-Learn Pipeline Architecture &bull; All Rights Reserved
</div>
""", unsafe_allow_html=True)