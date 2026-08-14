# 💰 CreditWise - Loan Approval Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning web application designed to predict whether a loan application will be **Approved** or **Rejected**. Built using **Python**, **Scikit-Learn**, and **Streamlit**, CreditWise processes applicant financial and demographic details to provide real-time probability estimates and risk factor insights.

---

## 📌 Project Overview

Loan approval prediction is a critical task in financial technology and banking. Manual evaluation of loan applications can be slow and subject to human bias. CreditWise automates this process using a trained Machine Learning model that analyzes 18 applicant features to deliver instant, data-driven loan decisions along with approval probability percentages.

### Key Highlights:
- **Interactive UI**: User-friendly input interface with real-time prediction output and progress meters.
- **Robust Data Pipeline**: Scikit-Learn `Pipeline` handling missing value imputation, feature scaling, and one-hot encoding without data leakage.
- **Risk & Factor Analysis**: Automated profile summary highlighting positive financial indicators (e.g., high credit score, healthy DTI ratio) and potential risk factors.
- **Streamlit Cloud Deployment**: Optimized for high performance and zero-config deployment on Streamlit Community Cloud using `@st.cache_resource`.

---

## 🛠️ Technology Stack

| Component | Technology / Library |
| :--- | :--- |
| **Programming Language** | Python 3.10+ |
| **Web Framework** | [Streamlit](https://streamlit.io/) |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) (`Pipeline`, `ColumnTransformer`, `LogisticRegression`) |
| **Data Manipulation** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Data Visualization & EDA** | [Seaborn](https://seaborn.pydata.org/), [Matplotlib](https://matplotlib.org/) |
| **Interactive Notebook** | Jupyter Notebook (`LoanApproval.ipynb`) |

---

## 📊 Dataset & Feature Description

The model is trained on a comprehensive financial dataset (`loan_approval_data.csv`) comprising 10,000 applicant records.

### Numerical Features:
- **Applicant Income ($/month)**: Primary applicant income.
- **Co-applicant Income ($/month)**: Additional household/partner income.
- **Credit Score**: Credit score ranging from 300 to 850.
- **Loan Amount ($)**: Requested loan amount.
- **Loan Term (Months)**: Duration of loan repayment.
- **Savings ($)**: Total liquid savings available.
- **Collateral Value ($)**: Pledged asset value.
- **DTI Ratio (Debt-to-Income, %)**: Total monthly debt payments divided by total monthly income.
- **Age**: Applicant age (18 to 80).
- **Dependents**: Number of financial dependents (0 to 5).
- **Existing Loans**: Count of current active loans (0 to 6).

### Categorical Features:
- **Employment Status**: `Salaried`, `Self-employed`, `Contract`, `Unemployed`
- **Employer Category**: `Private`, `Government`, `MNC`, `Business`, `Unemployed`
- **Loan Purpose**: `Home`, `Car`, `Personal`, `Education`, `Business`
- **Property Area**: `Urban`, `Semiurban`, `Rural`
- **Education Level**: `Graduate`, `Not Graduate`
- **Gender**: `Male`, `Female`
- **Marital Status**: `Married`, `Single`

### Target Variable:
- **`Loan_Approved`**: `Yes` (1) or `No` (0)

---

## 🤖 Machine Learning Workflow & Performance

The machine learning workflow follows standard industry best practices:

```
Raw Data (CSV) ➔ Data Cleaning ➔ Feature Preprocessing Pipeline ➔ Model Training ➔ Real-Time Web Prediction
```

### 1. Data Cleaning & Imputation
- Rows missing the target label (`Loan_Approved`) are removed.
- Numerical features are imputed using **Median** strategy (robust against financial outliers).
- Categorical features are imputed using **Mode** (most frequent value) strategy.

### 2. Feature Engineering & Preprocessing
- **Numerical Scaling**: `StandardScaler` standardizes numeric variables.
- **Categorical Encoding**: `OneHotEncoder` converts categorical features into binary vectors.

### 3. Model Benchmark Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **71.21%** | **0.6120** | **0.6143** | **0.6131** | **0.7912** |
| **Gradient Boosting** | 71.06% | 0.6100 | 0.6231 | 0.6165 | 0.7792 |
| **Random Forest** | 67.79% | 0.5890 | 0.4780 | 0.5276 | 0.7574 |

*Logistic Regression was selected for deployment due to its superior ROC-AUC score, stable probability calibration, fast inference, and high interpretability.*

---

## 🚀 Local Installation & Setup

To run CreditWise locally on your machine, follow these steps:

### Prerequisites:
Make sure you have **Python 3.10+** installed on your system.

### 1. Clone the Repository:
```bash
git clone https://github.com/rowhasan446/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

### 2. Install Required Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App:
```bash
python -m streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## ☁️ Deployment on Streamlit Cloud

This project is fully configured for automated deployment on **Streamlit Community Cloud**:

1. Push your latest code to your GitHub repository.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select your GitHub repository (`Loan-Approval-Prediction`).
4. Set the main file path to `app.py` and click **Deploy**.

---

## 📂 Repository File Structure

```
Loan-Approval-Prediction/
├── app.py                      # Main Streamlit Web Application
├── LoanApproval.ipynb          # Jupyter Notebook for EDA & Model Evaluation
├── loan_approval_data.csv      # Project Dataset (10,000 rows)
├── requirements.txt            # Python dependencies for deployment
└── README.md                   # Project Documentation
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
