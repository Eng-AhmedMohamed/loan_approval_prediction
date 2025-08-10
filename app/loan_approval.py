import streamlit as st
import joblib
import numpy as np

# Load trained model and scaler
model = joblib.load("loan_approval_rf_98.pkl")
scaler = joblib.load("loan_approval_scaler.pkl")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="💳", layout="centered")

st.title("💳 Loan Approval Prediction")
st.write("Fill in the applicant details below to check if the loan will be approved.")

# Sidebar inputs
st.sidebar.header("Applicant Information")

dependents = st.sidebar.number_input("Number of Dependents", min_value=0)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.sidebar.number_input("Annual Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Term (in months)", min_value=1)
cibil_score = st.sidebar.slider("CIBIL Score", min_value=300, max_value=900)
residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0)
luxury_assets_value = st.sidebar.number_input("Luxury Assets Value", min_value=0)
bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0)

# Encoding (same as in training)
education_map = {"Graduate": 1, "Not Graduate": 0}
self_emp_map = {"Yes": 1, "No": 0}

# Prepare input data
input_data = np.array([[
    dependents,
    education_map[education],
    self_emp_map[self_employed],
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
]])

# Apply MinMaxScaler
input_data_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Loan Status"):
    prediction = model.predict(input_data_scaled)[0]
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")
