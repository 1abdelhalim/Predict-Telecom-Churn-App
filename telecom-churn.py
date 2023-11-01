import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained logistic regression model
logistic_model = joblib.load('logistic_model.pkl')

# Streamlit App
st.title("Customer Churn Prediction")

# Input features
st.sidebar.header("User Input Features")

# Create input widgets for user input
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", [0, 1])
phone_service = st.sidebar.selectbox("Phone Service", [0, 1])
dependents = st.sidebar.selectbox("Dependents", [0, 1])
tenure = st.sidebar.number_input("Tenure", min_value=0)
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0)
contract = st.sidebar.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
internet_service = st.sidebar.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
payment_method = st.sidebar.selectbox("Payment Method", ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])

# Preprocess user input
user_input = pd.DataFrame({
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'PhoneService': [phone_service],
    'Dependents': [dependents],
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'Contract_Month-to-month': [1 if contract == 'Month-to-month' else 0],
    'Contract_One year': [1 if contract == 'One year' else 0],
    'Contract_Two year': [1 if contract == 'Two year' else 0],
    'InternetService_DSL': [1 if internet_service == 'DSL' else 0],
    'InternetService_Fiber optic': [1 if internet_service == 'Fiber optic' else 0],
    'InternetService_No': [1 if internet_service == 'No' else 0],
    'PaymentMethod_Bank transfer (automatic)': [1 if payment_method == 'Bank transfer (automatic)' else 0],
    'PaymentMethod_Credit card (automatic)': [1 if payment_method == 'Credit card (automatic)' else 0],
    'PaymentMethod_Electronic check': [1 if payment_method == 'Electronic check' else 0],
    'PaymentMethod_Mailed check': [1 if payment_method == 'Mailed check' else 0]
})

# Button to make predictions
if st.button("Make Predictions"):
    # Make predictions
    prediction_logistic = logistic_model.predict(user_input)

    st.subheader("Prediction")
    if prediction_logistic[0] == 1:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is likely to stay.")