import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_path = "models/random_forest_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Function to preprocess input data
def preprocess_input(data):
    # Drop unnecessary columns if present
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    for col in drop_cols:
        if col in data.columns:
            data = data.drop(columns=[col])

    # Encode categorical variables
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    return data

# Streamlit UI
st.title("Customer Churn Prediction")
st.write("This app predicts whether a customer will churn or not based on input features.")

# Option 1: Manually input customer details
st.sidebar.header("Enter Customer Details")

# Define input fields
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
age = st.sidebar.slider("Age", 18, 100, 40)
tenure = st.sidebar.slider("Tenure (Years)", 0, 10, 5)
balance = st.sidebar.number_input("Balance", 0.0, 300000.0, 50000.0)
num_of_products = st.sidebar.slider("Number of Products", 1, 4, 1)
has_cr_card = st.sidebar.selectbox("Has Credit Card?", [0, 1])
is_active_member = st.sidebar.selectbox("Is Active Member?", [0, 1])
estimated_salary = st.sidebar.number_input("Estimated Salary", 0.0, 200000.0, 100000.0)

# Encode categorical fields
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])

# Convert gender and geography to numerical format
gender = 1 if gender == "Male" else 0
geography_map = {"France": 0, "Germany": 1, "Spain": 2}
geography = geography_map[geography]

# Create a DataFrame from user input
user_input = pd.DataFrame([[credit_score, geography, gender, age, tenure, balance, num_of_products,
                            has_cr_card, is_active_member, estimated_salary]],
                          columns=["CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
                                   "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"])

# Predict manually entered customer data
if st.sidebar.button("Predict"):
    prediction = model.predict(user_input)
    st.sidebar.write(f"Prediction: {'Churn' if prediction[0] == 1 else 'No Churn'}")

# Option 2: Upload a CSV file
st.subheader("Upload CSV File for Bulk Predictions")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    processed_data = preprocess_input(data)
    
    # Predict
    predictions = model.predict(processed_data)
    data["Churn Prediction"] = ["Churn" if pred == 1 else "No Churn" for pred in predictions]
    
    # Display results
    st.write("Predictions:", data.head())

    # Download results
    st.download_button("Download Predictions", data.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
