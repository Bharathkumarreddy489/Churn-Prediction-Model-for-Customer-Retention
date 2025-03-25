import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Trained Model
model_path = "models/random_forest_model.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Train the model first.")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# Function to preprocess new data
def preprocess_data(data):
    if "RowNumber" in data.columns:
        data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    return data

# Load & Preprocess Test Data
data_path = "data/Churn_Modelling.csv"
new_data = pd.read_csv(data_path)
processed_data = preprocess_data(new_data)

# Extract Features
if "Exited" in processed_data.columns:
    X_test = processed_data.drop(columns=["Exited"])
else:
    X_test = processed_data

# Predict
predictions = model.predict(X_test[:5])  # Predicting only first 5 rows to check if it works

print("Model Prediction Output (First 5 Predictions):", predictions)
