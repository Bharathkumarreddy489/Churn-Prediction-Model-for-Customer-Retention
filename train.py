import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

# Load Data
data_path = "data/Churn_Modelling.csv"
data = pd.read_csv(data_path)

# Drop unnecessary columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Encode categorical variables
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Split Features and Target
X = data.drop(columns=["Exited"])
Y = data["Exited"]

# Compute Class Weights Dynamically
value_counts = dict(Y.value_counts(normalize=True))
class_weights = {key: sum(value_counts.values()) - value / sum(value_counts.values()) for key, value in value_counts.items()}

# Train-Test Split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=4)

# Define and Train RandomForest Model
model = RandomForestClassifier(
    n_estimators=200,
    min_samples_leaf=5,
    class_weight=class_weights,
    max_features="sqrt",
    n_jobs=-1,
    oob_score=True,
    random_state=4
)

model.fit(X_train, Y_train)

# Model Evaluation
Y_pred = model.predict(X_val)
accuracy = accuracy_score(Y_val, Y_pred)
roc_auc = roc_auc_score(Y_val, model.predict_proba(X_val)[:, 1])

# Print Metrics
print(f"Training Accuracy: {model.score(X_train, Y_train):.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Save Model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "random_forest_model.pkl")

with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved at: {model_path}")
