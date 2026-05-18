import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_heart_disease.csv")


# 2. Remove ID column if it exists
if "id" in df.columns:
    df = df.drop("id", axis=1)


# 3. Convert multiclass target to binary target
# 0 = no heart disease
# 1 = heart disease present
df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)


# 4. Drop original target column
df = df.drop("num", axis=1)


# 5. Convert boolean columns to 0/1
bool_cols = df.select_dtypes(include=["bool"]).columns

for col in bool_cols:
    df[col] = df[col].astype(int)


# 6. One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)


# 7. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]


# 8. Use same train/test split as train.py
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 9. Load saved model
model = joblib.load("models/random_forest_model.pkl")


# 10. Make predictions
y_pred = model.predict(X_test)


# 11. Evaluate model
print("\nModel Evaluation Results")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
import os
import json

os.makedirs("reports", exist_ok=True)

results = {
    "accuracy": accuracy_score(y_test, y_pred),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "classification_report": classification_report(
        y_test,
        y_pred,
        zero_division=0,
        output_dict=True
    )
}

with open("reports/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nEvaluation results saved to reports/evaluation_results.json")