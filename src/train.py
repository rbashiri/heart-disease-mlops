# Step 1 — Understand the Target Column
# The target column is the column we want to predict. In this case, it is the "target" column in our dataset. We will use this column to train our model and evaluate its performance.
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_heart_disease.csv")


# 2. Remove ID column if it exists
if "id" in df.columns:
    df = df.drop("id", axis=1)


# 3. Convert original multiclass target to binary target
# 0 = no heart disease
# 1 = heart disease present
df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)


# 4. Drop original num column
df = df.drop("num", axis=1)


# 5. Convert boolean columns to 0/1
bool_cols = df.select_dtypes(include=["bool"]).columns

for col in bool_cols:
    df[col] = df[col].astype(int)


# 6. One-hot encode text/categorical columns
df = pd.get_dummies(df, drop_first=True)


# 7. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]


# 8. Check target balance
print("\nTarget distribution:")
print(y.value_counts())
print(y.value_counts(normalize=True))


# 9. Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 10. Logistic Regression model with scaling
lr_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ))
])

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred, zero_division=0))


# 11. Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred, zero_division=0))
import matplotlib.pyplot as plt
import pandas as pd

importance = rf_model.feature_importances_

feature_importance = pd.Series(
    importance,
    index=X.columns
).sort_values(ascending=False)


# Print top features
print("\nTop Features:")
print(feature_importance.head(10))


# Create plot
plt.figure(figsize=(10,6))

feature_importance.head(10).sort_values().plot(kind="barh")

plt.title("Top 10 Important Features")
plt.xlabel("Importance")


# Save figure
plt.savefig("feature_importance.png")

print("\nFeature importance plot saved as feature_importance.png")