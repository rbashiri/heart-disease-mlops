import os
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def prepare_data():
    df = pd.read_csv("data/processed/cleaned_heart_disease.csv")

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)
    df = df.drop("num", axis=1)

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("target", axis=1)
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def build_model(config):
    if config["model_type"] == "random_forest":
        return RandomForestClassifier(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            random_state=42,
            class_weight="balanced"
        )

    if config["model_type"] == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=config["max_iter"],
                class_weight=config["class_weight"],
                random_state=42
            ))
        ])

    raise ValueError("Unsupported model type")


with open("configs/experiments.yaml", "r") as file:
    config_file = yaml.safe_load(file)

data_version = config_file["data_version"]
experiments = config_file["experiments"]

mlflow.set_experiment("heart-disease-binary-classification")

X_train, X_test, y_train, y_test = prepare_data()

best_f1 = 0
best_model = None

for exp in experiments:
    with mlflow.start_run(run_name=exp["name"]):

        model = build_model(exp)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_params(exp)
        mlflow.log_param("data_version", data_version)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"\nExperiment: {exp['name']}")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

print("\nBest model saved to models/best_model.pkl")