import os
import joblib
import pandas as pd


def prepare_features():
    """Prepare features exactly the same way as train.py."""

    df = pd.read_csv("data/processed/cleaned_heart_disease.csv")

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

    df = df.drop("num", axis=1)

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("target", axis=1)

    return X


def test_model_file_exists():
    model_path = "models/random_forest_model.pkl"

    assert os.path.exists(model_path)


def test_model_can_load():
    model = joblib.load("models/random_forest_model.pkl")

    assert model is not None


def test_model_can_predict():
    model = joblib.load("models/random_forest_model.pkl")

    X = prepare_features()

    X = X.reindex(columns=model.feature_names_in_, fill_value=0)

    predictions = model.predict(X.head())

    assert len(predictions) == 5