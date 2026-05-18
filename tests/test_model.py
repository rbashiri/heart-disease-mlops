import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def prepare_sample_data():
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


def test_model_predictions_shape_and_type():
    X_train, X_test, y_train, y_test = prepare_sample_data()

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert predictions.dtype.kind in "iu"


def test_model_minimum_performance_threshold():
    X_train, X_test, y_train, y_test = prepare_sample_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.75
