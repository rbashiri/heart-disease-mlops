import pandas as pd


def test_expected_columns_present():
    df = pd.read_csv("data/processed/cleaned_heart_disease.csv")

    expected_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "thalch",
        "num"
    ]

    for col in expected_columns:
        assert col in df.columns


def test_target_values_are_valid():
    df = pd.read_csv("data/processed/cleaned_heart_disease.csv")

    valid_values = [0, 1, 2, 3, 4]

    assert df["num"].isin(valid_values).all()


def test_numeric_features_within_expected_ranges():
    df = pd.read_csv("data/processed/cleaned_heart_disease.csv")

    assert df["age"].between(20, 100).all()
    assert df["trestbps"].between(80, 250).all()
    assert df["chol"].between(0, 700).all()
    assert df["thalch"].between(50, 250).all()