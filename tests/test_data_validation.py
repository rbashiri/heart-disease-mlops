import pandas as pd


def test_required_columns_exist():
    """Check that required columns exist."""

    df = pd.read_csv(
        "data/processed/cleaned_heart_disease.csv"
    )

    required_columns = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "thalch",
        "num"
    ]

    for col in required_columns:
        assert col in df.columns


def test_dataset_not_empty():
    """Check dataset is not empty."""

    df = pd.read_csv(
        "data/processed/cleaned_heart_disease.csv"
    )

    assert len(df) > 0


def test_target_values_valid():
    """Check target values are valid."""

    df = pd.read_csv(
        "data/processed/cleaned_heart_disease.csv"
    )

    valid_values = [0, 1, 2, 3, 4]

    assert df["num"].isin(valid_values).all()


def test_age_values_positive():
    """Check age values are positive."""

    df = pd.read_csv(
        "data/processed/cleaned_heart_disease.csv"
    )

    assert (df["age"] > 0).all()