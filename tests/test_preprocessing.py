import pandas as pd
import os
def test_cleaned_dataset_exists():
    """Check that cleaned dataset file exists."""

    file_path = "data/processed/cleaned_heart_disease.csv"

    assert os.path.exists(file_path)


def test_cleaned_dataset_not_empty():
    """Check that cleaned dataset is not empty."""

    df = pd.read_csv(
        "data/processed/cleaned_heart_disease.csv"
    )

    assert len(df) > 0


def test_target_column_exists():
    """Check that target column exists."""

    df = pd.read_csv(
        "data/processed/cleaned_heart_disease.csv"
    )

    assert "num" in df.columns


def test_no_missing_values():
    """Check that dataset has no missing values."""

    df = pd.read_csv(
        "data/processed/cleaned_heart_disease.csv"
    )

    assert df.isnull().sum().sum() == 0