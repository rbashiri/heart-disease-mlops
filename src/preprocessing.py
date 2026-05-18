# =========================
# 1. Import Libraries
# =========================

import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# =========================
# 2. Preprocessing Functions
# =========================

def validate_dataframe(df):
    """Validate that input is a pandas DataFrame."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input DataFrame cannot be empty.")

    return True


def clean_text_columns(df):
    """Clean text columns without modifying original dataframe."""

    validate_dataframe(df)

    cleaned_df = df.copy()

    cat_cols = cleaned_df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        cleaned_df[col] = cleaned_df[col].astype(str)
        cleaned_df[col] = cleaned_df[col].str.lower()
        cleaned_df[col] = cleaned_df[col].str.strip()

    cleaned_df = cleaned_df.replace("nan", np.nan)

    return cleaned_df


def handle_missing_values(df):
    """Fill missing values without modifying original dataframe."""

    validate_dataframe(df)

    cleaned_df = df.copy()

    cleaned_df = cleaned_df.replace({pd.NA: np.nan})

    num_cols = cleaned_df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())

    cat_cols = cleaned_df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

    return cleaned_df


def encode_categorical_variables(df):
    """Encode categorical variables without modifying original dataframe."""

    validate_dataframe(df)

    encoded_df = df.copy()

    encoded_df = pd.get_dummies(encoded_df, drop_first=True)

    return encoded_df


def remove_duplicates(df):
    """Remove duplicate rows without modifying original dataframe."""

    validate_dataframe(df)

    cleaned_df = df.copy()

    cleaned_df = cleaned_df.drop_duplicates()

    return cleaned_df


# =========================
# 3. Main Preprocessing Pipeline
# =========================

if __name__ == "__main__":

    # Load raw dataset
    df = pd.read_csv("data/raw/heart_disease_uci.csv")

    print("First 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nDataset information:")
    print(df.info())

    print("\nData types:")
    print(df.dtypes)

    print("\nSummary statistics:")
    print(df.describe(include="all"))

    # Missing data summary before cleaning
    missing_summary = pd.DataFrame({
        "missing_count": df.isnull().sum(),
        "missing_percent": (df.isnull().sum() / len(df)) * 100
    }).sort_values(by="missing_percent", ascending=False)

    print("\nMissing Data Summary:")
    print(missing_summary)

    print("\nDuplicate rows before cleaning:")
    print(df.duplicated().sum())

    print("\nCategorical Columns:")
    print(df.select_dtypes(include=["object"]).columns)

    # Clean text columns
    df = clean_text_columns(df)

    # Inspect unique values
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        print(f"\nUnique values in {col}:")
        print(df[col].unique())

    # Remove duplicate rows
    df = remove_duplicates(df)

    print("\nDuplicate rows after cleaning:")
    print(df.duplicated().sum())

    print("\nDataset shape after removing duplicates:")
    print(df.shape)

    print("\nNumerical Columns:")
    print(df.select_dtypes(include=["int64", "float64"]).columns)

    # Fill missing values
    df = handle_missing_values(df)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    print("\nTotal missing values after cleaning:")
    print(df.isnull().sum().sum())

    # Save cleaned dataset
    os.makedirs("data/processed", exist_ok=True)

    df.to_csv(
        "data/processed/cleaned_heart_disease.csv",
        index=False
    )

    print("\nCleaned dataset saved successfully.")