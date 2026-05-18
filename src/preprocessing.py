# =========================
# =========================
# 1. Import Libraries
# =========================

import os
import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")


# =========================
# 2. Load Data
# =========================

df = pd.read_csv("data/raw/heart_disease_uci.csv")


# =========================
# 3. Basic Data Inspection
# =========================

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


# =========================
# 4. Missing Data Check
# =========================

missing_summary = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": (df.isnull().sum() / len(df)) * 100
}).sort_values(by="missing_percent", ascending=False)

print("\nMissing Data Summary:")
print(missing_summary)


# =========================
# 5. Duplicate Check
# =========================

print("\nDuplicate rows before cleaning:")
print(df.duplicated().sum())


# =========================
# 6. Identify Categorical Columns
# =========================

cat_cols = df.select_dtypes(include=["object"]).columns

print("\nCategorical Columns:")
print(cat_cols)


# =========================
# 7. Clean Text Columns
# =========================

for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.lower()
    df[col] = df[col].str.strip()


# =========================
# 8. Replace string 'nan' back with real missing values
# =========================

df = df.replace("nan", pd.NA)


# =========================
# 9. Inspect Unique Values
# =========================

for col in cat_cols:
    print(f"\nUnique values in {col}:")
    print(df[col].unique())


# =========================
# 10. Remove Duplicate Rows
# =========================

df = df.drop_duplicates()

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nDataset shape after removing duplicates:")
print(df.shape)


# =========================
# 11. Identify Numerical Columns
# =========================

num_cols = df.select_dtypes(include=["int64", "float64"]).columns

print("\nNumerical Columns:")
print(num_cols)


# =========================
# 12. Fill Missing Values
# =========================

# Replace pandas NA with numpy NaN
df = df.replace({pd.NA: np.nan})

# Numerical columns -> median
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())


# Categorical columns -> most frequent value
cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# =========================
# 13. Confirm Missing Values Removed
# =========================

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nTotal missing values after cleaning:")
print(df.isnull().sum().sum())


# =========================
# 14. Save Cleaned Dataset
# =========================

os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/cleaned_heart_disease.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")