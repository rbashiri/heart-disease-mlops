# =========================
# Import Libraries
# =========================

# Data handling
import pandas as pd
import numpy as np

# Train-test split
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Machine learning model
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
# =========================
# Load Data
# ========================= 
# Load the dataset
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

missing_count = df.isnull().sum()

missing_percent = (df.isnull().sum() / len(df)) * 100

missing_summary = pd.DataFrame({
    "missing_count": missing_count,
    "missing_percent": missing_percent
})

missing_summary = missing_summary.sort_values(
    by="missing_percent",
    ascending=False
)

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

cat_cols = df.select_dtypes(include="object").columns

print("\nCategorical Columns:")
print(cat_cols)


# =========================
# 7. Clean Text Columns
# =========================

for col in cat_cols:

    # Convert values to string first
    df[col] = df[col].astype(str)

    # Convert text to lowercase
    df[col] = df[col].str.lower()

    # Remove extra spaces
    df[col] = df[col].str.strip()


# =========================
# 8. Inspect Unique Values
# =========================

for col in cat_cols:

    print(f"\nUnique values in {col}:")
    print(df[col].unique())


# =========================
# 9. Replace Incorrect Values
# =========================
# Add replacements only if needed

# Example:
# df["sex"] = df["sex"].replace({
#     "male ": "male"
# })


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

# Numerical columns -> median

num_imputer = SimpleImputer(strategy="median")

df[num_cols] = num_imputer.fit_transform(df[num_cols])


# Categorical columns -> most frequent

cat_imputer = SimpleImputer(strategy="most_frequent")

df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])


# =========================
# 13. Confirm Missing Values Removed
# =========================

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# =========================
# 14. Save Cleaned Dataset
# =========================

df.to_csv(
    "data/processed/cleaned_heart_disease.csv",
    index=False)

print("\nCleaned dataset saved successfully.")




