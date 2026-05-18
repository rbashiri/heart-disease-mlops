import pandas as pd


# 1. Load cleaned dataset
df = pd.read_csv("data/processed/cleaned_heart_disease.csv")


# 2. Remove ID column if it exists
if "id" in df.columns:
    df = df.drop("id", axis=1)


# 3. Convert target to binary classification
# 0 = no heart disease
# 1 = heart disease present
df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

# Remove original target
df = df.drop("num", axis=1)


# 4. Convert boolean columns to integers
bool_cols = df.select_dtypes(include=["bool"]).columns

for col in bool_cols:
    df[col] = df[col].astype(int)


# 5. One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)


# 6. Split dataset into reference and current datasets
middle_index = len(df) // 2

reference_data = df.iloc[:middle_index]
current_data = df.iloc[middle_index:]


# 7. Calculate feature mean differences
drift_results = {}

for column in df.columns:

    if column == "target":
        continue

    reference_mean = reference_data[column].mean()
    current_mean = current_data[column].mean()

    mean_difference = abs(reference_mean - current_mean)

    drift_results[column] = mean_difference


# 8. Convert drift results to DataFrame
drift_df = pd.DataFrame(
    drift_results.items(),
    columns=["feature", "mean_difference"]
)

drift_df = drift_df.sort_values(
    by="mean_difference",
    ascending=False
)


# 9. Print drift report
print("\nData Drift Report")
print(drift_df.head(10))


# 10. Set drift threshold
threshold = 0.10

drifted_features = drift_df[
    drift_df["mean_difference"] > threshold
]


# 11. Print drift warnings
print("\nFeatures with possible drift:")
print(drifted_features)


# 12. Final status
if len(drifted_features) > 0:
    print("\nWARNING: Possible data drift detected.")
else:
    print("\nNo major data drift detected.")

import os

# 13. Create reports folder
os.makedirs("reports", exist_ok=True)


# 14. Save drift report
drift_df.to_csv(
    "reports/drift_report.csv",
    index=False
)

print("\nDrift report saved to reports/drift_report.csv")