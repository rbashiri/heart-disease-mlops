import os
import sys
import json
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


DRIFT_THRESHOLD = 0.40


def prepare_data():
    """Load and prepare cleaned dataset for drift monitoring."""

    df = pd.read_csv("data/processed/cleaned_heart_disease.csv")

    if "id" in df.columns:
        df = df.drop("id", axis=1)

    df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

    df = df.drop("num", axis=1)

    df = pd.get_dummies(df, drop_first=True)

    return df


def create_production_data(current_data):
    """Create simulated production data with intentional drift."""

    production_data = current_data.copy()

    if "age" in production_data.columns:
        production_data["age"] = production_data["age"] + 5

    if "chol" in production_data.columns:
        production_data["chol"] = production_data["chol"] * 1.10

    return production_data


def create_html_report(drift_share, drifted_features):
    """Create a simple HTML report."""

    html_path = "reports/data_drift_report.html"

    with open(html_path, "w") as file:
        file.write("<html><body>")
        file.write("<h1>Data Drift Report</h1>")
        file.write(f"<p><strong>Drift Share:</strong> {drift_share}</p>")
        file.write("<h2>Drifted Features</h2>")
        file.write("<ul>")

        for feature in drifted_features:
            file.write(f"<li>{feature}</li>")

        file.write("</ul>")
        file.write("</body></html>")

    print(f"\nHTML report saved to {html_path}")


def extract_drift_information(result):
    """Extract drift share and drifted feature names safely."""

    dataset_drift_metric = result["metrics"][0]["result"]

    drift_share = dataset_drift_metric.get(
        "share_of_drifted_columns",
        0
    )

    drifted_features = []

    # Evidently version option 1
    if "drift_by_columns" in dataset_drift_metric:
        for feature, values in dataset_drift_metric["drift_by_columns"].items():
            if values.get("drift_detected") is True:
                drifted_features.append(feature)

    # Evidently version option 2
    elif "drift_by_columns_names" in dataset_drift_metric:
        drifted_features = dataset_drift_metric["drift_by_columns_names"]

    # Evidently version option 3
    elif "number_of_drifted_columns" in dataset_drift_metric:
        drifted_features = ["See JSON report for feature-level drift details"]

    return drift_share, drifted_features


def main():
    """Run Evidently drift detection and save reports."""

    os.makedirs("reports", exist_ok=True)

    df = prepare_data()

    split_index = int(len(df) * 0.7)

    reference_data = df.iloc[:split_index]
    current_data = df.iloc[split_index:]

    production_data = create_production_data(current_data)

    report = Report(metrics=[
        DataDriftPreset()
    ])

    report.run(
        reference_data=reference_data,
        current_data=production_data
    )

    report.save_html("reports/evidently_data_drift_report.html")

    result = report.as_dict()

    with open("reports/data_drift_results.json", "w") as file:
        json.dump(result, file, indent=4)

    drift_share, drifted_features = extract_drift_information(result)

    print("\nData Drift Summary")
    print("Drift share:", drift_share)

    print("\nDrifted Features:")

    if len(drifted_features) == 0:
        print("No drifted features detected.")
    else:
        for feature in drifted_features:
            print("-", feature)

    create_html_report(drift_share, drifted_features)

    if drift_share > DRIFT_THRESHOLD:
        print("\nWARNING: Drift exceeds threshold.")
        sys.exit(1)

    print("\nDrift is below threshold.")
    sys.exit(0)


if __name__ == "__main__":
    main()