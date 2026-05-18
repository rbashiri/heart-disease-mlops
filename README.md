# Heart Disease MLOps Project

## Project Overview

This project builds an end-to-end MLOps pipeline for predicting heart disease using the UCI Heart Disease dataset. The project demonstrates practical machine learning engineering workflows including preprocessing, model training, testing, experiment tracking, CI/CD automation, DVC data versioning, and drift monitoring.

The final machine learning task was converted into a binary classification problem:
- 0 = No heart disease
- 1 = Heart disease present

The project compares multiple machine learning models and automatically tracks experiments using MLflow.

---

## Project Structure

```text
heart-disease-mlops/
│
├── .github/workflows/
│   └── ci.yml
│
├── configs/
│   └── experiments.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── reports/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── monitor_drift.py
│   └── compare_experiments.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_data_validation.py
│
├── requirements.txt
├── README.md
└── .gitignore
==========================================================================================================================
# Dataset:
UCI Heart Disease Dataset

The raw dataset is tracked using DVC and is not committed directly to Git.
======================================================================================================================
Data Preprocessing

Run preprocessing: python src/preprocessing.py
This script:
cleans text columns
handles missing values
removes duplicates
saves cleaned dataset
=======================================================================================================================================
Model Training

Run training:

python src/train.py

Training includes:

Logistic Regression
Random Forest
MLflow experiment tracking
model comparison
best model selection

Saved model: models/best_model.pkl
=========================================================================================================================================================
Experiment Tracking (MLflow)

Start MLflow UI:

mlflow ui

Open browser:

http://127.0.0.1:5000

MLflow logs:
hyperparameters
metrics
model artifacts
experiment comparisons
=====================================================================================================================================================
Model Evaluation

Run evaluation:

python src/evaluate.py

Metrics include:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
====================================================================================================================================================
Drift Monitoring

Run drift monitoring:

python src/monitor_drift.py

The script:

compares reference and simulated production data
detects feature drift using Evidently
generates HTML drift reports
exits with code 1 if drift exceeds threshold

Reports are saved in:

reports/
Drift Analysis

The drift monitoring script detected drift above the configured threshold. The drift share was approximately 0.77, meaning many monitored features showed distribution changes between the reference and simulated production datasets.

This drift was intentionally introduced by modifying age and cholesterol distributions in the simulated production dataset.

This level of drift could affect model performance because the production data distribution differs from the training data. The recommended action is to investigate the source of drift and retrain the model if similar drift occurs in real production environments.
========================================================================================================================================================
Testing

Run all tests:

PYTHONPATH=. pytest tests/ -v

The test suite includes:
preprocessing unit tests
data validation tests
model validation tests
==========================================================================================================================================================
CI/CD Pipeline

GitHub Actions automatically:
runs tests
trains the model
validates performance thresholds

The pipeline triggers on:
pushes to main
pull requests to main
=============================================================================================================================================================
DVC Data Versioning
Initialize DVC:
dvc pull
The dataset is tracked using DVC rather than Git.
===============================================================================================================================================================
Technologies Used
Python
pandas
scikit-learn
pytest
DVC
MLflow
Evidently
GitHub Actions
=============================================================================================================================================================
## Final Results and Findings
The final machine learning pipeline successfully predicted heart disease using a binary classification approach based on the UCI Heart Disease dataset.

Two machine learning models were evaluated:
- Logistic Regression
- Random Forest Classifier

The best-performing model was the Random Forest Classifier with the following performance:

- Accuracy: approximately 0.86
- Precision: approximately 0.86
- Recall: approximately 0.89
- F1-score: approximately 0.88

The project also demonstrated a complete MLOps workflow including:
- data preprocessing
- automated testing with pytest
- DVC data versioning
- MLflow experiment tracking
- GitHub Actions CI/CD pipeline
- drift monitoring using Evidently

Five MLflow experiments were conducted using different hyperparameter configurations and model types. The best experiment was automatically identified using a comparison script based on F1-score performance.

Drift monitoring detected significant simulated drift between the reference dataset and production dataset. Approximately 77% of monitored features showed distribution drift after intentionally modifying selected variables such as age and cholesterol. This result demonstrates how monitoring tools can detect changes in incoming production data that may affect model performance over time.

The final project provides a reproducible and production-oriented machine learning workflow suitable for demonstrating practical MLOps concepts.


Author
Robabeh Bashiri