import mlflow


mlflow.set_experiment("heart-disease-binary-classification")

runs = mlflow.search_runs(
    order_by=["metrics.f1_score DESC"]
)

best_run = runs.iloc[0]

print("\nBest MLflow Run")
print("Run ID:", best_run["run_id"])
print("Run Name:", best_run["tags.mlflow.runName"])
print("F1 Score:", best_run["metrics.f1_score"])
print("Accuracy:", best_run["metrics.accuracy"])
print("Precision:", best_run["metrics.precision"])
print("Recall:", best_run["metrics.recall"])
print("Model Type:", best_run["params.model_type"])
print("Data Version:", best_run["params.data_version"])
