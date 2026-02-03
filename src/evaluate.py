import os
import pickle
import yaml
import json
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'))

with open('params.yaml') as f:
    params = yaml.safe_load(f)
evaluate_params = params['evaluate']

def evaluate(model_path, test_data_path, metrics_path):
    print(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loading test data from {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
    print(f"Test set size: {len(X_test)} samples")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    os.makedirs(metrics_path, exist_ok=True)
    
    with open(f"{metrics_path}/accuracy.json", "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=2)
    
    with open(f"{metrics_path}/scores.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }, f, indent=2)
    
    with open(f"{metrics_path}/confusion_matrix.json", "w") as f:
        json.dump({"confusion_matrix": cm.tolist()}, f, indent=2)
    
    with open(f"{metrics_path}/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Metrics saved to {metrics_path}")
    
    # Log în MLflow (opțional - doar dacă vrei să asociezi cu run-ul de train)
    # Poți să citești run_id din models/mlflow_run_id.txt și să continui run-ul
    # try:
    #     with open("models/mlflow_run_id.txt", "r") as f:
    #         run_id = f.read().strip()
        
    #     with mlflow.start_run(run_id=run_id):
    #         mlflow.log_metric("test_accuracy", accuracy)
    #         mlflow.log_metric("test_precision", precision)
    #         mlflow.log_metric("test_recall", recall)
    #         mlflow.log_metric("test_f1_score", f1)
    #         mlflow.log_dict({"confusion_matrix": cm.tolist()}, "metrics/confusion_matrix.json")
    #         mlflow.log_dict(report, "metrics/classification_report.json")
    # except FileNotFoundError:
    #     print("Warning: Could not find MLflow run_id. Skipping MLflow logging.")

if __name__ == "__main__":
    evaluate(
        model_path=evaluate_params['model'],
        test_data_path=evaluate_params['test_data'],
        metrics_path=evaluate_params['metrics']
    )