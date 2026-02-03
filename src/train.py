import os
import pickle
import yaml
import json
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
import mlflow
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns'))
mlflow.set_experiment("diabetes_classification")

with open('params.yaml') as f:
    params = yaml.safe_load(f)
train_params = params['train']
tuning_params = params['tuning']

def hyperparameter_tuning(X_train, y_train, param_grid, random_state=42):
    clf = RandomForestClassifier(random_state=random_state)
    random_search = RandomizedSearchCV(
        estimator=clf,
        n_iter = 20,
        param_distributions=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
        random_state=random_state
    )
    random_search.fit(X_train, y_train)
    print("Best hyperparameters found:", random_search.best_params_)
    return random_search


def train(data_path, model_path, random_state):
    data = pd.read_csv(data_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    with mlflow.start_run() as run:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )

        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("cv_folds", 3)                   

        grid_search = hyperparameter_tuning(X_train, y_train, tuning_params, random_state)
        best_model = grid_search.best_estimator_
        
        for k, v in grid_search.best_params_.items():
            mlflow.log_param(k, v)
        
        signature = infer_signature(X_train, y_train)

        mlflow.sklearn.log_model(
            best_model, 
            "model", 
            signature=signature,
            registered_model_name="Best_model"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)       
        with open("models/mlflow_run_id.txt", "w") as f:
            f.write(run.info.run_id)
        

        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train(train_params['data'], train_params['models'], train_params['random_state'])