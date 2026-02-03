# Diabetes Risk Pipeline

## About the Project

This project is a machine learning pipeline to **predict diabetes risk** using patient data (from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download)

It demonstrates a reproducible workflow using:

- **Python** for data processing and ML modeling
- **Scikit-learn** for **Random Forest classifier** and **RandomizedSearchCV (RSCV)** hyperparameter tuning
- **MLflow** for experiment tracking and model registry
- **DVC** for data and pipeline versioning
- **Dagshub** https://dagshub.com/andreeamnl/diabetes-risk-pipeline

The pipeline includes data preprocessing, model training with hyperparameter tuning via RSCV, and evaluation with metrics tracking.

## Data

The raw data file:

data/raw/diabetes.csv.dvc

## Pipeline Overview

```

+---------------------------+
| data/raw/diabetes.csv.dvc |
+---------------------------+
               *
               *
               *
        +------------+
        | preprocess |
        +------------+
         **        **
       **            **
      *                **
+-------+                *
| train |              **
+-------+            **
         **        **
           **    **
             *  *
         +----------+
         | evaluate |
         +----------+
```

## Stages

### Preprocess

- Input: data/raw/diabetes.csv
- Output: data/processed/data.csv
- Script: src/preprocess.py

### Train

- Input: data/processed/data.csv
- Model output: models/model.pkl
- Script: src/train.py
- Parameters:
  random_state: 42
  n_estimators: 100
  max_depth: 5

- Hyperparameter tuning options using **RandomizedSearchCV**:
  n_estimators: [50, 100, 150]
  max_depth: [5, 10, 15]
  min_samples_split: [2, 5]
  min_samples_leaf: [1, 2]
  max_features: ["sqrt", "log2"]

### Evaluate

- Input model: models/model.pkl
- Test data: data/processed/data.csv
- Metrics output to: metrics/accuracy.json, metrics/scores.json, metrics/confusion_matrix.json, metrics/classification_report.json
- Script: src/evaluate.py

## Metrics (example `dvc metrics diff` output)

```

Path Metric HEAD workspace Change
metrics/accuracy.json accuracy - 0.83963 -
metrics/classification_report.json 0.f1-score - 0.88275 -
metrics/classification_report.json 0.precision - 0.84335 -
metrics/classification_report.json 0.recall - 0.926 -
metrics/classification_report.json 0.support - 500.0 -
metrics/classification_report.json 1.f1-score - 0.74639 -
metrics/classification_report.json 1.precision - 0.83028 -
metrics/classification_report.json 1.recall - 0.6779 -
metrics/classification_report.json 1.support - 267.0 -
metrics/classification_report.json accuracy - 0.83963 -
metrics/classification_report.json macro avg.f1-score - 0.81457 -
metrics/classification_report.json macro avg.precision - 0.83681 -
metrics/classification_report.json macro avg.recall - 0.80195 -
metrics/classification_report.json macro avg.support - 767.0 -
metrics/classification_report.json weighted avg.f1-score - 0.83528 -
metrics/classification_report.json weighted avg.precision - 0.8388 -
metrics/classification_report.json weighted avg.recall - 0.83963 -
metrics/classification_report.json weighted avg.support - 767.0 -
metrics/scores.json accuracy - 0.83963 -
metrics/scores.json f1_score - 0.83528 -
metrics/scores.json precision - 0.8388 -
metrics/scores.json recall - 0.83963 -

```

## Notes

- Make sure folders exist before running DVC stages: data/processed, models, metrics.
- Use `dvc repro` to run the full pipeline.
- Use `dvc metrics show` or `dvc metrics diff` to track evaluation changes.
