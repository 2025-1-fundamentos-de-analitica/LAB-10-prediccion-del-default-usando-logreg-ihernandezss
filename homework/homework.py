
# flake8: noqa: E501
# Este script implementa un modelo de clasificación para predecir si un cliente incumplirá el pago de su crédito.
# Se trabaja con un conjunto de datos que ya está dividido en entrenamiento y prueba. Las variables explicativas son:

# LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE
# PAY_0 to PAY_6: Historial de pagos mensuales
# BILL_AMT1 to BILL_AMT6: Monto facturado en los últimos 6 meses
# PAY_AMT1 to PAY_AMT6: Monto pagado en los últimos 6 meses
# La variable objetivo es 'default payment next month'

# El flujo del modelo incluye:
# 1. Carga y limpieza de datos
# 2. División en X/y para entrenamiento y prueba
# 3. Pipeline: OneHotEncoder + MinMaxScaler + SelectKBest + LogisticRegression
# 4. Optimización de hiperparámetros con GridSearchCV (10 folds, balanced accuracy)
# 5. Almacenamiento del modelo y métricas en formato JSON

import gzip
import json
import os
import pickle
from typing import Dict, Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def load_data():
    train = pd.read_csv("files/input/train_data.csv.zip", compression="zip")
    test = pd.read_csv("files/input/test_data.csv.zip", compression="zip")
    return train, test

def clean_data(df):
    df = df.copy()
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns=["ID"], inplace=True)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return df.dropna()

def split(df):
    return df.drop(columns=["default"]), df["default"]

def build_pipeline():
    cat = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer([("cat", OneHotEncoder(), cat)], remainder=MinMaxScaler())
    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_regression)),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

def train_model(pipeline, x_train, y_train):
    param_grid = {
        "feature_selection__k": range(1, x_train.shape[1] + 1),
        "classifier__C": [0.1, 1, 10],
        "classifier__solver": ["liblinear", "lbfgs"],
    }
    search = GridSearchCV(pipeline, param_grid, scoring="balanced_accuracy", cv=10, refit=True, n_jobs=-1)
    return search.fit(x_train, y_train)

def save_model(estimator, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)

def compute_metrics(name, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

def compute_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": name,
        "true_0": {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1": {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }

def save_metrics(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

def main():
    train_df, test_df = load_data()
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)

    x_train, y_train = split(train_df)
    x_test, y_test = split(test_df)

    pipeline = build_pipeline()
    model = train_model(pipeline, x_train, y_train)
    save_model(model, "files/models/model.pkl.gz")

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    metrics = [
        compute_metrics("train", y_train, y_pred_train),
        compute_metrics("test", y_test, y_pred_test),
        compute_confusion("train", y_train, y_pred_train),
        compute_confusion("test", y_test, y_pred_test),
    ]
    save_metrics(metrics, "files/output/metrics.json")

if __name__ == "__main__":
    main()
