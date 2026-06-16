from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class AutoMLResult:
    problem_type: str
    best_model_name: str
    best_score: float
    test_rows: int
    leaderboard: pd.DataFrame
    feature_importance: pd.DataFrame | None
    target_column: str
    row_count: int
    column_count: int


def detect_problem_type(y: pd.Series) -> str:
    if y.dtype == "object" or y.dtype.name == "category" or y.dtype == "bool":
        return "classification"
    unique_ratio = y.nunique(dropna=True) / max(len(y), 1)
    if y.nunique(dropna=True) <= 20 and unique_ratio < 0.1:
        return "classification"
    return "regression"


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def candidate_models(problem_type: str) -> dict[str, object]:
    if problem_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        }
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
    }


def score_classification(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    average = "binary" if pd.Series(y_true).nunique() == 2 else "weighted"
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "f1": f1_score(y_true, predictions, average=average, zero_division=0),
        "precision": precision_score(y_true, predictions, average=average, zero_division=0),
        "recall": recall_score(y_true, predictions, average=average, zero_division=0),
    }


def score_regression(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    rmse = mean_squared_error(y_true, predictions, squared=False)
    return {
        "r2": r2_score(y_true, predictions),
        "rmse": rmse,
        "negative_rmse": -rmse,
    }


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = get_feature_names(preprocessor)

    if not hasattr(model, "feature_importances_") or not feature_names:
        return None

    importances = model.feature_importances_
    rows = [
        {
            "feature": name.replace("numeric__", "").replace("categorical__", ""),
            "importance": float(value),
        }
        for name, value in zip(feature_names, importances)
    ]
    return pd.DataFrame(rows).sort_values("importance", ascending=False).head(15)


def run_automl(df: pd.DataFrame, target_column: str) -> AutoMLResult:
    cleaned = df.dropna(subset=[target_column]).copy()
    if target_column not in cleaned.columns:
        raise ValueError(f"Target column `{target_column}` was not found.")
    if len(cleaned) < 10:
        raise ValueError("Dataset must contain at least 10 usable rows.")

    x = cleaned.drop(columns=[target_column])
    y = cleaned[target_column]
    problem_type = detect_problem_type(y)

    stratify = y if problem_type == "classification" and y.nunique() > 1 else None
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42, stratify=stratify
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42
        )

    rows = []
    trained_pipelines: dict[str, Pipeline] = {}
    primary_metric = "f1" if problem_type == "classification" else "r2"

    for name, model in candidate_models(problem_type).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(x_train)),
                ("model", model),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = (
            score_classification(y_test, predictions)
            if problem_type == "classification"
            else score_regression(y_test, predictions)
        )
        rows.append({"model": name, **metrics})
        trained_pipelines[name] = pipeline

    leaderboard = pd.DataFrame(rows).sort_values(primary_metric, ascending=False)
    best_model_name = str(leaderboard.iloc[0]["model"])
    best_score = float(leaderboard.iloc[0][primary_metric])
    feature_importance = extract_feature_importance(trained_pipelines[best_model_name])

    return AutoMLResult(
        problem_type=problem_type,
        best_model_name=best_model_name,
        best_score=best_score,
        test_rows=len(x_test),
        leaderboard=leaderboard.reset_index(drop=True),
        feature_importance=feature_importance,
        target_column=target_column,
        row_count=len(df),
        column_count=len(df.columns),
    )
