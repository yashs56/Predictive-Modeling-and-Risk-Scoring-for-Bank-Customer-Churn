from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    columns: int
    missing_cells: int
    numeric_columns: list[str]
    categorical_columns: list[str]


@dataclass(frozen=True)
class AgentStep:
    name: str
    action: str


def summarize_dataset(df: pd.DataFrame) -> DatasetSummary:
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return DatasetSummary(
        rows=len(df),
        columns=len(df.columns),
        missing_cells=int(df.isna().sum().sum()),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def build_agent_plan(df: pd.DataFrame, target_column: str) -> list[AgentStep]:
    summary = summarize_dataset(df)
    feature_count = max(summary.columns - 1, 0)
    return [
        AgentStep(
            "Planner Agent",
            f"Build a bank churn prediction workflow for {summary.rows} rows, {feature_count} features, and target `{target_column}`.",
        ),
        AgentStep(
            "Data Quality Agent",
            f"Check {summary.missing_cells} missing cells and separate numeric/categorical columns.",
        ),
        AgentStep(
            "Preprocessing Agent",
            "Impute missing values, scale numeric features, and one-hot encode categorical features.",
        ),
        AgentStep(
            "Model Training Agent",
            "Train multiple customer churn models and evaluate them on a holdout test split.",
        ),
        AgentStep(
            "Evaluation Agent",
            "Rank churn models using accuracy, F1-score, precision, and recall.",
        ),
        AgentStep(
            "Risk Scoring Agent",
            "Convert churn probability into Low, Medium, and High customer-risk segments.",
        ),
        AgentStep(
            "Report Agent",
            "Generate a final bank churn report and reusable Kaggle-style notebook code.",
        ),
    ]
