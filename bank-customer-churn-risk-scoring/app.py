from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents import build_agent_plan, summarize_dataset
from src.automl import run_automl
from src.reporting import build_html_report, build_notebook_code


APP_DIR = Path(__file__).parent
SAMPLE_DATA = APP_DIR / "sample_data" / "bank_customer_churn_sample.csv"


st.set_page_config(
    page_title="Bank Churn Risk Scoring",
    page_icon="AI",
    layout="wide",
)

st.title("Predictive Modeling and Risk Scoring for Bank Customer Churn")
st.caption("Predict customer churn, rank churn risk, and explain the main drivers behind customer attrition.")


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_DATA)


def read_uploaded_csv() -> pd.DataFrame | None:
    uploaded_file = st.sidebar.file_uploader("Upload bank customer CSV dataset", type=["csv"])
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


with st.sidebar:
    st.header("Dataset")
    use_sample = st.toggle("Use sample bank customer churn dataset", value=True)
    uploaded_df = None if use_sample else read_uploaded_csv()

df = load_sample_data() if use_sample else uploaded_df

if df is None:
    st.info("Upload a CSV dataset or enable the sample dataset.")
    st.stop()

summary = summarize_dataset(df)

left, right = st.columns([2, 1])
with left:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
with right:
    st.subheader("Dataset Summary")
    st.metric("Rows", summary.rows)
    st.metric("Columns", summary.columns)
    st.metric("Missing Cells", summary.missing_cells)

target_column = st.selectbox(
    "Select target column",
    options=list(df.columns),
    index=list(df.columns).index("Exited") if "Exited" in df.columns else len(df.columns) - 1,
)

agent_plan = build_agent_plan(df, target_column)
with st.expander("Agent Plan", expanded=True):
    for step in agent_plan:
        st.write(f"**{step.name}:** {step.action}")

if st.button("Run Churn Risk Model", type="primary"):
    with st.spinner("Agents are analyzing bank customer data, training churn models, and preparing risk insights..."):
        result = run_automl(df, target_column)
        report_html = build_html_report(result)
        notebook_code = build_notebook_code(target_column)

    st.success(f"Best model: {result.best_model_name}")

    st.subheader("Model Leaderboard")
    st.dataframe(result.leaderboard, use_container_width=True)

    metric_cols = st.columns(3)
    metric_cols[0].metric("Problem Type", result.problem_type.title())
    metric_cols[1].metric("Best Score", f"{result.best_score:.4f}")
    metric_cols[2].metric("Test Rows", result.test_rows)

    st.subheader("Risk Scoring Logic")
    st.write(
        "The project converts churn probability into a 0-100 risk score. "
        "Low risk: 0-39, Medium risk: 40-69, High risk: 70-100."
    )

    if result.feature_importance is not None and not result.feature_importance.empty:
        st.subheader("Feature Importance")
        st.bar_chart(result.feature_importance.set_index("feature")["importance"])

    st.subheader("Generated Report")
    st.download_button(
        "Download HTML Report",
        data=report_html,
        file_name="bank_churn_risk_report.html",
        mime="text/html",
    )
    st.components.v1.html(report_html, height=500, scrolling=True)

    st.subheader("Generated Kaggle Notebook Code")
    st.download_button(
        "Download Notebook Code",
        data=notebook_code,
        file_name="bank_churn_risk_notebook.py",
        mime="text/x-python",
    )
    st.code(notebook_code, language="python")
