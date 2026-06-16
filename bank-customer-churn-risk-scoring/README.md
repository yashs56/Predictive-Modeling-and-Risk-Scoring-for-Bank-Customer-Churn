# Predictive Modeling and Risk Scoring for Bank Customer Churn

**Project title:** Predictive Modeling and Risk Scoring for Bank Customer Churn

This capstone project predicts whether a bank customer is likely to leave the bank and assigns a customer churn risk score. It uses customer demographic, account, credit, balance, and product-usage data to support customer retention decisions.

## Why This Project Stands Out

- It solves a real banking analytics problem.
- It combines predictive modeling, risk scoring, explainable AI, and reporting.
- It produces business-friendly customer risk segments.
- It gives a polished demo with model scores, feature importance, and a generated report.

## Features

- Bank customer dataset upload through Streamlit
- Churn prediction using classification models
- Risk score concept from churn probability
- Risk segmentation into Low, Medium, and High churn risk
- Data summary and missing-value analysis
- Automatic preprocessing for numeric and categorical columns
- Multiple model training
- Best model selection
- Feature importance explanation
- Downloadable HTML report
- Generated Kaggle-style notebook code
- Built-in sample bank dataset

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Plotly
- Joblib

## Installation

```bash
pip install -r requirements.txt
```

## Windows Quick Start

If you are using PowerShell or Command Prompt on Windows:

```bat
setup_windows.bat
run_windows.bat
```

## Run the App

```bash
streamlit run app.py
```

If you see `ModuleNotFoundError: No module named 'sklearn'`, install the required packages inside your active virtual environment:

```bash
python -m pip install -r requirements.txt
```

## Demo Steps

1. Open the app.
2. Upload `sample_data/bank_customer_churn_sample.csv`.
3. Select `Exited` as the target column.
4. Click **Run Churn Risk Model**.
5. Show the agent plan, model leaderboard, feature importance, report, and generated notebook code.

## Competition Pitch

Our project predicts bank customer churn and converts model probability into a clear risk score. It helps banks identify high-risk customers early, understand why they may leave, and take retention actions before revenue is lost.

## Suggested Future Enhancements

- Add LLM-based customer retention recommendations.
- Add automatic hyperparameter tuning.
- Add SHAP explanations.
- Add live deployment to Streamlit Community Cloud.
- Add `.ipynb` export.
