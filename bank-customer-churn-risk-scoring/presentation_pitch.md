# Presentation Pitch

## Problem

Banks lose revenue when valuable customers close accounts or stop using banking services. If high-risk customers are identified early, the bank can offer personalized retention actions.

## Proposed Solution

This project predicts customer churn and assigns each customer a churn risk score. The model uses customer profile, balance, credit score, tenure, products, card usage, and activity status to identify customers likely to leave.

## Agent Modules

1. Planner Agent: Decides the churn modeling workflow.
2. Data Quality Agent: Checks missing values and column types.
3. Preprocessing Agent: Cleans and transforms data.
4. Model Training Agent: Trains multiple churn prediction models.
5. Evaluation Agent: Selects the best model.
6. Risk Scoring Agent: Converts churn probability into Low, Medium, and High risk.
7. Report Agent: Generates the final report and notebook code.

## Winning Points

- It solves a banking business problem.
- It includes ML, risk scoring, explainability, and reporting.
- It creates actionable retention insights.
- It produces a professional final output during the live demo.

## Future Scope

- Add LLM-based customer retention suggestions.
- Add automatic hyperparameter tuning.
- Add SHAP explanations.
- Add deployment to Streamlit Community Cloud.
- Add `.ipynb` export.
