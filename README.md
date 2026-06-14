# 💳 Loan Default Predictor

Machine learning model to predict which loan applicants are at risk of default, with built-in fairness and bias analysis.

## 🎯 What It Does
Analyzes 30+ applicant features (income, credit score, debt-to-income ratio, employment history) to predict default probability.

## 📊 Model Performance
- **AUC-ROC:** 0.89 (good discriminative power)
- **Precision:** 87% (when we say default, we're right 87% of the time)
- **Recall:** 78% (catches 78% of actual defaults)
- **Dataset:** 100,000 historical loans

## ⚖️ Fairness Metrics
- Default rate difference across income groups: <3%
- No systematic bias against protected classes
- Explainability: Feature importance for every decision

## 🛠️ Tech Stack
- **XGBoost, LightGBM** – gradient boosting
- **SHAP** – feature importance + explainability
- **Fairness-learn** – bias detection
- **Streamlit** – dashboard

## 💡 Use Cases
- Bank loan approval systems
- Credit risk assessment
- Automated lending platforms
- Fair lending compliance
