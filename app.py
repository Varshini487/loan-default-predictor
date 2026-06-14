import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="💳 Loan Default Predictor", layout="wide")
st.title("💳 Loan Default Predictor")
st.markdown("Predict loan default risk with fairness analysis")

@st.cache_data
def generate_loan_data(n=2000):
    np.random.seed(42)
    df = pd.DataFrame({
        "annual_income": np.random.lognormal(10.5, 0.8, n),  # ~$25k-500k
        "credit_score": np.random.randint(300, 850, n),
        "employment_years": np.random.randint(0, 40, n),
        "loan_amount": np.random.lognormal(10.3, 1.2, n),
        "debt_to_income": np.random.uniform(0.1, 0.8, n),
        "num_accounts": np.random.randint(1, 15, n),
        "num_inquiries": np.random.randint(0, 5, n),
        "age": np.random.randint(18, 80, n),
        "income_group": np.random.choice(["Low", "Middle", "High"], n, p=[0.35, 0.35, 0.30])
    })
    # Default probability: lower credit score, higher debt ratio → more likely to default
    default_prob = (
        0.05 +
        (850 - df["credit_score"]) * 0.0001 +
        df["debt_to_income"] * 0.2 -
        df["credit_score"] * 0.0002 +
        np.random.normal(0, 0.05, n)
    )
    default_prob = np.clip(default_prob, 0, 1)
    df["default"] = (default_prob > 0.5).astype(int)
    return df

df = generate_loan_data()

tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Model", "🔮 Predict", "⚖️ Fairness"])

with tab1:
    st.subheader("Loan Dataset Overview")
    st.dataframe(df.head(20))
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Loans", len(df))
    col2.metric("Defaults", df["default"].sum())
    col3.metric("Default Rate", f"{df['default'].mean():.1%}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(df[df.default==0]["credit_score"], bins=30, alpha=0.7, label="No Default", color="green")
    axes[0, 0].hist(df[df.default==1]["credit_score"], bins=30, alpha=0.7, label="Default", color="red")
    axes[0, 0].set_title("Credit Score Distribution"); axes[0, 0].legend()
    
    axes[0, 1].scatter(df[df.default==0]["debt_to_income"], df[df.default==0]["annual_income"], alpha=0.3, s=20, label="No Default")
    axes[0, 1].scatter(df[df.default==1]["debt_to_income"], df[df.default==1]["annual_income"], alpha=0.5, s=20, color="red", label="Default")
    axes[0, 1].set_title("Debt-to-Income vs Income"); axes[0, 1].legend()
    
    df.groupby("income_group")["default"].mean().plot(kind="bar", ax=axes[1, 0], color=["gray", "orange", "green"])
    axes[1, 0].set_title("Default Rate by Income Group"); axes[1, 0].set_ylabel("Default Rate")
    
    df.groupby("age")["default"].rolling(window=5).mean().plot(ax=axes[1, 1])
    axes[1, 1].set_title("Default Rate by Age"); axes[1, 1].set_xlabel("Age")
    
    st.pyplot(fig)

with tab2:
    if st.button("🚀 Train Default Model"):
        X = df.drop(["default", "income_group"], axis=1)
        y = df["default"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        
        st.success(f"✅ Model Trained! AUC-ROC: {auc:.3f}")
        
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)
        
        st.text(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
        
        st.session_state["model"] = model
        st.session_state["X_train"] = X_train
        st.session_state["feature_names"] = X.columns.tolist()

with tab3:
    st.subheader("Predict Default Risk for Applicant")
    c1, c2 = st.columns(2)
    income = c1.number_input("Annual Income ($)", 20000, 500000, 50000)
    credit = c2.slider("Credit Score", 300, 850, 700)
    emp_years = c1.slider("Employment Years", 0, 40, 5)
    loan_amt = c2.number_input("Loan Amount ($)", 5000, 500000, 200000)
    debt_ratio = c1.slider("Debt-to-Income Ratio", 0.1, 0.8, 0.3)
    accounts = c2.slider("Number of Accounts", 1, 15, 5)
    inquiries = c1.slider("Recent Inquiries", 0, 5, 1)
    age = c2.slider("Age", 18, 80, 35)
    
    if st.button("💳 Check Default Risk") and "model" in st.session_state:
        inp = np.array([[income, credit, emp_years, loan_amt, debt_ratio, accounts, inquiries, age]])
        default_prob = st.session_state["model"].predict_proba(inp)[0][1]
        
        st.markdown("### 📊 Risk Assessment")
        st.metric("Default Probability", f"{default_prob:.1%}")
        
        if default_prob > 0.6:
            st.error(f"🔴 HIGH RISK: {default_prob:.1%} — Application recommend REJECTION")
        elif default_prob > 0.4:
            st.warning(f"🟡 MEDIUM RISK: {default_prob:.1%} — Request additional documentation")
        else:
            st.success(f"🟢 LOW RISK: {default_prob:.1%} — Application APPROVED")

with tab4:
    st.subheader("⚖️ Fairness Analysis")
    st.info("Checking for bias across demographic groups")
    
    income_groups = df.groupby("income_group")["default"].agg(["sum", "count", "mean"])
    income_groups.columns = ["Defaults", "Total", "Default Rate"]
    st.dataframe(income_groups)
    
    fig, ax = plt.subplots()
    income_groups["Default Rate"].plot(kind="bar", color=["red", "orange", "green"], ax=ax)
    ax.set_title("Default Rate by Income Group (Fairness Check)")
    ax.set_ylabel("Default Rate")
    ax.axhline(y=df["default"].mean(), color="gray", linestyle="--", label="Overall Avg")
    ax.legend()
    st.pyplot(fig)
    
    max_diff = (income_groups["Default Rate"].max() - income_groups["Default Rate"].min()) * 100
    if max_diff < 3:
        st.success(f"✅ Fair: Default rate difference = {max_diff:.1f}% (< 3% threshold)")
    else:
        st.warning(f"⚠️ Potential bias: Default rate difference = {max_diff:.1f}% (> 3% threshold)")
