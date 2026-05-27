import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="LoanGuard AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size: 3.2rem; color: #1E3A8A; text-align: center; font-weight: bold;}
    .sub-header {font-size: 1.6rem; color: #3B82F6; text-align: center; margin-bottom: 30px;}
    .success-box {background-color: #10B981; color: white; padding: 25px; border-radius: 15px; text-align: center; font-size: 1.8rem; font-weight: bold;}
    .danger-box {background-color: #EF4444; color: white; padding: 25px; border-radius: 15px; text-align: center; font-size: 1.8rem; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏦 LoanGuard AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Loan Approval Prediction System</p>', unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=90)
    st.title("Navigation")
    page = st.radio("Go to", ["🏠 Home", "🔮 Predict Loan", "📊 Model Insights", "ℹ️ About"])

# ====================== LOAD & TRAIN MODEL ======================
@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv("train.csv")
        
        df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
        df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
        if 'Dependents' in df.columns: df = df.drop('Dependents', axis=1)
        df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
        df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median()).astype(int)
        df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median()).astype(int)
        df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].median()).astype(int)
        if 'Loan_ID' in df.columns: df = df.drop('Loan_ID', axis=1)

        le = LabelEncoder()
        for col in ['Gender', 'Married', 'Self_Employed', 'Property_Area']:
            df[col] = le.fit_transform(df[col])

        df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
        df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})

        X = df.drop('Loan_Status', axis=1)
        y = df['Loan_Status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        accuracy = accuracy_score(y_test, model.predict(X_test))
        return model, accuracy, X_test, y_test, df, X.columns.tolist()
    
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None, None, None

model, accuracy, X_test, y_test, df, feature_names = load_and_train_model()

# ====================== PAGES ======================
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Model Accuracy", f"{accuracy*100:.2f}%" if accuracy else "N/A")
    with col2: st.metric("Total Applications", len(df) if df is not None else "N/A")
    with col3: st.metric("Approval Rate", "68.7%")

elif page == "🔮 Predict Loan":
    st.title("🔮 Loan Approval Prediction")
    st.markdown("#### Enter Applicant Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name","XYZ")
        address = st.text_area("Address", "Mumbai, Maharashtra", height=80)
        
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    with col2:
        applicant_income = st.number_input(
            "Applicant Income (₹ in hundreds)", 
            min_value=0, 
            value=500,
            help="Enter in hundreds. Example: 500 = ₹50,000. Value will be used directly."
        )
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹ in hundreds)", 
            min_value=0, 
            value=200,
            help="Enter in hundreds. Example: 200 = ₹20,000. Value will be used directly."
        )
        loan_amount = st.number_input(
            "Loan Amount (₹ in hundreds)", 
            min_value=10, 
            value=1500,
            help="Enter in hundreds. Example: 1500 = ₹1,50,000. Value will be used directly."
        )
        
        loan_term = st.selectbox("Loan Term", [120, 180, 240, 360, 480])
        credit_history = st.selectbox("Credit History", [1, 0], 
                                    format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")
        property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    if st.button("🚀 Predict Loan Status", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded.")
        else:
            prop_map = {"Rural": 0, "Semiurban": 1, "Urban": 2}
            
            input_data = pd.DataFrame([[
                1 if gender == "Male" else 0,
                1 if married == "Yes" else 0,
                1 if education == "Graduate" else 0,
                1 if self_employed == "Yes" else 0,
                applicant_income,          # Used directly (no *100)
                coapplicant_income,        # Used directly
                loan_amount,               # Used directly
                loan_term,
                credit_history,
                prop_map[property_area]
            ]], columns=feature_names)

            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]

            if pred == 1:
                st.markdown('<div class="success-box">🎉 LOAN APPROVED!</div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown('<div class="danger-box">❌ Loan Rejected</div>', unsafe_allow_html=True)

            st.metric("Approval Probability", f"{prob*100:.1f}%")

elif page == "📊 Model Insights":
    st.title("📊 Model Insights")
    tab1, tab2 = st.tabs(["Performance", "Feature Importance"])
    
    with tab1:
        st.metric("Accuracy", f"{accuracy*100:.2f}%")
        if X_test is not None:
            cm = confusion_matrix(y_test, model.predict(X_test))
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        imp = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_})
        imp = imp.sort_values('Importance', ascending=False)
        fig = px.bar(imp, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.title("About LoanGuard AI")
    st.info("A beginner ML-based Loan Approval System")

st.caption("© 2026 LoanGuard AI ")