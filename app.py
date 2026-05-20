import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="💰",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #e0ecff, #f8fbff);
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #0F172A;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #475569;
    margin-bottom: 30px;
}

.stButton > button {
    width: 100%;
    height: 3.2em;
    border-radius: 12px;
    background: linear-gradient(to right, #2563EB, #1D4ED8);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(to right, #1D4ED8, #1E40AF);
    color: white;
}

div[data-baseweb="select"] {
    border-radius: 10px;
}

input {
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.markdown("""
<h1 style='
text-align: center;
font-size: 95px;
font-weight: 800;
color: #ffd700;
margin-bottom: 0px;
'>
💰 Loan Approval Prediction
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='
text-align: center;
color: #475569;
font-size: 28px;
margin-top: 0px;
margin-bottom: 40px;
'>
Machine Learning Based Loan Prediction System
</h3>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

df = pd.read_csv("train.csv")

# ---------------- DATA CLEANING ----------------

df['Gender'] = df['Gender'].fillna(
    df['Gender'].mode()[0]
)

df['Married'] = df['Married'].fillna(
    df['Married'].mode()[0]
)

df = df.drop('Dependents', axis=1)

df['Self_Employed'] = df['Self_Employed'].fillna(
    df['Self_Employed'].mode()[0]
)

df['LoanAmount'] = df['LoanAmount'].fillna(
    df['LoanAmount'].median()
)

df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(
    df['Loan_Amount_Term'].median()
)

df['Credit_History'] = df['Credit_History'].fillna(
    df['Credit_History'].median()
)

df = df.drop('Loan_ID', axis=1)

# ---------------- ENCODING ----------------

df['Gender'] = df['Gender'].map({
    'Male': 1,
    'Female': 0
})

df['Married'] = df['Married'].map({
    'Yes': 1,
    'No': 0
})

df['Education'] = df['Education'].map({
    'Graduate': 1,
    'Not Graduate': 0
})

df['Self_Employed'] = df['Self_Employed'].map({
    'Yes': 1,
    'No': 0
})

df['Property_Area'] = df['Property_Area'].map({
    'Rural': 0,
    'Semiurban': 1,
    'Urban': 2
})

df['Loan_Status'] = df['Loan_Status'].map({
    'Y': 1,
    'N': 0
})

# ---------------- FEATURES & TARGET ----------------

x = df.drop('Loan_Status', axis=1)

y = df['Loan_Status']

# ---------------- TRAIN TEST SPLIT ----------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- MODEL TRAINING ----------------

model = LogisticRegression(max_iter=1000)

model.fit(x_train, y_train)

# ---------------- INPUT SECTION ----------------

st.markdown("## 📝 Enter Applicant Details")

left, right = st.columns(2)

with left:

    gender = st.selectbox(
        "👤 Gender",
        ["Male", "Female"]
    )

    married = st.selectbox(
        "💍 Married",
        ["Yes", "No"]
    )

    education = st.selectbox(
        "🎓 Education",
        ["Graduate", "Not Graduate"]
    )

    self_employed = st.selectbox(
        "💼 Self Employed",
        ["Yes", "No"]
    )

    applicant_income = st.text_input(
        "💰 Applicant Income",
        placeholder="Enter monthly income"
    )

with right:

    coapplicant_income = st.text_input(
        "👨‍👩‍👧 Coapplicant Income",
        placeholder="Enter coapplicant income"
    )

    loan_amount = st.text_input(
        "🏦 Loan Amount",
        placeholder="Enter loan amount"
    )

    loan_term = st.selectbox(
        "📅 Loan Amount Term",
        [360.0, 240.0, 120.0]
    )

    credit_history = st.selectbox(
        "📊 Credit History",
        [1, 0]
    )

    property_area = st.selectbox(
        "📍 Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

# ---------------- INPUT ENCODING ----------------

gender = 1 if gender == "Male" else 0

married = 1 if married == "Yes" else 0

education = 1 if education == "Graduate" else 0

self_employed = 1 if self_employed == "Yes" else 0

property_area_map = {
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
}

property_area = property_area_map[property_area]

# ---------------- PREDICTION ----------------

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Loan Status"):

    try:

        applicant_income = int(applicant_income)

        coapplicant_income = int(coapplicant_income)

        loan_amount = int(loan_amount)

        data = np.array([
            gender,
            married,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area
        ]).reshape(1, -1)

        prediction = model.predict(data)

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction[0] == 1:

            st.success("✅ Congratulations! Loan Approved")

            st.balloons()

        else:

            st.error("❌ Sorry! Loan Rejected")

    except:

        st.warning("⚠️ Please enter valid numeric values")
# ---------------- FOOTER ----------------

st.markdown("---")

