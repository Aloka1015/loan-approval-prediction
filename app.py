
import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load trained model

model = joblib.load("loan_approval_model.pkl")

# Load feature information

with open("feature_metadata.json", "r") as f:
    metadata = json.load(f)

st.title("🏦 Loan Approval Prediction System")

st.write(
    "Enter applicant information to predict "
    "the likely loan approval status."
)

# Create input form

with st.form("loan_form"):

    input_data = {}

    st.subheader("Applicant Information")

    for col in metadata["feature_columns"]:

        if col in metadata["numerical_columns"]:

            info = metadata["numeric_info"][col]

            input_data[col] = st.number_input(
                col.replace("_", " "),
                min_value=float(info["min"]),
                max_value=float(info["max"]),
                value=float(info["default"])
            )

        else:

            options = metadata["categorical_info"][col]

            input_data[col] = st.selectbox(
                col.replace("_", " "),
                options
            )

    submit = st.form_submit_button(
        "Predict Loan Approval"
    )

# Make prediction

if submit:

    applicant = pd.DataFrame([input_data])

    # Feature engineering

    income = applicant["Annual_Income"].replace(0, float("nan"))

    applicant["Debt_to_Income"] = (
        applicant["Outstanding_Debt"] / income
    )

    applicant["Loan_to_Income"] = (
        applicant["Loan_Amount_Requested"] / income
    )

    applicant["Monthly_Net_Income"] = (
        applicant["Annual_Income"] / 12
        - applicant["Monthly_Expenses"]
    )

    # Predict

    prediction = model.predict(applicant)[0]

    probability = model.predict_proba(applicant)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("Predicted Result: Loan Approved")

    else:

        st.error("Predicted Result: Loan Not Approved")

    st.write(
        f"Estimated approval probability: "
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    st.caption(
        "Educational machine learning demonstration only. "
        "This prediction is not an actual lending decision "
        "or a substitute for a bank's assessment."
    )
