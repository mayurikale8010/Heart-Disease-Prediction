
import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessing files
model = joblib.load("KNN_Heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# App title
st.title("Heart Disease Prediction by Mayuri")
st.markdown("Provide the following details")

# User inputs
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type", ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure (mm Hg)", 80, 200, 120
)

cholesterol = st.number_input(
    "Cholesterol (mg/dl)", 100, 600, 200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl", [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG", ["Normal", "ST", "LVH"]
)

max_hr = st.slider("Max Heart Rate", 60, 220, 150)

exercise_angina = st.selectbox(
    "Exercise-Induced Angina", ["Y", "N"]
)

oldpeak = st.slider(
    "Oldpeak (ST Depression)", 0.0, 6.0, 1.0
)

st_slope = st.selectbox(
    "ST Slope", ["Up", "Flat", "Down"]
)

# Prediction
if st.button("Predict"):

    # Prepare raw input
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1
    }

    # Convert input into DataFrame
    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in training order
    input_df = input_df[expected_columns]

    # Scale input data
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]
    st.write("Model Classes:")
    st.code(str(list(model.classes_)))

    st.write("Expected Columns:")
    st.code(str(expected_columns))

    st.write("Input Data:")
    st.dataframe(input_df)

    st.write("Prediction:", prediction)   

    
    # Display prediction
    st.write("Model Prediction:", prediction)

    if prediction == 1:
        st.error("High Risk of Heart Disease")
    elif prediction == 0:
        st.success("Low Risk of Heart Disease")
    else:
        st.warning("Unexpected prediction value")
        
st.caption(
    "This application is for educational purposes only "
    "and is not a medical diagnosis."
)

