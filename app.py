import streamlit as st
import joblib
import pandas as pd

model = joblib.load("eye_strain_model.pkl")
data = pd.read_csv("eye_strain_data.csv")

st.title("Eye Strain Prediction System")
st.write("This application predicts eye strain based on digital device usage habits.")

st.header("User Inputs")

feature_columns = data.drop("eye_strain", axis=1).columns

user_input = {}

for col in feature_columns:
    if col == "ID":
        user_input[col] = st.number_input("ID", min_value=1, value=1)

    elif col == "age":
        user_input[col] = st.slider("Age", 10, 80, 20)

    elif col == "phone_usage_hours":
        user_input[col] = st.slider("Phone Usage Hours", 0, 15, 5)

    elif col == "computer_usage_hours":
        user_input[col] = st.slider("Computer Usage Hours", 0, 15, 5)

    elif col == "sleep_hours":
        user_input[col] = st.slider("Sleep Hours", 0, 12, 7)

    elif col == "screen_brightness":
        user_input[col] = st.slider("Screen Brightness", 0, 100, 50)

    elif col == "break_frequency":
        user_input[col] = st.slider("Break Frequency", 0, 10, 2)

    else:
        values = sorted(data[col].dropna().unique())
        user_input[col] = st.selectbox(col, values)

input_data = pd.DataFrame([user_input])

if st.button("Predict"):
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Prediction: High Eye Strain")
    else:
        st.success("Prediction: Low Eye Strain")