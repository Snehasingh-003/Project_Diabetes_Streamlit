import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression


st.title("Diabetes Prediction Site")
model = pickle.load(open('diabetes_model.pkl', 'rb'))


def add_bg():
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1624454002429-40ed87a5ec04?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8ZGlhYmV0ZXN8ZW58MHx8MHx8fDA%3D");
            background-size: cover;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
add_bg()


pregnancies = st.number_input("**Pregnancies**", min_value=0)
glucose = st.number_input("**Glucose Level**")
bp = st.number_input("**Blood Pressure**")
skin_thickness = st.number_input("**Skin Thickness**")
insulin = st.number_input("**Insulin Level**")
bmi = st.number_input("**BMI**")
dpf = st.number_input("**Diabetes Pedigree Function**")
age = st.number_input("**Age**")

DEFAULTS = {
    "Pregnancies": 3,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 80,
    "BMI": 32.0,
    "DiabetesPedigreeFunction": 0.47,
    "Age": 33
}

def get_value(user_input, default):
    try:
        return float(user_input)
    except:
        return default

st.markdown("""
    <style>
    div.stButton > button:first-child {
        color: black !important;
        background-color: #f9f9f9;
        border: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)


if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness,
                            insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.success(f"The person is {result}")