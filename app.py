import streamlit as st
import joblib
import numpy as np

# Load models
class_model = joblib.load("classification_model.pkl")
reg_model = joblib.load("regression_model.pkl")
st.title("Multi-Model Prediction App")

problem = st.selectbox(
    "Choose Problem Type",
    ["Classification", "Regression"]
)

if problem == "Classification":
    st.subheader("Iris Classification")

    f1 = st.number_input("Sepal Length")
    f2 = st.number_input("Sepal Width")
    f3 = st.number_input("Petal Length")
    f4 = st.number_input("Petal Width")

    if st.button("Predict"):
        data = np.array([[f1, f2, f3, f4]])
        result = reg_model.predict(data)
        st.success(round(result[0], 2))

else:
    st.subheader("Student Math Score Prediction")

    f1 = st.number_input("Gender")
    f2 = st.number_input("Race/Ethnicity")
    f3 = st.number_input("Parental Education")
    f4 = st.number_input("Lunch")
    f5 = st.number_input("Test Preparation")
    f6 = st.number_input("Reading Score")
    f7 = st.number_input("Writing Score")

    if st.button("Predict"):
        data = np.array([[f1, f2, f3, f4, f5, f6, f7]])
        result = reg_model.predict(data)
        st.success(result[0])
