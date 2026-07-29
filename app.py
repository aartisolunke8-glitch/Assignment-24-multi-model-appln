import streamlit as st
import pandas as pd
import joblib

pipe = joblib.load('full_pipeline.pkl')

st.title("Student Math Score Prediction")

st.sidebar.header("Student Details")
gender = st.sidebar.selectbox('Gender', ['female', 'male'])
race = st.sidebar.selectbox('Race/Ethnicity', ['group A', 'group B', 'group C', 'group D', 'group E'])
parent = st.sidebar.selectbox('Parental Education', ['some high school', 'high school', 'associates degree', 'some college', 'bachelors degree', 'masters degree'])
lunch = st.sidebar.selectbox('Lunch', ['free/reduced', 'standard'])
test = st.sidebar.selectbox('Test Preparation', ['none', 'completed'])

st.header("Scores")
reading = st.slider('Reading Score', 0, 100, 72)
writing = st.slider('Writing Score', 0, 100, 74)

if st.button('Predict'):
    input_df = pd.DataFrame([[gender,race,parent,lunch,test,reading,writing]],
        columns=['gender','race/ethnicity','parental level of education','lunch','test preparation course','reading score','writing score'])
    pred = pipe.predict(input_df)
    st.success(f"Predicted Math Score: {pred[0]:.2f} / 100")

    
