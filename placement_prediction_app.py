import pandas as pd
import numpy as np
import joblib
import streamlit as st

st.title("Placement Predictioon App")

# Load The Models

model = joblib.load("placement_prd_model.pkl")
encoder = joblib.load("placement_prd_encoder.pkl")
scaler = joblib.load("placement_prd_scaler.pkl")
output_encoder = joblib.load("placement_prd_output_encoder.pkl")

# User input

cgpa = st.number_input("CGPA",min_value=0.0 , max_value=10.0)
internships = st.number_input("No of Internships",min_value=0 , max_value=3)
projects = st.number_input("No of Projects",min_value=0 , max_value=10)
workshop_certificates = st.number_input("Workshops or Certifications", min_value=0 , max_value=10)
aptitude_test_score = st.number_input("Aptitude Test Score",min_value=0 , max_value=100)
soft_skill_rating = st.number_input("Soft Skill Rating (0-5)",min_value=0.0 , max_value=5.0)
extra_curricular_activities = st.radio("Extra Curricular Activities",['Yes','No'])
placement_training = st.radio("Placement Training",['Yes','No'])
ssc_mark = st.number_input("SSC Percentage",min_value=1 , max_value=100)
hsc_mark = st.number_input("HSC Percentage",min_value=1 ,max_value=100)

# Input DataFrame

input_data = [[cgpa, internships, projects, workshop_certificates,
              aptitude_test_score, soft_skill_rating, extra_curricular_activities,
              placement_training, ssc_mark, hsc_mark]]

columns = ['CGPA', 'Internships', 'Projects', 'Workshops/Certifications',
       'AptitudeTestScore', 'SoftSkillsRating', 'ExtracurricularActivities',
       'PlacementTraining', 'SSC_Marks', 'HSC_Marks']

input_df = pd.DataFrame(input_data,columns=columns)

# Encoding categorical column

categorical_column = input_df.select_dtypes(include=["object"]).columns
numerical_column = input_df.select_dtypes(include=["int64","float64"]).columns

input_df[categorical_column] = encoder.transform(input_df[categorical_column])
input_df[numerical_column] = scaler.transform(input_df[numerical_column])

if st.button("Predict"):
    prediction = model.predict(input_df)
    orignal_prediction = output_encoder.inverse_transform(prediction.reshape(1 , -1))[0][0]
    st.success(orignal_prediction)