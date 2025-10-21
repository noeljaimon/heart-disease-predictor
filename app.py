# Streamlit app code starts here


import streamlit as st
import pandas as pd
import joblib

# Load pre-trained model and scaler
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title('Heart Disease Prediction App')

# User input fields
age = st.slider('Age', 20, 80, 50)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
trestbps = st.slider('Resting Blood Pressure', 80, 200, 120)
chol = st.slider('Serum Cholesterol (mg/dl)', 100, 400, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', [0, 1])
restecg = st.selectbox('Resting ECG Results', [0, 1, 2])
thalach = st.slider('Max Heart Rate Achieved', 70, 200, 150)
exang = st.selectbox('Exercise Induced Angina', [0, 1])
oldpeak = st.slider('ST Depression', 0.0, 6.0, 1.0)
slope = st.selectbox('Slope of ST Segment', [0, 1, 2])
ca = st.slider('Number of Major Vessels Colored', 0, 4, 0)
thal = st.selectbox('Thalassemia', [1, 2, 3])

# Prepare input data
input_dict = {
    'age': age,
    'sex': 1 if sex == 'Male' else 0,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}
input_df = pd.DataFrame([input_dict])

import joblib

# Load the feature columns used during training
feature_columns = joblib.load('feature_columns.pkl')

input_df = pd.DataFrame([your_dict_of_features])
input_dummies = pd.get_dummies(input_df)

# Align to columns used during training
input_aligned = input_dummies.reindex(columns=feature_columns, fill_value=0)


# Scale input
input_scaled = scaler.transform(input_df)

# Predict button
if st.button('Predict'):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error('Warning: Heart Disease Detected')
    else:
        st.success('No Heart Disease Detected')


