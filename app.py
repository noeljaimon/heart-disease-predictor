
import streamlit as st
import pandas as pd
import joblib

# Load pre-trained model, scaler, and feature columns
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title('Heart Disease Prediction App')

# Collect user inputs with verbal categories and mappings

age = st.slider('Age', 20, 80, 50)

sex_options = {'Male': 1, 'Female': 0}
sex_choice = st.selectbox('Sex', list(sex_options.keys()))
sex = sex_options[sex_choice]

cp_options = {
    'Typical Angina': 0,
    'Atypical Angina': 1,
    'Non-anginal Pain': 2,
    'Asymptomatic': 3
}
cp_choice = st.selectbox('Chest Pain Type', list(cp_options.keys()))
cp = cp_options[cp_choice]

trestbps = st.slider('Resting Blood Pressure', 80, 200, 120)
chol = st.slider('Serum Cholesterol (mg/dl)', 100, 400, 200)

fbs_options = {'False': 0, 'True': 1}
fbs_choice = st.selectbox('Fasting Blood Sugar > 120 mg/dl', list(fbs_options.keys()))
fbs = fbs_options[fbs_choice]

restecg_options = {
    'Normal': 0,
    'ST-T Wave Abnormality': 1,
    'Left Ventricular Hypertrophy': 2
}
restecg_choice = st.selectbox('Resting ECG Results', list(restecg_options.keys()))
restecg = restecg_options[restecg_choice]

thalach = st.slider('Max Heart Rate Achieved', 70, 200, 150)

exang_options = {'No': 0, 'Yes': 1}
exang_choice = st.selectbox('Exercise Induced Angina', list(exang_options.keys()))
exang = exang_options[exang_choice]

oldpeak = st.slider('ST Depression', 0.0, 6.0, 1.0)

slope_options = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
slope_choice = st.selectbox('Slope of ST Segment', list(slope_options.keys()))
slope = slope_options[slope_choice]

ca = st.slider('Number of Major Vessels Colored', 0, 4, 0)

thal_options = {'Normal': 1, 'Fixed Defect': 2, 'Reversible Defect': 3}
thal_choice = st.selectbox('Thalassemia', list(thal_options.keys()))
thal = thal_options[thal_choice]

# Prepare input dictionary for prediction
input_dict = {
    'age': age,
    'sex': sex,
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

# One-hot encode and align columns with training features
input_dummies = pd.get_dummies(input_df)
input_aligned = input_dummies.reindex(columns=feature_columns, fill_value=0)

# Scale data and predict
input_scaled = scaler.transform(input_aligned)

if st.button('Predict'):
    prediction = model.predict(input_scaled)
    if prediction[0] == 1:
        st.error('Warning: Heart Disease Detected')
    else:
        st.success('No Heart Disease Detected')

