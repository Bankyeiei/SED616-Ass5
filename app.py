import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("best_model.pkl")

# Streamlit UI
st.title("My first ML App (Study on Imbalanced Data Classification by 67130701724")

# Input fields
features = []
features_mean = [6.50e+00, 2.50e-01, 3.00e-01, 5.00e+00, 5.00e-02, 4.00e+01, 1.40e+02, 9.90e-01, 3.20e+00, 5.00e-01, 1.00e+01]
features_step = [0.5, 0.05, 0.05, 0.25, 0.01, 10.0, 20.0, 0.002, 0.1, 0.05, 0.5]

for i in range(11):  # Adjust based on dataset
    if i != 7 :
        value = st.number_input(f"Feature_{i}", value=features_mean[i], step=features_step[i])
    else :
        value = st.number_input(f"Feature_{i}", value=features_mean[i], step=features_step[i], format="%0.3f")
    features.append(value)

# Prediction
if st.button("Predict"):
    prediction = model.predict([np.array(features)])
    st.write(f"Predicted Class: {prediction[0]}")
