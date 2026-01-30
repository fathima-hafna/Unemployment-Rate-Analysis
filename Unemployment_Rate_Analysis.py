import streamlit as st
from PIL import Image
import joblib
import numpy as np

model = joblib.load("best_model.pkl")
gender_le= joblib.load("le1.pkl")
age_le= joblib.load("le2.pkl")
residence_le= joblib.load("le3.pkl")
country_le=joblib.load("le4.pkl")
education_le=joblib.load("le5.pkl")
scaler= joblib.load("scaler.pkl")

st.title("Unemployment Rate Analysis")
st.header("ML Based Unemployment Analysis")
st.subheader("Unemployment Rate Prediction")
st.write("Enter economic indicators to predict unemployment rate.")

st.image(Image.open(r"C:\Users\Afees\OneDrive\Desktop\DS\ML\ml project\Essay-on-Unemployment.webp"))

# Numeric Inputs
year = st.number_input('Year', min_value=2010, max_value=2030, value=2024)
country = st.selectbox('Country', options=['India', 'Australia', 'UK', 'USA', 'Japan', 'Brazil', 'South Africa'])
gender = st.selectbox('Gender', options=['Male', 'Female'])
age_group = st.selectbox('Age Group', options=['18-24', '25-34', '35-44', '45-54', '55+'])
education = st.selectbox('Education Level', options=['No Formal', 'Primary', 'Secondary', 'Undergraduate', 'Postgraduate'])
residence = st.selectbox('Residence Area', options=['Urban', 'Rural'])
experience = st.number_input('Experience Years', min_value=0, max_value=50, value=5)
gdp_growth = st.number_input('GDP Growth Rate (%)', value=2.0)
inflation = st.number_input('Inflation Rate (%)', value=3.0)
literacy = st.number_input('Literacy Rate (%)', min_value=0.0, max_value=100.0, value=95.0)
interest_rate = st.number_input('Interest Rate (%)', value=5.0)

#prediction
if st.button("Predict Unemployment Rate"):
    try:
        country_encoded=country_le.transform([country])[0]
        edu_encoded=education_le.transform([education])[0]
        gender_encoded=gender_le.transform([gender])[0]
        age_encoded=age_le.transform([age_group])[0]
        residence_encoded=residence_le.transform([residence])[0]

        input_data = np.array([[
            year,
            country_encoded,
            gender_encoded,
            age_encoded,
            edu_encoded,
            residence_encoded,
            experience,
            gdp_growth,
            inflation,
            literacy,
            interest_rate
            
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.success(
            f"Predicted Unemployment Rate: {prediction:.2f} %"
        )

    except Exception as e:
        st.error(f"Error in prediction: {e}")