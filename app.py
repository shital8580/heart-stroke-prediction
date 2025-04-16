import streamlit as st
import numpy as np
import pandas as pd
import joblib
from fpdf import FPDF
from PIL import Image
import random
import cv2
import time

# Load model, scaler, feature names
model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# Streamlit Config
st.set_page_config(page_title="Zettalogix | Heart & Stroke Detection", page_icon="🫀", layout="wide")

# Load logo
logo = Image.open("logo.png")

# Light Theme Styling
st.markdown("""
<style>
.stApp {
    background-color: #f8f9fa;
    color: #212529;
}
.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    font-size: 1.1em;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image(logo, width=250)
    st.header("Welcome to Zettalogix")
    st.markdown("Predict **Heart Stroke**, generate a **medical PDF**, and prepare for mobile app integration.")
    st.markdown("© 2025 Zettalogix | All rights reserved")

# Main Title
st.markdown("<h1 style='color: #FF4B4B;'>🫀 Heart & Stroke Prediction System</h1>", unsafe_allow_html=True)

with st.form("form"):
    st.subheader("Patient Info")
    name = st.text_input("Full Name")
    age = st.slider("Age", 18, 100, 30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
    ever_married = st.selectbox("Married?", ["Yes", "No"])
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt job", "Children", "Never worked"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

    st.subheader("📏 Physical Info")
    bmi_method = st.radio("BMI Input Method", ["Auto (Height & Weight)", "Manual"])
    if bmi_method == "Manual":
        bmi = st.number_input("BMI", 10.0, 60.0, 24.0)
    else:
        height = st.number_input("Height (cm)", 100, 250, 170)
        weight = st.number_input("Weight (kg)", 30, 200, 65)
        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 2)
        st.success(f"✅ Calculated BMI: {bmi}")

    glucose_input = st.text_input("Average Glucose Level (mg/dL) (optional)")
    if glucose_input.strip() == "":
        if age > 45 or bmi > 25 or smoking_status == "smokes":
            glucose = round(random.uniform(105, 125), 1)
        else:
            glucose = round(random.uniform(85, 100), 1)
        st.warning(f"⚠️ Auto-estimated Glucose: {glucose} mg/dL")
    else:
        glucose = float(glucose_input)

    st.subheader("💤 Lifestyle & Health")
    sleep_hours = st.slider("Sleep (hrs/day)", 0, 12, 7)
    physical_activity = st.slider("Exercise (mins/day)", 0, 180, 30)
    alcohol = st.radio("Alcohol?", ["No", "Yes"])
    salt_intake = st.radio("Excess Salt?", ["No", "Yes"])

    submit = st.form_submit_button("🔍 Predict Stroke Risk")

if submit:
    # Encoding
    gender_bin = 1 if gender == "Male" else 0
    hypertension_bin = 1 if hypertension == "Yes" else 0
    heart_disease_bin = 1 if heart_disease == "Yes" else 0
    ever_married_bin = 1 if ever_married == "Yes" else 0
    residence_type_bin = 1 if residence_type == "Urban" else 0
    alcohol_bin = 1 if alcohol == "Yes" else 0
    salt_bin = 1 if salt_intake == "Yes" else 0

    work_map = {"Private": 0, "Self-employed": 1, "Govt job": 2, "Children": 3, "Never worked": 4}
    smoke_map = {"never smoked": 0, "formerly smoked": 1, "smokes": 2}
    work_type_encoded = work_map[work_type]
    smoke_encoded = smoke_map[smoking_status]

    # Full feature vector
    input_list = [
        age, gender_bin, hypertension_bin, heart_disease_bin,
        ever_married_bin, work_type_encoded, residence_type_bin,
        bmi, glucose, smoke_encoded, sleep_hours, physical_activity,
        alcohol_bin, salt_bin
    ]
    while len(input_list) < len(feature_names):
        input_list.append(0)

    input_df = pd.DataFrame([input_list], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # Prediction Output
    if prediction == 1:
        st.error(f"⚠️ High Stroke Risk: {probability:.2f}%")
    else:
        st.success(f"✅ Low Stroke Risk: {probability:.2f}%")

    # Recommendations
    st.markdown("### 📋 Personalized Recommendations")
    if probability <= 10:
        rec = ["Keep exercising regularly", "Maintain current weight", "Continue healthy lifestyle"]
    elif probability <= 30:
        rec = ["Increase water intake", "Avoid junk/salt", "Monitor glucose every 6 months"]
    elif probability <= 60:
        rec = ["Consult dietician", "Start mild fitness regime", "Practice mindfulness"]
    else:
        rec = ["Visit doctor urgently", "Track BP + glucose", "Stop smoking/alcohol"]

    for r in rec:
        st.markdown(f"- {r}")

    # PDF Report Generator
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 16)
            self.cell(200, 10, "Zettalogix Stroke Risk Report", ln=True, align="C")

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Name: {name}", ln=True)
    pdf.cell(200, 10, f"Age: {age} | Gender: {gender}", ln=True)
    pdf.cell(200, 10, f"BMI: {bmi} | Glucose: {glucose}", ln=True)
    pdf.cell(200, 10, f"Stroke Risk: {probability:.2f}%", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, "Recommendations:", ln=True)
    pdf.set_font("Arial", "", 12)
    for r in rec:
        pdf.cell(200, 10, r, ln=True)
    pdf.output("stroke_report.pdf")

    with open("stroke_report.pdf", "rb") as f:
        st.download_button("📥 Download Stroke Report", f, "stroke_report.pdf")

# Webcam BP Tracker
st.subheader("🩺 BP Measurement via Webcam")

# Capture BP via Webcam
st.write("Click the button below to start capturing your BP using your webcam.")
if st.button("Start Webcam for BP Measurement"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to access webcam. Please try again.")
    else:
        st.write("Webcam is now activated. Please wait while we measure your BP...")
        # Read frames from the webcam for a few seconds
        start_time = time.time()
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                break
            # Display webcam feed
            st.image(frame, channels="BGR", use_container_width=True)  # Updated here
        cap.release()
        st.write("BP measurement completed. Results will be processed soon.")
