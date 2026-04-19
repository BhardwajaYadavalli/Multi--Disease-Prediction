"""
Multi-Disease Prediction App
Author: Bhardwaja
Master's in AI & Business Intelligence — University of Leicester

Predicts: Kidney Disease | Diabetes | Breast Cancer
Models: Random Forest / SVM trained in individual notebooks
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Disease Predictor",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disease-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 5px solid #3498db;
        margin-bottom: 1rem;
    }
    .result-positive {
        background: #fdecea;
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 1.2rem;
        font-size: 1.2rem;
        font-weight: bold;
        color: #c0392b;
        text-align: center;
    }
    .result-negative {
        background: #eafaf1;
        border: 2px solid #2ecc71;
        border-radius: 10px;
        padding: 1.2rem;
        font-size: 1.2rem;
        font-weight: bold;
        color: #1e8449;
        text-align: center;
    }
    .disclaimer {
        background: #fef9e7;
        border: 1px solid #f39c12;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.85rem;
        color: #7d6608;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🏥 Multi-Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-powered prediction for Kidney Disease · Diabetes · Breast Cancer</p>', unsafe_allow_html=True)

st.markdown('<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This tool is built for educational and portfolio purposes only. It is NOT a substitute for professional medical advice or clinical diagnosis.</div>', unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    try:
        models['kidney'] = {
            'model':   joblib.load('kidney_model.pkl'),
            'imputer': joblib.load('kidney_imputer.pkl'),
        }
    except FileNotFoundError:
        models['kidney'] = None

    try:
        models['diabetes'] = {
            'model':   joblib.load('diabetes_model.pkl'),
            'imputer': joblib.load('diabetes_imputer.pkl'),
        }
    except FileNotFoundError:
        models['diabetes'] = None

    try:
        models['breast'] = {
            'model':  joblib.load('breast_cancer_model.pkl'),
            'scaler': joblib.load('breast_cancer_scaler.pkl'),
        }
    except FileNotFoundError:
        models['breast'] = None

    return models

models = load_models()

# ─────────────────────────────────────────────
# Disease Selector
# ─────────────────────────────────────────────
disease = st.selectbox(
    "🔬 Select Disease to Predict",
    ["🫘 Chronic Kidney Disease (CKD)", "🩸 Diabetes", "🎗️ Breast Cancer"],
    index=0
)

st.markdown("---")

# ═══════════════════════════════════════════════
# ① KIDNEY DISEASE
# ═══════════════════════════════════════════════
if disease == "🫘 Chronic Kidney Disease (CKD)":
    st.subheader("🫘 Kidney Disease — Enter Patient Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        age  = st.number_input("Age (years)",        min_value=1,   max_value=100, value=45)
        bp   = st.number_input("Blood Pressure (mm/Hg)", min_value=50, max_value=180, value=80)
        sg   = st.selectbox("Specific Gravity",      [1.005,1.010,1.015,1.020,1.025], index=2)
        al   = st.slider("Albumin",                  0, 5, 1)
        su   = st.slider("Sugar",                    0, 5, 0)
        bgr  = st.number_input("Blood Glucose Random (mg/dL)", 70, 500, 120)
        bu   = st.number_input("Blood Urea (mg/dL)", 10, 200, 40)
        sc   = st.number_input("Serum Creatinine (mg/dL)", 0.4, 15.0, 1.2)

    with col2:
        sod  = st.number_input("Sodium (mEq/L)",     100, 165, 138)
        pot  = st.number_input("Potassium (mEq/L)",  2.0, 8.0, 4.5)
        hemo = st.number_input("Haemoglobin (g/dL)", 3.0, 18.0, 13.5)
        pcv  = st.number_input("Packed Cell Volume", 10, 55, 44)
        wc   = st.number_input("White Blood Cell Count (cells/cumm)", 2000, 20000, 8000)
        rc   = st.number_input("Red Blood Cell Count (millions/cmm)", 1.5, 7.0, 5.0)

    with col3:
        rbc   = st.selectbox("Red Blood Cells",       ["Normal","Abnormal"])
        pc    = st.selectbox("Pus Cell",              ["Normal","Abnormal"])
        pcc   = st.selectbox("Pus Cell Clumps",       ["Not Present","Present"])
        ba    = st.selectbox("Bacteria",              ["Not Present","Present"])
        htn   = st.selectbox("Hypertension",          ["No","Yes"])
        dm    = st.selectbox("Diabetes Mellitus",     ["No","Yes"])
        cad   = st.selectbox("Coronary Artery Disease",["No","Yes"])
        appet = st.selectbox("Appetite",              ["Good","Poor"])
        pe    = st.selectbox("Pedal Edema",           ["No","Yes"])
        ane   = st.selectbox("Anaemia",               ["No","Yes"])

    if st.button("🔍 Predict Kidney Disease", use_container_width=True):
        if models['kidney'] is None:
            st.error("❌ Kidney model not found. Please run the kidney notebook first to generate kidney_model.pkl")
        else:
            features = np.array([[
                age, bp, sg, al, su,
                1 if rbc=="Normal" else 0,
                1 if pc=="Normal" else 0,
                1 if pcc=="Present" else 0,
                1 if ba=="Present" else 0,
                bgr, bu, sc, sod, pot, hemo, pcv, wc, rc,
                1 if htn=="Yes" else 0,
                1 if dm=="Yes" else 0,
                1 if cad=="Yes" else 0,
                1 if appet=="Good" else 0,
                1 if pe=="Yes" else 0,
                1 if ane=="Yes" else 0
            ]])
            features_imp = models['kidney']['imputer'].transform(features)
            pred = models['kidney']['model'].predict(features_imp)[0]
            proba = models['kidney']['model'].predict_proba(features_imp)[0]

            st.markdown("### 📊 Prediction Result")
            if pred == 1:
                st.markdown(f'<div class="result-positive">⚠️ HIGH RISK: Chronic Kidney Disease Detected<br><small>Confidence: {proba[1]*100:.1f}%</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-negative">✅ LOW RISK: No CKD Detected<br><small>Confidence: {proba[0]*100:.1f}%</small></div>', unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                'Outcome': ['Not CKD', 'CKD'],
                'Probability': [proba[0], proba[1]]
            })
            st.bar_chart(prob_df.set_index('Outcome'))


# ═══════════════════════════════════════════════
# ② DIABETES
# ═══════════════════════════════════════════════
elif disease == "🩸 Diabetes":
    st.subheader("🩸 Diabetes — Enter Patient Details")

    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Number of Pregnancies",      0, 20, 1)
        glucose     = st.number_input("Glucose Level (mg/dL)",      40, 300, 110)
        bp          = st.number_input("Blood Pressure (mm/Hg)",     30, 130, 70)
        skin        = st.number_input("Skin Thickness (mm)",        0, 100, 20)
    with col2:
        insulin     = st.number_input("Insulin (µU/mL)",            0, 900, 80)
        bmi         = st.number_input("BMI (kg/m²)",                10.0, 70.0, 25.0)
        dpf         = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age         = st.number_input("Age (years)",                1, 120, 30)

    if st.button("🔍 Predict Diabetes", use_container_width=True):
        if models['diabetes'] is None:
            st.error("❌ Diabetes model not found. Please run the diabetes notebook first to generate diabetes_model.pkl")
        else:
            features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
            features_imp = models['diabetes']['imputer'].transform(features)
            pred  = models['diabetes']['model'].predict(features_imp)[0]
            proba = models['diabetes']['model'].predict_proba(features_imp)[0]

            st.markdown("### 📊 Prediction Result")
            if pred == 1:
                st.markdown(f'<div class="result-positive">⚠️ HIGH RISK: Diabetes Detected<br><small>Confidence: {proba[1]*100:.1f}%</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-negative">✅ LOW RISK: No Diabetes Detected<br><small>Confidence: {proba[0]*100:.1f}%</small></div>', unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                'Outcome': ['Non-Diabetic', 'Diabetic'],
                'Probability': [proba[0], proba[1]]
            })
            st.bar_chart(prob_df.set_index('Outcome'))


# ═══════════════════════════════════════════════
# ③ BREAST CANCER
# ═══════════════════════════════════════════════
elif disease == "🎗️ Breast Cancer":
    st.subheader("🎗️ Breast Cancer — Enter Tumour Measurements")
    st.info("💡 These values come from digitised images of fine needle aspirate (FNA) of breast masses.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Mean Values**")
        radius_mean         = st.number_input("Radius Mean",           5.0, 30.0, 14.0)
        texture_mean        = st.number_input("Texture Mean",          5.0, 45.0, 19.0)
        perimeter_mean      = st.number_input("Perimeter Mean",        40.0, 200.0, 92.0)
        area_mean           = st.number_input("Area Mean",             140.0, 2600.0, 655.0)
        smoothness_mean     = st.number_input("Smoothness Mean",       0.05, 0.20, 0.096)
        compactness_mean    = st.number_input("Compactness Mean",      0.01, 0.40, 0.104)
        concavity_mean      = st.number_input("Concavity Mean",        0.0, 0.45, 0.089)
        concave_points_mean = st.number_input("Concave Points Mean",   0.0, 0.25, 0.049)
        symmetry_mean       = st.number_input("Symmetry Mean",         0.10, 0.35, 0.181)
        fractal_dim_mean    = st.number_input("Fractal Dimension Mean",0.04, 0.10, 0.063)

    with col2:
        st.markdown("**SE Values**")
        radius_se        = st.number_input("Radius SE",          0.1, 3.0, 0.41)
        texture_se       = st.number_input("Texture SE",         0.3, 5.0, 1.22)
        perimeter_se     = st.number_input("Perimeter SE",       0.7, 22.0, 2.87)
        area_se          = st.number_input("Area SE",            6.0, 550.0, 40.0)
        smoothness_se    = st.number_input("Smoothness SE",      0.001, 0.03, 0.007)
        compactness_se   = st.number_input("Compactness SE",     0.001, 0.15, 0.025)
        concavity_se     = st.number_input("Concavity SE",       0.0, 0.40, 0.032)
        concave_pts_se   = st.number_input("Concave Points SE",  0.0, 0.06, 0.012)
        symmetry_se      = st.number_input("Symmetry SE",        0.007, 0.08, 0.020)
        fractal_dim_se   = st.number_input("Fractal Dimension SE",0.0008, 0.03, 0.004)

    with col3:
        st.markdown("**Worst Values**")
        radius_worst         = st.number_input("Radius Worst",          7.0, 40.0, 16.0)
        texture_worst        = st.number_input("Texture Worst",         10.0, 55.0, 25.0)
        perimeter_worst      = st.number_input("Perimeter Worst",       50.0, 260.0, 107.0)
        area_worst           = st.number_input("Area Worst",            180.0, 4300.0, 880.0)
        smoothness_worst     = st.number_input("Smoothness Worst",      0.07, 0.25, 0.132)
        compactness_worst    = st.number_input("Compactness Worst",     0.02, 1.10, 0.254)
        concavity_worst      = st.number_input("Concavity Worst",       0.0, 1.30, 0.272)
        concave_pts_worst    = st.number_input("Concave Points Worst",  0.0, 0.30, 0.115)
        symmetry_worst       = st.number_input("Symmetry Worst",        0.15, 0.70, 0.290)
        fractal_dim_worst    = st.number_input("Fractal Dimension Worst",0.05, 0.25, 0.084)

    if st.button("🔍 Predict Breast Cancer", use_container_width=True):
        if models['breast'] is None:
            st.error("❌ Breast cancer model not found. Please run the breast cancer notebook first to generate breast_cancer_model.pkl")
        else:
            features = np.array([[
                radius_mean, texture_mean, perimeter_mean, area_mean,
                smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
                symmetry_mean, fractal_dim_mean,
                radius_se, texture_se, perimeter_se, area_se,
                smoothness_se, compactness_se, concavity_se, concave_pts_se,
                symmetry_se, fractal_dim_se,
                radius_worst, texture_worst, perimeter_worst, area_worst,
                smoothness_worst, compactness_worst, concavity_worst, concave_pts_worst,
                symmetry_worst, fractal_dim_worst
            ]])
            features_scaled = models['breast']['scaler'].transform(features)
            pred  = models['breast']['model'].predict(features_scaled)[0]
            proba = models['breast']['model'].predict_proba(features_scaled)[0]

            st.markdown("### 📊 Prediction Result")
            if pred == 1:
                st.markdown(f'<div class="result-positive">⚠️ HIGH RISK: Malignant Tumour Detected<br><small>Confidence: {proba[1]*100:.1f}%</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-negative">✅ LOW RISK: Benign Tumour Detected<br><small>Confidence: {proba[0]*100:.1f}%</small></div>', unsafe_allow_html=True)

            prob_df = pd.DataFrame({
                'Outcome': ['Benign', 'Malignant'],
                'Probability': [proba[0], proba[1]]
            })
            st.bar_chart(prob_df.set_index('Outcome'))

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#95a5a6; font-size:0.85rem;'>
    Built by <strong>Bhardwaja</strong> · MSc Artificial Intelligence & Business Intelligence · University of Leicester<br>
    Models: Random Forest · SVM · Logistic Regression · Gradient Boosting
</div>
""", unsafe_allow_html=True)
