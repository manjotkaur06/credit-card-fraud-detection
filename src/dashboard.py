import streamlit as st
import numpy as np
import pickle
import os

# Page Configuration Styling
st.set_page_config(page_title="Fraud Detection Shield", page_icon="💳")

st.title("💳 Enterprise Credit Card Fraud Detection Shield")
st.write("Real-Time Transaction Integrity Analytics Layer powered by XGBoost")
st.write("---")

# Safe Paths Mapping according to image_fe87af.png structure
# Jab cloud par run hoga toh root directory se models directly read honge
MODEL_PATH = "models/xgb_model.pkl"
SCALER_PATH = "models/scaler.pkl"

@st.cache_resource
def load_production_assets():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        # Fallback tracking agar relative structures parent call se change hon
        return pickle.load(open("../models/xgb_model.pkl", "rb")), pickle.load(open("../models/scaler.pkl", "rb"))
    
    return pickle.load(open(MODEL_PATH, "rb")), pickle.load(open(SCALER_PATH, "rb"))

try:
    model, scaler = load_production_assets()
except Exception as e:
    st.error(f"Asset Ingestion Failure: Check if pkl models exist under models/ directory. Details: {e}")

# Main User Input Panel
st.subheader("📁 Live Transaction Feature Array")
st.write("Enter the 30 continuous numerical PCA feature components (separated by commas):")

# Pre-filled mock values for user ease
default_features = "0.0, -1.35, 1.22, -0.67, 1.01, -0.87, 0.45, -0.12, 0.98, -0.56, 0.22, -0.11, 0.88, -0.45, 0.12, -0.01, 0.23, -0.44, 0.11, -0.05, 0.01, -0.02, 0.05, -0.12, 0.11, -0.32, 0.01, -0.05, 15.25, 0.0"

user_input = st.text_area("Input Feature Matrix (V1 to V30 / Amount):", default_features, height=100)

if st.button("Evaluate Transaction Risk Profile"):
    try:
        # Array formatting exactly matching your app.py logic
        raw_features = [float(x.strip()) for x in user_input.split(",") if x.strip() != ""]
        
        features_array = np.zeros((1, 30))
        features_array[0, :len(raw_features)] = raw_features[:30]
        
        # Scale and Predict
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)
        
        st.write("---")
        # Visual Alerts matching fraud status flags
        if prediction[0] == 1:
            st.error("🚨 FRAUD ANOMALY FLAG! High risk score detected. Transaction suspended.")
        else:
            st.success("✅ TRANSACTION SECURE. The input vector conforms to normal consumer profiles.")
            
    except Exception as e:
        st.error(f"Execution Error: Ensure 30 numeric variables are populated. Error context: {e}")