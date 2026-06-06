import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="Fraud Detection Shield", page_icon="💳")

st.title("💳 Enterprise Credit Card Fraud Detection Shield")
st.write("Real-Time Transaction Integrity Analytics Layer powered by XGBoost")
st.write("---")

# Direct relative tracking from cloud execution setup
MODEL_PATH = "models/xgb_model.pkl"
SCALER_PATH = "models/scaler.pkl"

@st.cache_resource
def load_production_assets():
    # If direct path succeeds
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return pickle.load(open(MODEL_PATH, "rb")), pickle.load(open(SCALER_PATH, "rb"))
    
    # Standalone absolute path for Streamlit Linux Cloud container
    cloud_model = "/mount/src/credit-card-fraud-detection/models/xgb_model.pkl"
    cloud_scaler = "/mount/src/credit-card-fraud-detection/models/scaler.pkl"
    
    if os.path.exists(cloud_model) and os.path.exists(cloud_scaler):
        return pickle.load(open(cloud_model, "rb")), pickle.load(open(cloud_scaler, "rb"))
        
    raise FileNotFoundError("Pre-trained structural model assets are missing in working directory paths.")

try:
    model, scaler = load_production_assets()
    st.success("🎯 All Production Assets Loaded Flawlessly!")
except Exception as e:
    st.error(f"Asset Ingestion Failure: {e}")

# Live Data Verification Terminal
st.subheader("📁 Live Transaction Feature Array")
default_features = "0.0, -1.35, 1.22, -0.67, 1.01, -0.87, 0.45, -0.12, 0.98, -0.56, 0.22, -0.11, 0.88, -0.45, 0.12, -0.01, 0.23, -0.44, 0.11, -0.05, 0.01, -0.02, 0.05, -0.12, 0.11, -0.32, 0.01, -0.05, 15.25, 0.0"
user_input = st.text_area("Input Feature Matrix (V1 to V30 / Amount):", default_features, height=100)

if st.button("Evaluate Transaction Risk Profile"):
    try:
        raw_features = [float(x.strip()) for x in user_input.split(",") if x.strip() != ""]
        features_array = np.zeros((1, 30))
        features_array[0, :len(raw_features)] = raw_features[:30]
        
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)
        
        st.write("---")
        if prediction[0] == 1:
            st.error("🚨 FRAUD ANOMALY FLAG! High risk score detected. Action required.")
        else:
            st.success("✅ TRANSACTION SECURE. The input vector conforms to safe consumer signatures.")
    except Exception as e:
        st.error(f"Execution Failure Status: {e}")