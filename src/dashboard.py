import streamlit as st
import numpy as np
import pickle
import os

# Professional Banking UI Theme Configuration
st.set_page_config(page_title="Fraud Shield Enterprise", page_icon="💳", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size:2.2rem !important; color: #1E3A8A; font-weight: 700; }
    .sub-header { font-size:1.1rem !important; color: #4B5563; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💳 Risk Management Core: Fraud Detection Shield</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise-grade transaction security simulation pipeline driven by XGBoost Engine</div>', unsafe_allow_html=True)
st.write("---")

# Model Paths
MODEL_PATH = "models/xgb_model.pkl"
SCALER_PATH = "models/scaler.pkl"

@st.cache_resource
def load_production_assets():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        return pickle.load(open(MODEL_PATH, "rb")), pickle.load(open(SCALER_PATH, "rb"))
    
    cloud_model = "/mount/src/credit-card-fraud-detection/models/xgb_model.pkl"
    cloud_scaler = "/mount/src/credit-card-fraud-detection/models/scaler.pkl"
    
    if os.path.exists(cloud_model) and os.path.exists(cloud_scaler):
        return pickle.load(open(cloud_model, "rb")), pickle.load(open(cloud_scaler, "rb"))
    raise FileNotFoundError("Assets missing.")

try:
    model, scaler = load_production_assets()
    st.sidebar.success("🟢 System Status: SECURE / Core Operational")
except Exception as e:
    st.sidebar.error("🔴 System Status: Asset Ingestion Failure")

# Sidebar Meta Information
st.sidebar.header("System Specifications")
st.sidebar.info("""
- **Model Framework:** XGBoost Classifier
- **Input Dimension:** 30 Features (PCA Transformed Vectors + Amount)
- **Target Optimization:** Latency minimization & False Negative suppression
""")

# Splitting Interface into Two Columns for Professional Look
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Transaction Meta")
    merchant_id = st.text_input("Merchant Gateway ID", "E-COMM-MNT-9841")
    input_amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=125.50)
    
    st.subheader("⚙️ Simulation Trigger")
    run_simulation = st.button("Execute Vector Risk Assessment", use_container_width=True)

with col2:
    st.subheader("📡 Real-Time API Data Vector Stream")
    st.caption("Simulating incoming transaction payload string containing PCA components V1 to V28 + Normalized Amount attributes:")
    
    # Pre-filled mock values
    default_features = "0.0, -1.35, 1.22, -0.67, 1.01, -0.87, 0.45, -0.12, 0.98, -0.56, 0.22, -0.11, 0.88, -0.45, 0.12, -0.01, 0.23, -0.44, 0.11, -0.05, 0.01, -0.02, 0.05, -0.12, 0.11, -0.32, 0.01, -0.05, 15.25, 0.0"
    user_input = st.text_area("Raw Secure Network Token Input", default_features, height=135)

# Processing Logic Placement
if run_simulation:
    try:
        raw_features = [float(x.strip()) for x in user_input.split(",") if x.strip() != ""]
        features_array = np.zeros((1, 30))
        features_array[0, :len(raw_features)] = raw_features[:30]
        
        # Inject the manual user amount into the final element if needed
        features_array[0, -2] = input_amount 
        
        scaled_features = scaler.transform(features_array)
        prediction = model.predict(scaled_features)
        
        st.write("---")
        st.subheader("🛡️ Gateway Security Verdict")
        
        if prediction[0] == 1:
            st.error(f"🚨 FRAUD ANOMALY BLOCKED! Transaction on Merchant {merchant_id} for ${input_amount} violates behavioral baseline matrix rules.")
        else:
            st.success(f"✅ TRANSACTION AUTHENTICATED. Authorization token issued successfully for Merchant {merchant_id}.")
            
    except Exception as e:
        st.error(f"System Operational Error: {e}")