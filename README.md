# 💳 Enterprise Credit Card Fraud Detection Shield
### *Real-Time Transaction Integrity Analytics Layer powered by XGBoost*

[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

---

## 💼 Business Architecture & Context
In high-volume digital transaction ecosystems processing millions of events daily, human audit pipelines are bottlenecked by scale constraints. This project implements a decoupled, high-throughput fraud intervention layer designed to evaluate incoming network payloads in real-time, flag anomalous signatures, and isolate high-risk actions—ensuring compliance analysts review only the riskiest vectors.

### 📊 Dataset Parameters
* **Total Transactions Evaluated:** 284,807
* **Fraudulent Class Flag ($Y=1$):** 492 instances (**0.17%** minority footprint)
* **Legitimate Class Flag ($Y=0$):** 284,315 instances (**99.83%** baseline matrix)
* **Feature Topology:** 28 Obfuscated PCA Dimensional Components ($V_1$ to $V_{28}$), Temporal Interval (`Time`), and Fiat Volume (`Amount`).

---

## 🚀 Live Interactive Simulator
The system is deployed live on Streamlit Cloud, simulating real-time secure API network token ingestion:  
🔗 **Live Application Link:** [https://manjotkaur06-credit-card-fraud-detection.streamlit.app/](https://credit-card-fraud-detection-uytpykxj6jry9ea6qkagws.streamlit.app/)

---

## ⚡ Engineering & Machine Learning Roadmap

### 1. Exploratory Insights & Signal Extraction
* **Temporal Clustering:** Analytical sweeps revealed severe fraud clusters localized during late-night cycles. Created an `is_night` structural flag (10 PM – 5 AM).
* **Behavioral Scaling:** Small transaction thresholds correlated strongly with structural automated card-testing footprints. Implemented `is_small_amount` (< €10) bounds.
* **Statistical Deviation:** Computed global matrix z-scores across high-variance components ($V_{14}, V_{17}$) to isolate mathematical anomalies using `v_zscore_max`.

### 2. Imbalance Resolution Protocols
* **Data Leakage Mitigation:** Synthetic Minority Over-sampling Technique (**SMOTE**) was applied **strictly after** executing the train-test split array to prevent data contamination.
* **Cost-Sensitive Optimization:** Tuned the algorithmic optimization parameter within the gradient booster to penalize minority misclassifications:

$$\text{scale\_pos\_weight} = \frac{\text{Negative Counts}}{\text{Positive Counts}} = \frac{284,315}{492} = 577$$

* **Evaluation Benchmark:** Replaced standard ROC-AUC targets with Precision-Recall AUC (**PR-AUC**) to effectively penalize False Negatives under extreme imbalance constraints.

---

## 📈 Model Performance Matrix

Through rigorous cross-validation and optimization cycles across Logistic Regression, Random Forest, and Unsupervised Isolation Forests, the **XGBoost Classifier Architecture** delivered superior enterprise benchmarks:

| Evaluation Metric | Production Value | Business Optimization Context |
| :--- | :---: | :--- |
| **ROC-AUC** | `0.98` | Global spatial separator efficiency |
| **PR-AUC** | `0.87` | True reliability benchmark for skewed data |
| **Recall / Sensitivity** | `84%` | **Suppresses False Negatives** (Intercepts leaks) |
| **Precision** | `89%` | Minimizes analyst overhead from false alarms |
| **Classification Threshold** | `0.35` | Mathematically optimized decision cutoff |

---

## 💰 Projected Financial Impact Statement

Assuming a standard high-velocity processing gateway handling **10,000,000 daily transactions**:

* **Base Daily Fraud Inflow:** $10,000,000 \times 0.17\% = 17,000\text{ vectors}$
* **Legacy System Capture Rate (40%):** $6,800\text{ intercepted vectors}$
* **Production Engine Capture Rate (84%):** **$14,280\text{ intercepted vectors}$**
* **Net New Anomalies Caught:** **+7,480 vectors / day**
* **Average Ticket Chargeback Cost:** ₹12,000
* **Estimated Systemic Savings:** $7,480 \times \text{₹}12,000 = \mathbf{\text{₹}8.97\text{ Crore } (\approx \text{₹}9\text{ Crore / Day)}}$

---

## 🏗️ Technical Directory Architecture
```directory
├── data/
│   └── raw/                   # Secure raw ledger storage (Git-ignored)
├── models/
│   ├── scaler.pkl             # Serialized StandardScaler normalization weights
│   └── xgb_model.pkl          # Compressed Production XGBoost weights
├── notebooks/
│   ├── 01_eda.ipynb           # Temporal and transaction volume clustering analysis
│   ├── 02_feature_engineering.ipynb # Custom vector transformations and pipelines
│   └── 03_model_training.ipynb# Model grid benchmarks and PR-AUC calibration
├── src/
│   ├── dashboard.py           # Streamlit Live Simulation User Interface
│   ├── features.py            # Array conversion transformations
│   ├── predict.py             # Inference pipeline execution script
│   └── preprocess.py          # Real-time token matrix scaling handler
├── requirements.txt           # Cloud container package specifications
└── README.md                  # Project Documentation
