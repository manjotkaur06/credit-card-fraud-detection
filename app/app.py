from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("../models/xgb_model.pkl", "rb"))

scaler = pickle.load(open("../models/scaler.pkl", "rb"))

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]

        # Convert to numpy array
        features_array = np.zeros((1, 30))

        features_array[0, :len(features)] = features

        # Scale input
        scaled_features = scaler.transform(features_array)

        # Prediction
        prediction = model.predict(scaled_features)

        # Result
        if prediction[0] == 1:
            result = "Fraud Transaction 🚨"
        else:
            result = "Normal Transaction ✅"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return str(e)

# Run app
if __name__ == "__main__":
    app.run(debug=True)