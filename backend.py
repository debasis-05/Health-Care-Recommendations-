!pip install flask-ngrok scikit-learn pandas

from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import numpy as np
import joblib

# Load or define your model here
from sklearn.ensemble import RandomForestClassifier

# For demo: Train simple model
import pandas as pd
X = pd.DataFrame({
    "age": [60, 75, 50],
    "gender": [1, 0, 0],
    "diabetes": [1, 1, 0],
    "heart_disease": [1, 0, 1],
    "hypertension": [0, 1, 1],
    "num_visits": [4, 10, 3],
    "length_of_stay": [3, 7, 2]
})
y = [1, 1, 0]
model = RandomForestClassifier()
model.fit(X, y)

# Flask app
app = Flask(__name__)
run_with_ngrok(app)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([
        data["age"],
        data["gender"],
        data["diabetes"],
        data["heart_disease"],
        data["hypertension"],
        data["num_visits"],
        data["length_of_stay"]
    ]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    return jsonify({"readmitted": int(prediction)})

app.run()