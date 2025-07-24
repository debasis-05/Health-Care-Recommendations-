from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Setup Flask
app = Flask(__name__, static_folder='.')
CORS(app)  # Enable cross-origin access from HTML

# Create simple dataset for demo model
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

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Serve HTML
@app.route("/")
def serve_html():
    return send_from_directory('.', 'New Text Document.html')

# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    try:
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
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
