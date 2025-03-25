# main.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("model/random_forest_model.pkl")

# Define a route to check API status
@app.route("/")
def home():
    return "Water Potability Prediction API is running!"

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get data from request
        data = request.json
        # Convert JSON to DataFrame
        df = pd.DataFrame(data, index=[0])
        
        # Make prediction
        prediction = model.predict(df)
        result = "Potable" if prediction[0] == 1 else "Not Potable"
        
        # Return result
        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
