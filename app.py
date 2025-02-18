from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and label encoder
model = joblib.load("student_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return "Welcome to the Student Performance Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    math_score = data["math"]
    reading_score = data["reading"]
    writing_score = data["writing"]

    features = np.array([[math_score, reading_score, writing_score]])
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    return jsonify({"Predicted Race/Ethnicity": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
