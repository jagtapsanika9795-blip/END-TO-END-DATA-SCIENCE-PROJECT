from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return "Student Score Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    hours = data['hours_studied']
    attendance = data['attendance']

    prediction = model.predict([[hours, attendance]])

    return jsonify({
        "Predicted Marks": round(prediction[0], 2)
    })

if __name__ == "__main__":
    app.run(debug=True,port=5000)
