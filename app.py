# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

# Load model and vectorizer
with open("model/spam_classifier.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("email", "")
    if not email_text:
        return jsonify({"result": "Please enter email content."}), 400

    vectorized_text = vectorizer.transform([email_text])
    prediction = model.predict(vectorized_text)[0]

    result = "Spam" if prediction == 1 else "Not Spam"
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
