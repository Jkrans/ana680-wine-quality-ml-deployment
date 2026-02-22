from flask import Flask, request, jsonify
import joblib
import os

FEATURES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

app = Flask(__name__)
model = joblib.load("model.joblib")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify(error=f"Missing features: {missing}"), 400

    input_data = [[data[f] for f in FEATURES]]
    prediction = float(model.predict(input_data)[0])

    return jsonify(predicted_quality=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)