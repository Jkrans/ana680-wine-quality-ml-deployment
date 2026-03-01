from flask import Flask, jsonify, request, render_template
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


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    # If request is coming from the HTML form:
    if request.form:
        try:
            data = {f: float(request.form.get(f, "")) for f in FEATURES}
        except ValueError:
            return render_template("index.html", error="All inputs must be numbers.")

        pred = float(model.predict([[data[f] for f in FEATURES]])[0])
        return render_template("index.html", prediction=pred)

    # If request is JSON (curl/postman):
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify(error=f"Missing features: {missing}"), 400

    input_data = [[data[f] for f in FEATURES]]
    prediction = float(model.predict(input_data)[0])
    return jsonify(predicted_quality=prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)