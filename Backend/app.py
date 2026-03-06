from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask_cors import CORS
import os

with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)
with open("features_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])

    missing_features = [f for f in feature_order if f not in input_data.columns]
    if missing_features:
        return jsonify({"error": f"Missing features: {missing_features}"}), 400

    input_data = input_data[feature_order]

    categorical_cols = ['Location']
    for col in categorical_cols:
        if input_data[col][0] not in label_encoders[col].classes_:
            return jsonify({"error": f"Unknown category in {col}: {input_data[col][0]}"}), 400
        input_data[col] = label_encoders[col].transform(input_data[col])

    prediction = model.predict(input_data)[0]
    predicted_label = label_encoders['BusinessOutcome'].inverse_transform([prediction])[0]

    return jsonify({"prediction": predicted_label})


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

