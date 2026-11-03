from flask import Flask, jsonify, request
import joblib, numpy as np

app = Flask(__name__)
# Memuat model Green (Random Forest)
model = joblib.load("model_green.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get("data", [])
    pred = model.predict([np.array(data)]).tolist()
    # Mengembalikan response version "Green"
    return jsonify({"version": "Green", "prediction": pred})

if __name__ == "__main__":
    # Dijalankan di port 5001 (atau 5000 sesuai kebutuhan strategi deployment)
    app.run(host="0.0.0.0", port=5001)
