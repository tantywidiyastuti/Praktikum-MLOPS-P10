
from flask import Flask, jsonify, request
import joblib, numpy as np

app = Flask(__name__)
model = joblib.load("model_blue.pkl")

@app.route('/predict', methods=['POST'])
def predict():
  data = request.json.get("data", [])
  pred = model.predict([np.array(data)]).tolist()
  return jsonify({"version": "Blue", "prediction": pred})

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=5000)
