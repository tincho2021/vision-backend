from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")  # ponelo en Render > Env Vars
HF_MODEL = "google/vit-base-patch16-224"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    img = request.files["image"].read()

    r = requests.post(
        HF_URL,
        headers=HEADERS,
        data=img
    )

    return jsonify(r.json())

@app.route("/health")
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
