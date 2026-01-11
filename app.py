from flask import Flask, request, jsonify, render_template
import requests
import os
import time

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN no definido")

HF_MODEL = "google/vit-base-patch16-224"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "image/jpeg",
    "Accept": "application/json",
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image"}), 400

    img = request.files["image"].read()

    for attempt in range(3):
        r = requests.post(
            HF_URL,
            headers=HEADERS,
            data=img,
            timeout=40
        )

        if "application/json" not in r.headers.get("Content-Type", ""):
            time.sleep(4)
            continue

        data = r.json()

        if isinstance(data, dict) and data.get("error"):
            time.sleep(4)
            continue

        if isinstance(data, list) and len(data) > 0:
            top = data[0]
            return jsonify({
                "ok": True,
                "label": top["label"],
                "score": round(top["score"] * 100, 2),
                "model": HF_MODEL
            })

    return jsonify({
        "ok": False,
        "error": "Model sleeping or HF rejected request"
    }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
