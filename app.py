from flask import Flask, request, jsonify, render_template
import requests
import os
import numpy as np

app = Flask(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL = "google/vit-base-patch16-224"

HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/octet-stream",
    "Accept": "application/json",
    "x-wait-for-model": "true"
}

def cosine_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(img_bytes):
    r = requests.post(
        HF_URL,
        headers=HEADERS,
        data=img_bytes,
        params={"feature_extraction": True},
        timeout=120
    )
    r.raise_for_status()
    return r.json()[0]

@app.route("/compare", methods=["POST"])
def compare():
    if "before" not in request.files or "after" not in request.files:
        return jsonify({"ok": False, "error": "Need before and after images"}), 400

    img1 = request.files["before"].read()
    img2 = request.files["after"].read()

    try:
        emb1 = get_embedding(img1)
        emb2 = get_embedding(img2)
        dist = cosine_distance(emb1, emb2)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

    behavior = "changed" if dist > 0.08 else "stable"

    return jsonify({
        "ok": True,
        "behavior": behavior,
        "difference": round(dist, 4),
        "model": HF_MODEL
    })

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
