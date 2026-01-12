from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# ================= CONFIG =================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN no definido")

HF_MODEL = "google/vit-base-patch16-224"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/octet-stream",
    "Accept": "application/json",
    "x-wait-for-model": "true"
}

REFERENCE = None

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"ok": True, "service": "Behavior-Action Vision"})

@app.route("/analyze", methods=["POST"])
def analyze():
    image = request.files.get("image")
    if not image:
        return jsonify({"ok": False, "error": "No image"}), 400

    r = requests.post(HF_URL, headers=HEADERS, data=image.read(), timeout=120)
    data = r.json()

    if not isinstance(data, list) or not data:
        return jsonify({"ok": False, "error": "Invalid model response"}), 500

    top = data[0]
    return jsonify({
        "ok": True,
        "label": top["label"],
        "confidence": round(top["score"] * 100, 2),
        "model": HF_MODEL
    })

@app.route("/reference", methods=["POST"])
def set_reference():
    global REFERENCE
    image = request.files.get("image")
    if not image:
        return jsonify({"ok": False, "error": "No image"}), 400

    r = requests.post(HF_URL, headers=HEADERS, data=image.read(), timeout=120)
    data = r.json()

    if not isinstance(data, list) or not data:
        return jsonify({"ok": False, "error": "Invalid model response"}), 500

    REFERENCE = data[0]
    return jsonify({"ok": True, "reference": REFERENCE})

@app.route("/behavior", methods=["POST"])
def behavior():
    if not REFERENCE:
        return jsonify({"ok": False, "error": "Reference not set"}), 400

    image = request.files.get("image")
    if not image:
        return jsonify({"ok": False, "error": "No image"}), 400

    r = requests.post(HF_URL, headers=HEADERS, data=image.read(), timeout=120)
    data = r.json()

    if not isinstance(data, list) or not data:
        return jsonify({"ok": False, "error": "Invalid model response"}), 500

    current = data[0]

    changed = current["label"] != REFERENCE["label"]

    return jsonify({
        "ok": True,
        "behavior": "changed" if changed else "stable",
        "current": current,
        "reference": REFERENCE
    })

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
