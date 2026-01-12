from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

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

# ================= MEMORY =================
REFERENCE = None  # guarda {label, confidence}

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/set_reference", methods=["POST"])
def set_reference():
    global REFERENCE

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    img = request.files["image"].read()

    r = requests.post(HF_URL, headers=HEADERS, data=img, timeout=120)
    data = r.json()

    if not isinstance(data, list) or not data:
        return jsonify({"ok": False, "error": "Invalid model response"}), 500

    top = data[0]

    REFERENCE = {
        "label": top.get("label"),
        "confidence": round(top.get("score", 0) * 100, 2)
    }

    return jsonify({
        "ok": True,
        "reference": REFERENCE
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    img = request.files["image"].read()

    r = requests.post(HF_URL, headers=HEADERS, data=img, timeout=120)
    data = r.json()

    if not isinstance(data, list) or not data:
        return jsonify({"ok": False, "error": "Invalid model response"}), 500

    top = data[0]
    label = top.get("label")
    confidence = round(top.get("score", 0) * 100, 2)

    behavior = "unknown"

    if REFERENCE:
        if label != REFERENCE["label"]:
            behavior = "changed"
        elif abs(confidence - REFERENCE["confidence"]) > 20:
            behavior = "changed"
        else:
            behavior = "stable"

    return jsonify({
        "ok": True,
        "label": label,
        "confidence": confidence,
        "behavior": behavior,
        "reference": REFERENCE,
        "model": HF_MODEL
    })

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
