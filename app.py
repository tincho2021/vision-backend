import os
import io
import base64
import requests
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# =========================
# Config
# =========================
REFERENCE_IMAGE_PATH = "reference.jpg"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

DEFAULT_THRESHOLD = 0.25


# =========================
# Utils
# =========================
def load_image_from_base64(data):
    header, encoded = data.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(img_bytes)).convert("L")


def send_telegram_alert(change_score):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    msg = f"🚨 Cambio crítico detectado\nChange score: {change_score:.3f}"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": TELEGRAM_CHAT_ID,
        "text": msg
    })


def compare_images(img1, img2):
    img1 = np.array(img1.resize((300, 300)))
    img2 = np.array(img2.resize((300, 300)))
    similarity = ssim(img1, img2)
    change_score = 1.0 - similarity
    return similarity, change_score


# =========================
# Routes
# =========================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/set_reference", methods=["POST"])
def set_reference():
    data = request.json
    img = load_image_from_base64(data["image"])
    img.save(REFERENCE_IMAGE_PATH)

    return jsonify({
        "ok": True,
        "message": "Referencia seteada correctamente"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    if not os.path.exists(REFERENCE_IMAGE_PATH):
        return jsonify({"ok": False, "error": "Referencia no configurada"}), 400

    data = request.json
    threshold = float(data.get("threshold", DEFAULT_THRESHOLD))

    current_img = load_image_from_base64(data["image"])
    ref_img = Image.open(REFERENCE_IMAGE_PATH).convert("L")

    similarity, change_score = compare_images(ref_img, current_img)

    critical = change_score > threshold

    if critical:
        send_telegram_alert(change_score)

    return jsonify({
        "ok": True,
        "similarity": round(similarity, 4),
        "change_score": round(change_score, 4),
        "threshold": threshold,
        "critical_change": critical
    })


# =========================
# Static files
# =========================
@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)


# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
