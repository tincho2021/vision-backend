import os
import io
import requests
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

app = Flask(__name__, static_folder="static")
CORS(app)

# ==========================
# CONFIG
# ==========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

REFERENCE_IMAGE = None
REFERENCE_VECTOR = None

# ==========================
# UTILS
# ==========================

def image_to_vector(img: Image.Image) -> np.ndarray:
    img = img.resize((128, 128)).convert("L")
    arr = np.array(img).astype("float32")
    arr /= 255.0
    return arr.flatten()

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def send_telegram_alert(text, image_bytes=None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram no configurado")
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    if image_bytes:
        files = {
            "photo": ("alert.jpg", image_bytes)
        }
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "caption": text
        }
        requests.post(f"{base_url}/sendPhoto", data=data, files=files)
    else:
        data = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text
        }
        requests.post(f"{base_url}/sendMessage", data=data)

# ==========================
# ROUTES
# ==========================

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/reference", methods=["POST"])
def set_reference():
    global REFERENCE_IMAGE, REFERENCE_VECTOR

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    img_file = request.files["image"]
    img = Image.open(img_file.stream)

    REFERENCE_IMAGE = img.copy()
    REFERENCE_VECTOR = image_to_vector(img)

    return jsonify({
        "ok": True,
        "message": "Referencia seteada correctamente"
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    global REFERENCE_VECTOR

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    if REFERENCE_VECTOR is None:
        return jsonify({"ok": False, "error": "Referencia no seteada"}), 400

    img_file = request.files["image"]
    img = Image.open(img_file.stream)
    current_vector = image_to_vector(img)

    similarity = cosine_similarity(REFERENCE_VECTOR, current_vector)
    change_score = 1.0 - similarity

    CRITICAL_THRESHOLD = 0.25
    critical = change_score > CRITICAL_THRESHOLD

    result = {
        "ok": True,
        "similarity": round(similarity, 4),
        "change_score": round(change_score, 4),
        "critical_change": critical
    }

    if critical:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        send_telegram_alert(
            text=(
                "🚨 *ALERTA DE CAMBIO DETECTADO*\n\n"
                f"🔍 Change score: {change_score:.2f}\n"
                f"📉 Similaridad: {similarity:.2f}"
            ),
            image_bytes=buffer.read()
        )

    return jsonify(result)

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
