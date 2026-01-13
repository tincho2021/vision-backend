import os
import io
import base64
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ===============================
# ENV
# ===============================
HF_TOKEN = os.getenv("HF_TOKEN")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

reference_embedding = None

# ===============================
# UTILS
# ===============================
def image_to_embedding(image: Image.Image):
    """
    Convierte una imagen a un embedding simple (baseline).
    No interpreta, solo representa.
    """
    image = image.resize((224, 224)).convert("RGB")
    arr = np.asarray(image).astype(np.float32)
    arr = arr / 255.0
    return arr.flatten()

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:
        pass

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/reference", methods=["POST"])
def set_reference():
    global reference_embedding

    if "image" not in request.files:
        return jsonify(ok=False, error="No image provided"), 400

    image = Image.open(request.files["image"])
    reference_embedding = image_to_embedding(image)

    return jsonify(
        ok=True,
        message="Referencia seteada correctamente"
    )

@app.route("/analyze", methods=["POST"])
def analyze():
    global reference_embedding

    if reference_embedding is None:
        return jsonify(ok=False, error="Referencia no seteada"), 400

    if "image" not in request.files:
        return jsonify(ok=False, error="No image provided"), 400

    try:
        threshold = float(request.form.get("threshold", 0.25))
    except ValueError:
        threshold = 0.25

    image = Image.open(request.files["image"])
    current_embedding = image_to_embedding(image)

    similarity = cosine_similarity(
        [reference_embedding],
        [current_embedding]
    )[0][0]

    change_score = float(1.0 - similarity)
    critical_change = bool(change_score >= threshold)

    # 🚨 ALERTA TELEGRAM
    if critical_change:
        send_telegram_alert(
            "🚨 CAMBIO CRÍTICO DETECTADO\n"
            f"Change score: {round(change_score, 3)}\n"
            f"Threshold: {threshold}"
        )

    return jsonify(
        ok=True,
        similarity=round(similarity, 4),
        change_score=round(change_score, 4),
        threshold=threshold,
        critical_change=critical_change
    )

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
