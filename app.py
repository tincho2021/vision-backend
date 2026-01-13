import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import tempfile

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ===============================
# ENV
# ===============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

reference_embedding = None

# ===============================
# UTILS
# ===============================
def _get_uploaded_image():
    if "image" in request.files:
        f = request.files["image"]
    elif "file" in request.files:
        f = request.files["file"]
    else:
        return None

    if f.filename == "":
        return None

    try:
        return Image.open(f.stream).convert("RGB")
    except Exception:
        return None

def image_to_embedding(image):
    image = image.resize((128, 128)).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return arr.flatten()

def cosine_similarity(a, b):
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return dot / (na * nb) if na and nb else 0.0

def send_telegram_photo(image, caption):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        image.save(tmp.name, format="JPEG", quality=90)

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        with open(tmp.name, "rb") as photo:
            r = requests.post(
                url,
                data={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "caption": caption
                },
                files={"photo": photo},
                timeout=15
            )
        return r.status_code == 200

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/reference", methods=["POST"])
def set_reference():
    global reference_embedding

    img = _get_uploaded_image()
    if img is None:
        return jsonify(ok=False, error="No image provided"), 400

    reference_embedding = image_to_embedding(img)
    return jsonify(ok=True, message="Referencia seteada correctamente")

@app.route("/analyze", methods=["POST"])
def analyze():
    global reference_embedding

    if reference_embedding is None:
        return jsonify(ok=False, error="Referencia no seteada"), 400

    img = _get_uploaded_image()
    if img is None:
        return jsonify(ok=False, error="No image provided"), 400

    try:
        threshold = float(request.form.get("threshold", "0.25"))
    except ValueError:
        threshold = 0.25

    current_embedding = image_to_embedding(img)

    similarity = cosine_similarity(reference_embedding, current_embedding)
    change_score = 1.0 - similarity
    critical_change = change_score >= threshold

    telegram_sent = False

    if critical_change:
        caption = (
            "🚨 CAMBIO CRÍTICO DETECTADO\n\n"
            f"Change score: {change_score:.3f}\n"
            f"Similarity: {similarity:.3f}\n"
            f"Threshold: {threshold:.2f}"
        )
        telegram_sent = send_telegram_photo(img, caption)

    return jsonify(
        ok=True,
        similarity=round(similarity, 4),
        change_score=round(change_score, 4),
        threshold=round(threshold, 2),
        critical_change=critical_change,
        telegram_sent=telegram_sent
    )

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
