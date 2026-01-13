import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# ===============================
# ENV (Render -> Environment)
# ===============================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")  # ponelo en Render
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # ponelo en Render

reference_embedding = None

# ===============================
# UTILS
# ===============================
def _get_uploaded_image():
    """
    Acepta 'image' o 'file' para ser compatible con distintas GUIs.
    """
    f = None
    if "image" in request.files:
        f = request.files["image"]
    elif "file" in request.files:
        f = request.files["file"]

    if f is None or f.filename == "":
        return None

    try:
        img = Image.open(f.stream).convert("RGB")
        return img
    except Exception:
        return None

def image_to_embedding(image: Image.Image):
    # Embedding simple (baseline). Podés mejorar después con zonas/ROI.
    image = image.resize((128, 128)).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return arr.flatten()

def cosine_similarity(a, b):
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)

def send_telegram_alert(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}

    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# ===============================
# ROUTES
# ===============================
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/health")
def health():
    return jsonify(ok=True, service="Behavior-Action Vision")

@app.route("/reference", methods=["POST"])
def set_reference():
    global reference_embedding

    img = _get_uploaded_image()
    if img is None:
        return jsonify(ok=False, error="No image provided (field must be 'image' or 'file')"), 400

    reference_embedding = image_to_embedding(img)

    return jsonify(ok=True, message="Referencia seteada correctamente")

@app.route("/analyze", methods=["POST"])
def analyze():
    global reference_embedding

    if reference_embedding is None:
        return jsonify(ok=False, error="Referencia no seteada"), 400

    img = _get_uploaded_image()
    if img is None:
        return jsonify(ok=False, error="No image provided (field must be 'image' or 'file')"), 400

    # threshold viene del slider (form) o default
    try:
        threshold = float(request.form.get("threshold", "0.25"))
    except ValueError:
        threshold = 0.25

    current_embedding = image_to_embedding(img)

    similarity = cosine_similarity(reference_embedding, current_embedding)
    change_score = 1.0 - similarity
    critical_change = bool(change_score >= threshold)  # bool nativo (serializable)

    # Si es crítico, alertar
    telegram_sent = False
    if critical_change:
        telegram_sent = send_telegram_alert(
            "🚨 CAMBIO CRÍTICO DETECTADO\n"
            f"change_score: {change_score:.4f}\n"
            f"similarity: {similarity:.4f}\n"
            f"threshold: {threshold:.2f}"
        )

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
