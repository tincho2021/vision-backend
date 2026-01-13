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
# CONFIG (Telegram)
# ==========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==========================
# REFERENCE (en RAM)
# ==========================
REFERENCE_VECTOR = None

# ==========================
# UTILS
# ==========================
def image_to_vector(img: Image.Image) -> np.ndarray:
    # Vector simple y rápido (sirve para "cambio/no cambio")
    img = img.resize((128, 128)).convert("L")
    arr = np.array(img).astype("float32") / 255.0
    return arr.flatten()

def cosine_similarity(a, b) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def send_telegram_alert(text: str, image_bytes: bytes | None = None):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram no configurado (faltan TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID)")
        return

    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    try:
        if image_bytes:
            files = {"photo": ("alert.jpg", image_bytes)}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": text}
            requests.post(f"{base_url}/sendPhoto", data=data, files=files, timeout=20)
        else:
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
            requests.post(f"{base_url}/sendMessage", data=data, timeout=20)
    except Exception as e:
        print("⚠️ Error enviando Telegram:", e)

def read_image_from_request() -> Image.Image:
    if "image" not in request.files:
        raise ValueError("No image provided")
    img_file = request.files["image"]
    return Image.open(img_file.stream)

def analyze_core(img: Image.Image):
    global REFERENCE_VECTOR

    if REFERENCE_VECTOR is None:
        return jsonify({"ok": False, "error": "Referencia no seteada"}), 400

    current_vector = image_to_vector(img)
    similarity = cosine_similarity(REFERENCE_VECTOR, current_vector)
    change_score = 1.0 - similarity

    # Ajustable
    CRITICAL_THRESHOLD = float(os.getenv("CRITICAL_THRESHOLD", "0.25"))
    critical = change_score > CRITICAL_THRESHOLD

    result = {
        "ok": True,
        "similarity": round(similarity, 4),
        "change_score": round(change_score, 4),
        "critical_change": bool(critical),
        "threshold": CRITICAL_THRESHOLD
    }

    if critical:
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        send_telegram_alert(
            text=(
                "🚨 ALERTA: cambio detectado\n"
                f"change_score={change_score:.2f}  similarity={similarity:.2f}"
            ),
            image_bytes=buffer.read()
        )

    return jsonify(result)

# ==========================
# ROUTES
# ==========================
@app.route("/health")
def health():
    return jsonify({"ok": True, "service": "Behavior-Action Vision"})

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/reference", methods=["POST"])
def set_reference():
    global REFERENCE_VECTOR
    try:
        img = read_image_from_request()
        REFERENCE_VECTOR = image_to_vector(img)
        return jsonify({"ok": True, "message": "Referencia seteada correctamente"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        img = read_image_from_request()
        return analyze_core(img)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# ✅ ALIAS para tu GUI actual (tu botón llama /behavior)
@app.route("/behavior", methods=["POST"])
def behavior():
    try:
        img = read_image_from_request()
        return analyze_core(img)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
