from flask import Flask, request, jsonify
import requests
import os
import numpy as np
import hashlib

app = Flask(__name__)

# ================= CONFIG =================
HF_TOKEN = os.environ["HF_TOKEN"]
TG_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TG_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

HF_MODEL = "google/vit-base-patch16-224"
HF_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/octet-stream"
}

# ================= STATE =================
REFERENCE_EMBED = None
REFERENCE_HASH = None

THRESH_CRITICAL = 0.70
THRESH_LEVE = 0.90

# ================= HELPERS =================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def send_telegram(text, image_bytes=None):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    requests.post(url, json={
        "chat_id": TG_CHAT_ID,
        "text": text
    })

    if image_bytes:
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
        requests.post(url, files={
            "photo": image_bytes
        }, data={
            "chat_id": TG_CHAT_ID
        })

# ================= ROUTE =================
@app.route("/analyze", methods=["POST"])
def analyze():
    global REFERENCE_EMBED, REFERENCE_HASH

    img = request.data
    if not img:
        return jsonify({"ok": False, "error": "No image"}), 400

    # hash para evitar reprocesar la misma imagen
    h = hashlib.md5(img).hexdigest()
    if h == REFERENCE_HASH:
        return jsonify({"ok": True, "status": "same_image"})

    r = requests.post(HF_URL, headers=HEADERS, data=img, timeout=60)
    data = r.json()

    if not isinstance(data, list) or not data:
        return jsonify({"ok": False, "error": "Invalid HF response"}), 500

    vec = np.array([x["score"] for x in data])

    if REFERENCE_EMBED is None:
        REFERENCE_EMBED = vec
        REFERENCE_HASH = h
        return jsonify({"ok": True, "status": "reference_set"})

    sim = cosine(vec, REFERENCE_EMBED)

    if sim < THRESH_CRITICAL:
        send_telegram(
            f"🚨 CAMBIO CRÍTICO DETECTADO\nSimilitud: {sim:.2f}",
            img
        )
        return jsonify({"ok": True, "status": "critical", "similarity": sim})

    # cambio leve → actualizar referencia
    REFERENCE_EMBED = vec
    REFERENCE_HASH = h

    return jsonify({
        "ok": True,
        "status": "stable",
        "similarity": sim
    })

# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
