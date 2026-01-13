from flask import Flask, request, jsonify, render_template
import os
import uuid
import json
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import requests

# ================= CONFIG =================
BASE_DIR = "devices"
os.makedirs(BASE_DIR, exist_ok=True)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN no definido")

# ================= APP =================
app = Flask(__name__)

# ================= HELPERS =================
def device_dir(device_id):
    path = os.path.join(BASE_DIR, device_id)
    os.makedirs(path, exist_ok=True)
    return path

def load_config(device_id):
    cfg_path = os.path.join(device_dir(device_id), "config.json")
    if not os.path.exists(cfg_path):
        return {
            "threshold": 0.20,
            "telegram_chat": None
        }
    with open(cfg_path, "r") as f:
        return json.load(f)

def save_config(device_id, cfg):
    with open(os.path.join(device_dir(device_id), "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

def decode_image(data):
    img = np.frombuffer(data, np.uint8)
    return cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)

def send_telegram(chat_id, image_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, "rb") as f:
        requests.post(url, data={
            "chat_id": chat_id,
            "caption": caption
        }, files={"photo": f})

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

# ---------- SET REFERENCE ----------
@app.route("/reference", methods=["POST"])
def set_reference():
    device_id = request.form.get("device_id")
    if not device_id or "image" not in request.files:
        return jsonify({"ok": False, "error": "device_id or image missing"}), 400

    img = request.files["image"].read()
    path = os.path.join(device_dir(device_id), "reference.jpg")

    with open(path, "wb") as f:
        f.write(img)

    return jsonify({"ok": True, "msg": "Referencia guardada"})

# ---------- ANALYZE ----------
@app.route("/analyze", methods=["POST"])
def analyze():
    device_id = request.form.get("device_id")
    chat_id = request.form.get("chat_id")

    if not device_id or "image" not in request.files:
        return jsonify({"ok": False, "error": "device_id or image missing"}), 400

    cfg = load_config(device_id)

    if chat_id:
        cfg["telegram_chat"] = chat_id
        save_config(device_id, cfg)

    threshold = float(request.form.get("threshold", cfg["threshold"]))

    ref_path = os.path.join(device_dir(device_id), "reference.jpg")
    if not os.path.exists(ref_path):
        return jsonify({"ok": False, "error": "No reference image"}), 400

    img_data = request.files["image"].read()
    current = decode_image(img_data)
    reference = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)

    score = ssim(reference, current)
    changed = (1 - score) > threshold

    last_path = os.path.join(device_dir(device_id), "last.jpg")
    with open(last_path, "wb") as f:
        f.write(img_data)

    if changed and cfg["telegram_chat"]:
        send_telegram(
            cfg["telegram_chat"],
            last_path,
            f"⚠️ Cambio detectado\nDevice: {device_id}\nΔ={round(1-score,3)}"
        )

    return jsonify({
        "ok": True,
        "device": device_id,
        "similarity": round(score, 3),
        "delta": round(1 - score, 3),
        "threshold": threshold,
        "alert": changed
    })

# ---------- UPDATE CONFIG ----------
@app.route("/config", methods=["POST"])
def update_config():
    data = request.json
    device_id = data.get("device_id")

    if not device_id:
        return jsonify({"ok": False}), 400

    cfg = load_config(device_id)
    cfg.update(data)
    save_config(device_id, cfg)

    return jsonify({"ok": True})

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
