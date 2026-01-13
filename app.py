import os
import uuid
import json
import time
import urllib.request
import urllib.parse
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageChops, ImageStat

# =========================
# CONFIG / STORAGE
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REF_DIR = os.path.join(BASE_DIR, "references")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")  # opcional

# Flask: sirve /static/* y también / (index.html)
app = Flask(__name__, static_folder="static", static_url_path="")

# =========================
# UTILS
# =========================

def _safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)

def _device_id_from_request():
    # web => "web"
    return (request.form.get("device_id") or request.args.get("device_id") or "web").strip()

def _save_upload(file_storage, folder, filename):
    path = os.path.join(folder, filename)
    file_storage.save(path)
    return path

def _open_norm(path, size=(320, 320)):
    # normalizamos tamaño para comparación (puede cambiarlo si querés)
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    return img

def compute_similarity(ref_img, cur_img):
    """
    Similaridad simple (0..1) usando diferencia absoluta promedio.
    - 1.0 = idénticas
    - 0.0 = muy diferentes
    """
    diff = ImageChops.difference(ref_img, cur_img)
    stat = ImageStat.Stat(diff)
    # mean por canal RGB (0..255)
    mean = stat.mean  # [r,g,b]
    # promedio de los 3 canales
    mean_all = (mean[0] + mean[1] + mean[2]) / 3.0
    # normalizamos a 0..1
    change_score = mean_all / 255.0
    similarity = 1.0 - change_score
    # clamp
    if similarity < 0: similarity = 0.0
    if similarity > 1: similarity = 1.0
    if change_score < 0: change_score = 0.0
    if change_score > 1: change_score = 1.0
    return similarity, change_score

def telegram_send_message(chat_id, text):
    if not TELEGRAM_TOKEN or not chat_id:
        return False, "telegram disabled"
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            _ = resp.read()
        return True, "ok"
    except Exception as e:
        return False, str(e)

def telegram_send_photo(chat_id, caption, photo_path):
    """
    Enviar foto por Telegram sin 'requests' (multipart manual).
    """
    if not TELEGRAM_TOKEN or not chat_id:
        return False, "telegram disabled"

    try:
        boundary = "----WebKitFormBoundary" + uuid.uuid4().hex
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"

        def part(name, value):
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")

        filename = os.path.basename(photo_path)
        with open(photo_path, "rb") as f:
            file_bytes = f.read()

        file_header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="photo"; filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8")

        closing = f"\r\n--{boundary}--\r\n".encode("utf-8")

        body = b"".join([
            part("chat_id", chat_id),
            part("caption", caption or ""),
            file_header,
            file_bytes,
            closing
        ])

        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("Content-Length", str(len(body)))

        with urllib.request.urlopen(req, timeout=30) as resp:
            _ = resp.read()

        return True, "ok"
    except Exception as e:
        return False, str(e)

def json_error(status, message, extra=None):
    payload = {"ok": False, "error": message}
    if extra:
        payload.update(extra)
    return jsonify(payload), status

# =========================
# ROUTES
# =========================

@app.route("/health", methods=["GET"])
def health():
    return jsonify(ok=True, service="Behavior-Action Vision", telegram_enabled=bool(TELEGRAM_TOKEN))

@app.route("/", methods=["GET"])
def index():
    # GUI (static/index.html)
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/reference", methods=["POST"])
def set_reference():
    if "image" not in request.files:
        return json_error(400, "No image provided")

    device_id = _device_id_from_request()
    img = request.files["image"]

    ref_path = os.path.join(REF_DIR, f"{device_id}.jpg")
    img.save(ref_path)

    return jsonify(ok=True, message="Referencia seteada correctamente", device_id=device_id)

# Alias por compatibilidad con nombres viejos
@app.route("/behavior", methods=["POST"])
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return json_error(400, "No image provided")

    device_id = _device_id_from_request()
    chat_id = (request.form.get("chat_id") or request.args.get("chat_id") or "").strip()
    threshold = _safe_float(request.form.get("threshold") or request.args.get("threshold"), 0.25)

    ref_path = os.path.join(REF_DIR, f"{device_id}.jpg")
    if not os.path.exists(ref_path):
        return json_error(400, "No reference for this device_id", {"device_id": device_id})

    # Guardamos la imagen recibida
    current_filename = f"{device_id}_{int(time.time())}_{uuid.uuid4().hex}.jpg"
    current_path = os.path.join(UPLOAD_DIR, current_filename)
    request.files["image"].save(current_path)

    try:
        ref_img = _open_norm(ref_path, size=(320, 320))
        cur_img = _open_norm(current_path, size=(320, 320))
        similarity, change_score = compute_similarity(ref_img, cur_img)

        critical = bool(change_score > float(threshold))

        # Telegram: si hay cambio crítico, enviamos
        telegram_status = {"sent": False, "reason": "not triggered"}
        if critical and chat_id:
            caption = f"🚨 Cambio detectado\nDevice: {device_id}\nchange_score={change_score:.3f}\nthreshold={threshold:.3f}"
            ok_photo, msg_photo = telegram_send_photo(chat_id, caption, current_path)
            telegram_status = {"sent": bool(ok_photo), "reason": msg_photo}

        return jsonify(
            ok=True,
            device_id=device_id,
            similarity=float(round(similarity, 4)),
            change_score=float(round(change_score, 4)),
            threshold=float(round(float(threshold), 4)),
            critical_change=bool(critical),
            telegram=telegram_status
        )

    except Exception as e:
        return json_error(500, f"Analyze failed: {str(e)}")

# =========================
# STATIC
# =========================
@app.route("/<path:path>", methods=["GET"])
def static_files(path):
    # deja que /static/* funcione
    return send_from_directory(STATIC_DIR, path)

# =========================
# JSON 404 FOR API
# =========================
@app.errorhandler(404)
def not_found(e):
    # si llaman a endpoints de API, devolvemos JSON (evita el "<!doctype ...>" en el frontend)
    if request.path.startswith("/analyze") or request.path.startswith("/behavior") or request.path.startswith("/reference") or request.path.startswith("/health"):
        return json_error(404, "Not Found")
    return e, 404

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
