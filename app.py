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

# ================= STATE =================
REFERENCE = None   # se guarda en RAM

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    img = request.files.get("image")
    if not img:
        return jsonify(ok=False, error="No image"), 400

    return run_model(img.read())

@app.route("/reference", methods=["POST"])
def set_reference():
    global REFERENCE

    img = request.files.get("image")
    if not img:
        return jsonify(ok=False, error="No image"), 400

    result = run_model(img.read())
    if not result.get("ok"):
        return result, 500

    REFERENCE = result
    return jsonify(ok=True, reference=result)

@app.route("/behavior", methods=["POST"])
def behavior():
    if REFERENCE is None:
        return jsonify(
            ok=False,
            error="No reference set yet"
        ), 400

    img = request.files.get("image")
    if not img:
        return jsonify(ok=False, error="No image"), 400

    current = run_model(img.read())
    if not current.get("ok"):
        return current, 500

    changed = abs(
        current["confidence"] - REFERENCE["confidence"]
    ) > 10

    return jsonify(
        ok=True,
        behavior="changed" if changed else "stable",
        reference=REFERENCE,
        current=current
    )

# ================= CORE =================
def run_model(image_bytes):
    try:
        r = requests.post(
            HF_URL,
            headers=HEADERS,
            data=image_bytes,
            timeout=90
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}

    if "application/json" not in r.headers.get("Content-Type", ""):
        return {
            "ok": False,
            "error": "HF returned non-JSON",
            "raw": r.text[:200]
        }

    data = r.json()
    if not isinstance(data, list) or not data:
        return {"ok": False, "error": "Empty model response"}

    top = data[0]
    return {
        "ok": True,
        "label": top.get("label"),
        "confidence": round(top.get("score", 0) * 100, 2),
        "model": HF_MODEL
    }

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
