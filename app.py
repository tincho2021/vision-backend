from flask import Flask, request, jsonify, render_template
import requests
import os
import hashlib
import time

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

# ================= MEMORY (simple, en RAM) =================
REFERENCE_IMAGE = {
    "hash": None,
    "result": None,
    "timestamp": None
}

# ================= UTILS =================
def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def call_hf(image_bytes: bytes):
    if not image_bytes or len(image_bytes) < 1000:
        return {"ok": False, "error": "Invalid or empty image"}

    try:
        r = requests.post(
            HF_URL,
            headers=HEADERS,
            data=image_bytes,
            timeout=120
        )
    except Exception as e:
        return {"ok": False, "error": f"HF request failed: {str(e)}"}

    ct = r.headers.get("Content-Type", "")
    if "application/json" not in ct:
        return {
            "ok": False,
            "error": "HF returned non-JSON response",
            "raw": r.text[:300]
        }

    data = r.json()

    if isinstance(data, dict) and data.get("error"):
        return {"ok": False, "error": data["error"]}

    if not isinstance(data, list) or not data:
        return {"ok": False, "error": "Empty model response"}

    top = data[0]

    return {
        "ok": True,
        "label": top.get("label"),
        "confidence": round(top.get("score", 0) * 100, 2),
        "raw": data
    }

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

# ---------- CLASIFICACIÓN SIMPLE ----------
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    image_bytes = request.files["image"].read()
    result = call_hf(image_bytes)

    if not result["ok"]:
        return jsonify(result), 500

    return jsonify({
        "ok": True,
        "label": result["label"],
        "confidence": result["confidence"],
        "model": HF_MODEL
    })

# ---------- SETEAR REFERENCIA ----------
@app.route("/reference", methods=["POST"])
def set_reference():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    image_bytes = request.files["image"].read()
    result = call_hf(image_bytes)

    if not result["ok"]:
        return jsonify(result), 500

    REFERENCE_IMAGE["hash"] = image_hash(image_bytes)
    REFERENCE_IMAGE["result"] = result
    REFERENCE_IMAGE["timestamp"] = time.time()

    return jsonify({
        "ok": True,
        "reference": {
            "label": result["label"],
            "confidence": result["confidence"]
        }
    })

# ---------- ANALIZAR COMPORTAMIENTO ----------
@app.route("/behavior", methods=["POST"])
def behavior():
    if not REFERENCE_IMAGE["result"]:
        return jsonify({"ok": False, "error": "No reference image set"}), 400

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    image_bytes = request.files["image"].read()
    current = call_hf(image_bytes)

    if not current["ok"]:
        return jsonify(current), 500

    ref = REFERENCE_IMAGE["result"]

    same_label = ref["label"] == current["label"]
    confidence_delta = abs(ref["confidence"] - current["confidence"])

    if same_label and confidence_delta < 10:
        behavior_state = "stable"
    else:
        behavior_state = "changed"

    return jsonify({
        "ok": True,
        "behavior": behavior_state,
        "label": current["label"],
        "confidence": current["confidence"],
        "reference": {
            "label": ref["label"],
            "confidence": ref["confidence"]
        },
        "model": HF_MODEL
    })

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
