from flask import Flask, request, jsonify, render_template
import requests
import os
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

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    img = request.files["image"].read()

    for attempt in range(3):
        try:
            r = requests.post(
                HF_URL,
                headers=HEADERS,
                data=img,
                timeout=120
            )
        except Exception as e:
            return jsonify({
                "ok": False,
                "error": f"Request failed: {str(e)}"
            }), 500

        # HuggingFace a veces responde texto plano
        if "application/json" not in r.headers.get("Content-Type", ""):
            time.sleep(2)
            continue

        data = r.json()

        # Modelo dormido / error HF
        if isinstance(data, dict) and data.get("error"):
            time.sleep(2)
            continue

        # Respuesta válida
        if isinstance(data, list) and len(data) > 0:
            top = data[0]
            return jsonify({
                "ok": True,
                "label": top.get("label"),
                "confidence": round(top.get("score", 0) * 100, 2),
                "model": HF_MODEL
            })

    return jsonify({
        "ok": False,
        "error": "Model not ready or failed after retries"
    }), 500

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
