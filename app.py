from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# =========================
# CONFIG HUGGING FACE
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_MODEL = "google/vit-base-patch16-224"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # Validación básica
    if "image" not in request.files:
        return jsonify({
            "ok": False,
            "error": "No image uploaded"
        }), 400

    img = request.files["image"].read()

    # Llamada a HuggingFace
    try:
        r = requests.post(
            HF_URL,
            headers=HEADERS,
            data=img,
            timeout=30
        )
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Request error: {str(e)}"
        }), 500

    # HuggingFace a veces devuelve HTML cuando el modelo está dormido
    content_type = r.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        return jsonify({
            "ok": False,
            "error": "HuggingFace returned non-JSON response (model sleeping or error)",
            "raw": r.text[:200]
        }), 502

    # Parse JSON
    try:
        data = r.json()
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": "JSON parse error",
            "raw": r.text[:200]
        }), 500

    return jsonify({
        "ok": True,
        "model": HF_MODEL,
        "result": data
    })


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
