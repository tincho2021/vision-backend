from flask import Flask, request, jsonify, render_template
import requests
import os
import time

app = Flask(__name__)

# ================= CONFIG =================

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN no definido en variables de entorno")

HF_MODEL = "google/vit-base-patch16-224"
HF_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Accept": "application/json",
    "Content-Type": "application/octet-stream",
}

MAX_RETRIES = 3
RETRY_DELAY = 4  # segundos

# ================= ROUTES =================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    img_bytes = request.files["image"].read()

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                HF_URL,
                headers=HEADERS,
                data=img_bytes,
                timeout=30,
            )

            content_type = resp.headers.get("Content-Type", "")

            # 🧠 HF devolvió HTML (modelo dormido o error)
            if "application/json" not in content_type:
                last_error = {
                    "error": "HuggingFace returned non-JSON response (model sleeping or error)",
                    "attempt": attempt,
                    "raw": resp.text[:500],
                }
                time.sleep(RETRY_DELAY)
                continue

            data = resp.json()

            # 🧠 HF sigue despertando
            if isinstance(data, dict) and data.get("error"):
                last_error = data
                time.sleep(RETRY_DELAY)
                continue

            # ✅ Respuesta válida
            if isinstance(data, list) and len(data) > 0:
                top = data[0]
                return jsonify({
                    "ok": True,
                    "label": top.get("label"),
                    "score": round(top.get("score", 0) * 100, 2),
                    "model": HF_MODEL,
                })

            last_error = {"error": "Empty response from model"}

        except Exception as e:
            last_error = {"error": str(e)}
            time.sleep(RETRY_DELAY)

    # ❌ Fallaron todos los intentos
    return jsonify({
        "ok": False,
        "error": "Model not ready or failed after retries",
        "details": last_error,
    }), 500


# ================= MAIN =================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
