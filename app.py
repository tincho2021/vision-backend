from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# ================= CONFIG =================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN no definido")

HF_MODEL = "google/vit-base-patch16-224"

HF_URL = (
    f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"
    "?wait_for_model=true"
)

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/octet-stream",
    "Accept": "application/json"
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

    if "application/json" not in r.headers.get("Content-Type", ""):
        return jsonify({
            "ok": False,
            "error": "HF returned non-JSON response",
            "raw": r.text[:300]
        }), 500

    data = r.json()

    if isinstance(data, dict) and data.get("error"):
        return jsonify({
            "ok": False,
            "error": data["error"]
        }), 500

    if not isinstance(data, list) or not data:
        return jsonify({
            "ok": False,
            "error": "Empty model response"
        }), 500

    top = data[0]

    return jsonify({
        "ok": True,
        "label": top.get("label"),
        "confidence": round(top.get("score", 0) * 100, 2),
        "model": HF_MODEL
    })

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
