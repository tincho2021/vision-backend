import os
import io
import numpy as np
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# -------------------------
# CONFIG
# -------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_EMBED = "sentence-transformers/clip-ViT-B-32"
HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_EMBED}"

SIMILARITY_THRESHOLD = 0.92  # menor = más sensible al cambio

# -------------------------
# APP INIT
# -------------------------

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

reference_embedding = None

# -------------------------
# UTILS
# -------------------------

def image_to_bytes(file):
    image = Image.open(file).convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def get_image_embedding(image_bytes):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/octet-stream",
    }

    response = requests.post(
        HF_API_URL,
        headers=headers,
        data=image_bytes,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"HF embed error ({response.status_code}): {response.text}"
        )

    embedding = np.array(response.json(), dtype=np.float32)
    return embedding / np.linalg.norm(embedding)


def cosine_similarity(a, b):
    return float(np.dot(a, b))


# -------------------------
# ROUTES
# -------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return jsonify({"ok": True, "service": "Behavior-Action Vision"})


@app.route("/reference", methods=["POST"])
def set_reference():
    global reference_embedding

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    try:
        image_bytes = image_to_bytes(request.files["image"])
        reference_embedding = get_image_embedding(image_bytes)

        return jsonify({
            "ok": True,
            "message": "Reference image set successfully"
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


@app.route("/behavior", methods=["POST"])
def analyze_behavior():
    global reference_embedding

    if reference_embedding is None:
        return jsonify({
            "ok": False,
            "error": "Reference not set"
        }), 400

    if "image" not in request.files:
        return jsonify({
            "ok": False,
            "error": "No image provided"
        }), 400

    try:
        image_bytes = image_to_bytes(request.files["image"])
        current_embedding = get_image_embedding(image_bytes)

        similarity = cosine_similarity(reference_embedding, current_embedding)
        changed = similarity < SIMILARITY_THRESHOLD

        return jsonify({
            "ok": True,
            "behavior": "changed" if changed else "stable",
            "similarity": round(similarity, 4),
            "threshold": SIMILARITY_THRESHOLD,
            "confidence_change": round((1 - similarity) * 100, 2)
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500


# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
