from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tempfile
import os

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

REFERENCE_IMAGE = None


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return jsonify(ok=True, service="Behavior-Action Vision")


def read_image(file):
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    return img


@app.route("/reference", methods=["POST"])
def set_reference():
    global REFERENCE_IMAGE

    if "image" not in request.files:
        return jsonify(ok=False, error="No image provided"), 400

    img = read_image(request.files["image"])
    if img is None:
        return jsonify(ok=False, error="Invalid image"), 400

    REFERENCE_IMAGE = img
    return jsonify(ok=True, message="Reference image set")


@app.route("/behavior", methods=["POST"])
def analyze_behavior():
    global REFERENCE_IMAGE

    if REFERENCE_IMAGE is None:
        return jsonify(ok=False, error="Reference not set"), 400

    if "image" not in request.files:
        return jsonify(ok=False, error="No image provided"), 400

    img = read_image(request.files["image"])
    if img is None:
        return jsonify(ok=False, error="Invalid image"), 400

    # Resize to match
    img = cv2.resize(img, (REFERENCE_IMAGE.shape[1], REFERENCE_IMAGE.shape[0]))

    score, diff = ssim(REFERENCE_IMAGE, img, full=True)
    diff = (diff * 255).astype("uint8")

    change_level = float(1 - score)

    behavior = "stable"
    if change_level > 0.15:
        behavior = "changed"
    if change_level > 0.35:
        behavior = "critical change"

    return jsonify(
        ok=True,
        similarity=score,
        change=change_level,
        behavior=behavior
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
