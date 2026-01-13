from flask import Flask, request, jsonify, send_from_directory
import os
import base64
import cv2
import numpy as np

app = Flask(__name__, static_folder="static")

REFERENCE_PATH = "reference.jpg"

# ---------- UTILS ----------

def decode_image(base64_str):
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def similarity_score(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_gray = cv2.resize(img1_gray, (300, 300))
    img2_gray = cv2.resize(img2_gray, (300, 300))

    diff = cv2.absdiff(img1_gray, img2_gray)
    score = np.mean(diff) / 255.0
    similarity = 1.0 - score
    return similarity, score

# ---------- ROUTES ----------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/reference", methods=["POST"])
def set_reference():
    try:
        data = request.get_json(force=True)
        img_b64 = data.get("image")

        if not img_b64:
            return jsonify({"ok": False, "error": "No image received"}), 400

        img = decode_image(img_b64)
        cv2.imwrite(REFERENCE_PATH, img)

        return jsonify({
            "ok": True,
            "message": "Referencia seteada correctamente"
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if not os.path.exists(REFERENCE_PATH):
            return jsonify({
                "ok": False,
                "error": "Referencia no seteada"
            }), 400

        data = request.get_json(force=True)
        img_b64 = data.get("image")
        threshold = float(data.get("threshold", 0.25))

        if not img_b64:
            return jsonify({
                "ok": False,
                "error": "No image received"
            }), 400

        current_img = decode_image(img_b64)
        ref_img = cv2.imread(REFERENCE_PATH)

        similarity, change_score = similarity_score(ref_img, current_img)
        critical_change = change_score >= threshold

        return jsonify({
            "ok": True,
            "similarity": round(similarity, 4),
            "change_score": round(change_score, 4),
            "threshold": threshold,
            "critical_change": bool(critical_change)
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e)
        }), 500

# ---------- MAIN ----------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
