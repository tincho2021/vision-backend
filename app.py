from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# --- estado simple en memoria ---
REFERENCE_IMAGE = None

@app.route("/")
def home():
    return jsonify({"ok": True, "service": "Behavior-Action Vision"})

# -------------------------------
# SETEAR IMAGEN DE REFERENCIA
# -------------------------------
@app.route("/reference", methods=["POST"])
def set_reference():
    global REFERENCE_IMAGE

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files["image"].read()
    REFERENCE_IMAGE = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    return jsonify({
        "ok": True,
        "message": "Referencia seteada correctamente"
    })

# -------------------------------
# ANALIZAR COMPORTAMIENTO
# -------------------------------
@app.route("/behavior", methods=["POST"])
def analyze_behavior():
    if REFERENCE_IMAGE is None:
        return jsonify({
            "ok": False,
            "error": "No hay imagen de referencia"
        }), 400

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # ⚠️ Por ahora lógica dummy (próximo paso: visión real)
    return jsonify({
        "ok": True,
        "behavior": "stable",
        "confidence": 0.78,
        "message": "Sin cambios relevantes detectados"
    })

# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
