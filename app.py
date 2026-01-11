from flask import Flask, request, jsonify
import requests
import base64
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_URL = "https://api-inference.huggingface.co/models/llava-hf/llava-1.5-7b-hf"

PROMPT = (
    "Sos un asistente visual para una persona con discapacidad visual. "
    "Analizá la imagen y respondé SOLO lo importante para seguridad, "
    "estados anormales o advertencias prácticas. "
    "Si no hay nada relevante, decí: Todo parece normal. "
    "Respondé en español con una sola frase clara."
)

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Vision backend OK"

@app.route("/analyze", methods=["POST"])
def analyze():
    if not HF_TOKEN:
        return jsonify({"error": "HF_TOKEN no configurado"}), 500

    img_bytes = request.data
    if not img_bytes:
        return jsonify({"error": "No image received"}), 400

    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    payload = {
        "inputs": {
            "image": img_b64,
            "prompt": PROMPT
        }
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

    r = requests.post(HF_URL, headers=headers, json=payload, timeout=60)

    if r.status_code != 200:
        return jsonify({
            "error": "Hugging Face error",
            "status": r.status_code,
            "detail": r.text
        }), 500

    data = r.json()

    # Algunos modelos devuelven listas, otros dict
    if isinstance(data, list) and len(data) > 0:
        text = data[0].get("generated_text", "")
    else:
        text = data.get("generated_text", "")

    if not text:
        text = "No se pudo interpretar la imagen"

    return jsonify({"result": text.strip()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
