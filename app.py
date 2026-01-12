from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import time
import requests
import numpy as np
from PIL import Image, ImageChops, ImageFilter
from io import BytesIO

app = Flask(__name__)
CORS(app)

# ================= CONFIG =================
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN no definido (Render -> Environment Variables)")

# Clasificación (labels)
HF_MODEL_CLASSIFY = os.environ.get("HF_MODEL_CLASSIFY", "google/vit-base-patch16-224")

# Embeddings (para comparar "significado" entre dos imágenes)
# CLIP funciona bien para "similitud semántica"
HF_MODEL_EMBED = os.environ.get("HF_MODEL_EMBED", "openai/clip-vit-base-patch32")

HF_ROUTER_BASE = "https://router.huggingface.co/hf-inference/models"

HEADERS_BIN = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/octet-stream",
    "Accept": "application/json",
    "x-wait-for-model": "true",
}

# Referencia (en memoria). En Render free puede reiniciarse si el servicio duerme.
REFERENCE = {
    "image_bytes": None,
    "set_at": None,
    "classify": None,   # {label, score}
    "embed": None,      # np.array
    "thumb": None,      # bytes jpg chico (debug opcional)
}

# ================= HELPERS =================

def _safe_json(resp):
    ctype = resp.headers.get("Content-Type", "")
    if "application/json" not in ctype:
        return None, resp.text[:400]
    try:
        return resp.json(), None
    except Exception:
        return None, resp.text[:400]

def _hf_post(model_name: str, img_bytes: bytes, timeout=120):
    url = f"{HF_ROUTER_BASE}/{model_name}"
    r = requests.post(url, headers=HEADERS_BIN, data=img_bytes, timeout=timeout)
    data, raw = _safe_json(r)
    return r.status_code, data, raw

def _pil_from_bytes(img_bytes: bytes) -> Image.Image:
    im = Image.open(BytesIO(img_bytes))
    # Convertimos a RGB por consistencia
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im

def _make_thumb_jpg(im: Image.Image, max_w=512) -> bytes:
    w, h = im.size
    if w > max_w:
        new_h = int(h * (max_w / w))
        im = im.resize((max_w, new_h))
    bio = BytesIO()
    im.save(bio, format="JPEG", quality=85)
    return bio.getvalue()

def _visual_change_score(a_bytes: bytes, b_bytes: bytes) -> float:
    """
    0.0 = igual, 1.0 = muy diferente
    Mezcla diferencia de píxel + diferencia de bordes, robusto a pequeñas variaciones.
    """
    a = _pil_from_bytes(a_bytes).resize((320, 240))
    b = _pil_from_bytes(b_bytes).resize((320, 240))

    # Diferencia de pixel (promedio absoluto)
    diff = ImageChops.difference(a, b).convert("L")
    arr = np.asarray(diff, dtype=np.float32) / 255.0
    pixel = float(arr.mean())

    # Diferencia de bordes (reduce falsos por cambios leves de luz)
    ea = a.convert("L").filter(ImageFilter.FIND_EDGES)
    eb = b.convert("L").filter(ImageFilter.FIND_EDGES)
    ed = ImageChops.difference(ea, eb)
    earr = np.asarray(ed, dtype=np.float32) / 255.0
    edge = float(earr.mean())

    # Combinar
    return float(np.clip(0.55 * edge + 0.45 * pixel, 0.0, 1.0))

def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    nu = np.linalg.norm(u) + 1e-9
    nv = np.linalg.norm(v) + 1e-9
    sim = float(np.dot(u, v) / (nu * nv))
    # Distancia: 0 igual, 1 diferente
    return float(np.clip(1.0 - sim, 0.0, 2.0))

def _embed_clip(img_bytes: bytes) -> np.ndarray:
    """
    Llama a HF feature-extraction / embeddings (CLIP).
    Devuelve vector 1D.
    """
    code, data, raw = _hf_post(HF_MODEL_EMBED, img_bytes, timeout=180)

    if code != 200 or data is None:
        raise RuntimeError(f"HF embed error (code={code}): {raw or data}")

    # HF puede devolver:
    # - lista [seq_len, hidden]
    # - o lista [hidden]
    arr = np.array(data, dtype=np.float32)
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2:
        # mean pooling
        vec = arr.mean(axis=0)
    elif arr.ndim == 3:
        # a veces batch: [1, seq, hidden]
        vec = arr[0].mean(axis=0)
    else:
        raise RuntimeError(f"Embed shape raro: {arr.shape}")

    return vec

def _classify(img_bytes: bytes):
    code, data, raw = _hf_post(HF_MODEL_CLASSIFY, img_bytes, timeout=180)

    if code != 200 or data is None:
        return None, {"ok": False, "error": f"HF classify error (code={code})", "raw": raw}

    if isinstance(data, dict) and data.get("error"):
        return None, {"ok": False, "error": data.get("error")}

    if not isinstance(data, list) or not data:
        return None, {"ok": False, "error": "Respuesta vacía del modelo"}

    top = data[0]
    return {
        "label": top.get("label"),
        "score": float(top.get("score", 0.0)),
        "model": HF_MODEL_CLASSIFY,
    }, {"ok": True}

def _decision(change_visual: float, change_semantic: float):
    """
    Regla simple pero efectiva.
    Ajustable con thresholds.
    """
    # Ponderación (semántico manda un poco más)
    combined = float(np.clip(0.6 * change_semantic + 0.4 * change_visual, 0.0, 1.0))

    # Umbrales
    if combined >= 0.30:
        behavior = "changed"
        confidence = min(99.0, 50.0 + combined * 70.0)
    elif combined <= 0.15:
        behavior = "stable"
        confidence = min(99.0, 50.0 + (1.0 - combined) * 50.0)
    else:
        behavior = "uncertain"
        confidence = min(99.0, 40.0 + (0.30 - abs(combined - 0.225)) * 200.0)

    return behavior, float(round(confidence, 2)), float(round(combined, 4))

# ================= ROUTES =================

@app.route("/health")
def health():
    return jsonify({"ok": True, "service": "Behavior-Action Vision"})

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    img = request.files["image"].read()
    cls, meta = _classify(img)
    if not meta.get("ok"):
        return jsonify(meta), 500

    return jsonify({
        "ok": True,
        "label": cls["label"],
        "confidence": round(cls["score"] * 100, 2),
        "model": cls["model"]
    })

@app.route("/reference", methods=["POST"])
def set_reference():
    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    img = request.files["image"].read()

    # Guardar thumb (debug)
    try:
        im = _pil_from_bytes(img)
        thumb = _make_thumb_jpg(im)
    except Exception:
        thumb = None

    # Clasificación (opcional)
    cls, meta = _classify(img)
    if not meta.get("ok"):
        cls = None  # no bloqueamos referencia por falla de classify

    # Embedding (clave)
    try:
        emb = _embed_clip(img)
    except Exception as e:
        return jsonify({"ok": False, "error": f"No pude obtener embedding: {str(e)}"}), 500

    REFERENCE["image_bytes"] = img
    REFERENCE["set_at"] = time.time()
    REFERENCE["classify"] = cls
    REFERENCE["embed"] = emb
    REFERENCE["thumb"] = thumb

    return jsonify({
        "ok": True,
        "msg": "Referencia seteada",
        "reference": cls,
        "embed_model": HF_MODEL_EMBED
    })

@app.route("/behavior", methods=["POST"])
def behavior():
    if REFERENCE["image_bytes"] is None or REFERENCE["embed"] is None:
        return jsonify({"ok": False, "error": "No hay referencia seteada"}), 400

    if "image" not in request.files:
        return jsonify({"ok": False, "error": "No image provided"}), 400

    current = request.files["image"].read()

    # 1) Cambio visual
    change_visual = _visual_change_score(REFERENCE["image_bytes"], current)

    # 2) Cambio semántico (CLIP)
    try:
        emb_current = _embed_clip(current)
        change_semantic = _cosine_distance(REFERENCE["embed"], emb_current)
        # CLIP a veces da distancias >1 en casos extremos, lo recortamos
        change_semantic = float(np.clip(change_semantic, 0.0, 1.0))
    except Exception as e:
        return jsonify({"ok": False, "error": f"Embedding falló: {str(e)}"}), 500

    behavior_label, confidence, combined = _decision(change_visual, change_semantic)

    # Clasificación "para humanos" (sirve como pista, no como decisión)
    cls, meta = _classify(current)
    if not meta.get("ok"):
        cls = None

    return jsonify({
        "ok": True,
        "behavior": behavior_label,                 # stable / changed / uncertain
        "confidence": confidence,                   # %
        "change_score": combined,                   # 0..1
        "signals": {
            "visual_change": round(change_visual, 4),
            "semantic_change": round(change_semantic, 4),
        },
        "current": cls,
        "reference": REFERENCE["classify"],
        "models": {
            "classify": HF_MODEL_CLASSIFY,
            "embed": HF_MODEL_EMBED
        }
    })

# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
