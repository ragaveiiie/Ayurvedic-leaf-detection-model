import io, json, os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers

BASE = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model(os.path.join(BASE, "ayurvedic_mobilenetv2.keras"), compile=False)
names = json.load(open(os.path.join(BASE, "class_names.json"), encoding="utf-8"))
INFO = json.load(open(os.path.join(BASE, "plant_info.json"), encoding="utf-8"))
cfg = json.load(open(os.path.join(BASE, "config.json")))
centroids = np.load(os.path.join(BASE, "centroids.npy"))
SIM_T, CONF_T = cfg["sim_threshold"], cfg["conf_threshold"]

gap = [l for l in model.layers if isinstance(l, layers.GlobalAveragePooling2D)][0]
feat_model = keras.Model(model.input, [gap.output, model.output])

UNKNOWN_MSG = "Unknown image. This does not look like one of the known medicinal leaves. Please upload a clear photo of a single leaf."

def norm(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)

app = Flask(__name__)

@app.route("/")
def home():
    resp = send_from_directory(BASE, "index.html")
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify(error="No image uploaded"), 400
    try:
        img = Image.open(io.BytesIO(request.files["image"].read()))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return jsonify(error="This file is not a valid image"), 400

    a = np.array(img.resize((224, 224)), dtype="float32")
    f, p = feat_model.predict(np.stack([a, a[:, ::-1]]), verbose=0)
    p = p.mean(axis=0)
    sim = float((norm(f) @ centroids.T).max())
    top = int(np.argmax(p))
    conf = float(p[top])

    if sim < SIM_T or conf < CONF_T:
        return jsonify(error=UNKNOWN_MSG, confidence=conf)

    best3 = [{"plant": names[i], "confidence": float(p[i])} for i in np.argsort(p)[::-1][:3]]
    return jsonify(plant=names[top], confidence=conf, top3=best3,
                   info=INFO.get(names[top]), note="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
