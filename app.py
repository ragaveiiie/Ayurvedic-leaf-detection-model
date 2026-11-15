import io
import json
import os
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from PIL import Image, ImageOps

from tensorflow import keras
from tensorflow.keras import layers

from huggingface_hub import hf_hub_download


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_REPO = "Probo56/ayurvedic-plant-model"
MODEL_FILE = "ayurvedic_mobilenetv2.keras"


# ==========================================
# 2. DOWNLOAD MODEL FROM HUGGING FACE
# ==========================================

print("Downloading Ayurvedic Plant Detection Model...")

MODEL_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

print("Model downloaded successfully!")


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

print("Loading MobileNetV2 model...")

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ==========================================
# 4. LOAD PROJECT DATA
# ==========================================

names_path = os.path.join(BASE, "class_names.json")
info_path = os.path.join(BASE, "plant_info.json")
config_path = os.path.join(BASE, "config.json")
centroids_path = os.path.join(BASE, "centroids.npy")


with open(names_path, "r", encoding="utf-8") as f:
    names = json.load(f)

with open(info_path, "r", encoding="utf-8") as f:
    INFO = json.load(f)

with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

centroids = np.load(centroids_path)

SIM_T = cfg["sim_threshold"]
CONF_T = cfg["conf_threshold"]


# ==========================================
# 5. FEATURE EXTRACTION MODEL
# ==========================================

gap = [
    layer for layer in model.layers
    if isinstance(layer, layers.GlobalAveragePooling2D)
][0]

feat_model = keras.Model(
    model.input,
    [gap.output, model.output]
)


# ==========================================
# 6. UNKNOWN IMAGE MESSAGE
# ==========================================

UNKNOWN_MSG = (
    "Unknown image. This does not look like one "
    "of the known medicinal leaves. Please upload "
    "a clear photo of a single leaf."
)


# ==========================================
# 7. NORMALIZATION FUNCTION
# ==========================================

def norm(a):
    return a / (
        np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    )


# ==========================================
# 8. INITIALIZE FLASK APPLICATION
# ==========================================

app = Flask(__name__)


# ==========================================
# 9. HOME PAGE
# ==========================================

@app.route("/")
def home():

    resp = send_from_directory(BASE, "index.html")

    resp.headers["Cache-Control"] = "no-store"

    return resp


# ==========================================
# 10. PLANT IMAGE PREDICTION
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify(
            error="No image uploaded"
        ), 400

    try:

        img = Image.open(
            io.BytesIO(request.files["image"].read())
        )

        img = ImageOps.exif_transpose(img).convert("RGB")

    except Exception:

        return jsonify(
            error="This file is not a valid image"
        ), 400


    # Image preprocessing

    a = np.array(
        img.resize((224, 224)),
        dtype="float32"
    )


    # Original and horizontally flipped images

    f, p = feat_model.predict(
        np.stack([a, a[:, ::-1]]),
        verbose=0
    )


    # Average predictions

    p = p.mean(axis=0)


    # Centroid similarity

    sim = float(
        (norm(f) @ centroids.T).max()
    )


    # Highest prediction

    top = int(np.argmax(p))

    conf = float(p[top])


    # Unknown image detection

    if sim < SIM_T or conf < CONF_T:

        return jsonify(
            error=UNKNOWN_MSG,
            confidence=conf
        )


    # Top three predictions

    best3 = [
        {
            "plant": names[i],
            "confidence": float(p[i])
        }
        for i in np.argsort(p)[::-1][:3]
    ]


    # Final result

    return jsonify(
        plant=names[top],
        confidence=conf,
        top3=best3,
        info=INFO.get(names[top]),
        note=""
    )


# ==========================================
# 11. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )
