import base64
import io
import os
import uuid
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from skimage.feature import hog


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"

CLASS_LABELS = {
    "26": "๒๖",
    "27": "๒๗",
    "28": "๒๘",
    "29": "๒๙",
    "30": "๓๐",
}

app = Flask(__name__)
loaded_model = None
loaded_model_name = None


def ensure_directories():
    MODEL_DIR.mkdir(exist_ok=True)
    for class_id in CLASS_LABELS:
        (DATASET_DIR / class_id).mkdir(parents=True, exist_ok=True)


def decode_canvas_image(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    white_bg = Image.new("RGBA", image.size, "WHITE")
    white_bg.alpha_composite(image)
    return white_bg.convert("L")


def preprocess_image(image):
    image = ImageOps.invert(image)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    image.thumbnail((48, 48), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (64, 64), 0)
    x = (64 - image.width) // 2
    y = (64 - image.height) // 2
    canvas.paste(image, (x, y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr


def extract_features(image):
    arr = preprocess_image(image)
    return hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def get_model():
    global loaded_model, loaded_model_name
    if not MODEL_PATH.exists():
        return None

    current_name = MODEL_PATH.name
    current_mtime = MODEL_PATH.stat().st_mtime
    model_key = f"{current_name}:{current_mtime}"
    if loaded_model is None or loaded_model_name != model_key:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_model_name = model_key
    return loaded_model


@app.route("/")
def index():
    return render_template("index.html", labels=CLASS_LABELS)


@app.route("/collect")
def collect():
    return render_template("collect.html", labels=CLASS_LABELS)


@app.route("/admin")
def admin():
    has_model = MODEL_PATH.exists()
    return render_template("admin.html", has_model=has_model, model_name=MODEL_PATH.name)


@app.route("/save-sample", methods=["POST"])
def save_sample():
    data = request.get_json(force=True)
    label = data.get("label")
    image_data = data.get("image")

    if label not in CLASS_LABELS:
        return jsonify({"ok": False, "error": "Invalid label"}), 400
    if not image_data:
        return jsonify({"ok": False, "error": "Missing image"}), 400

    image = decode_canvas_image(image_data)
    output_dir = DATASET_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{label}_{uuid.uuid4().hex[:10]}.png"
    image.save(output_dir / filename)

    total = len(list(output_dir.glob("*.png")))
    return jsonify({"ok": True, "filename": filename, "label": CLASS_LABELS[label], "total": total})


@app.route("/predict", methods=["POST"])
def predict():
    model = get_model()
    if model is None:
        return jsonify({"ok": False, "error": "No model found. Train or upload model.joblib first."}), 400

    data = request.get_json(force=True)
    image_data = data.get("image")
    if not image_data:
        return jsonify({"ok": False, "error": "Missing image"}), 400

    image = decode_canvas_image(image_data)
    features = extract_features(image).reshape(1, -1)
    predicted_id = str(model.predict(features)[0])
    thai_label = CLASS_LABELS.get(predicted_id, predicted_id)

    confidence = None
    probabilities = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        class_ids = [str(c) for c in model.classes_]
        probabilities = [
            {"label": CLASS_LABELS.get(class_id, class_id), "confidence": float(prob)}
            for class_id, prob in zip(class_ids, probs)
        ]
        confidence = float(np.max(probs))

    return jsonify({
        "ok": True,
        "prediction": thai_label,
        "class_id": predicted_id,
        "confidence": confidence,
        "probabilities": probabilities,
    })


@app.route("/upload-model", methods=["POST"])
def upload_model():
    global loaded_model, loaded_model_name
    file = request.files.get("model")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "error": "No model file selected"}), 400
    if not file.filename.endswith(".joblib"):
        return jsonify({"ok": False, "error": "Please upload a .joblib model"}), 400

    MODEL_DIR.mkdir(exist_ok=True)
    file.save(MODEL_PATH)
    loaded_model = None
    loaded_model_name = None
    get_model()
    return jsonify({"ok": True, "model": MODEL_PATH.name})


if __name__ == "__main__":
    ensure_directories()
    app.run(debug=True)
