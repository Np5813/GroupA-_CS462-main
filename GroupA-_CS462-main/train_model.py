from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


def preprocess_image(path):
    image = Image.open(path).convert("RGBA")
    white_bg = Image.new("RGBA", image.size, "WHITE")
    white_bg.alpha_composite(image)
    image = white_bg.convert("L")
    image = ImageOps.invert(image)

    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    image.thumbnail((48, 48), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (64, 64), 0)
    x = (64 - image.width) // 2
    y = (64 - image.height) // 2
    canvas.paste(image, (x, y))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def extract_features(path):
    arr = preprocess_image(path)
    return hog(
        arr,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def load_dataset():
    features = []
    labels = []

    for class_id in CLASS_LABELS:
        class_dir = DATASET_DIR / class_id
        if not class_dir.exists():
            print(f"Missing folder: {class_dir}")
            continue

        image_paths = sorted(
            list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
        )
        print(f"{CLASS_LABELS[class_id]} ({class_id}): {len(image_paths)} images")

        for path in image_paths:
            features.append(extract_features(path))
            labels.append(class_id)

    if not features:
        raise RuntimeError("No dataset images found. Use /collect to save samples first.")

    return np.asarray(features), np.asarray(labels)


def main():
    x, y = load_dataset()
    class_count = len(set(y))
    if class_count < len(CLASS_LABELS):
        raise RuntimeError("Please collect images for all 5 classes before training.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])
    model.fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

    y_pred = model.predict(x_test)
    print(classification_report(
        y_test,
        y_pred,
        labels=list(CLASS_LABELS.keys()),
        target_names=[CLASS_LABELS[class_id] for class_id in CLASS_LABELS],
        zero_division=0,
    ))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
