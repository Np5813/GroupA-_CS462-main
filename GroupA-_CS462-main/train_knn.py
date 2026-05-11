import datetime
from pathlib import Path
import joblib
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --- ตั้งค่า Path และ Labels (Absolute Path) ---
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"

CLASS_LABELS = {
    "26": "๒๖",
    "27": "๒๗",
    "28": "๒๘",
    "29": "๒๙",
    "30": "๓๐",
}

# --- ฟังก์ชันจัดการรูปภาพ (Preprocessing เหมือนเดิมเป๊ะ) ---
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

# --- ฟังก์ชันโหลด Dataset ---
def load_dataset():
    features = []
    labels = []

    print(f"--- ตรวจสอบข้อมูลใน: {DATASET_DIR} ---")
    for class_id in CLASS_LABELS:
        class_dir = DATASET_DIR / class_id
        if not class_dir.exists():
            print(f"⚠️ ไม่พบโฟลเดอร์: {class_id}")
            continue

        image_paths = sorted(
            list(class_dir.glob("*.png"))
            + list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
        )
        print(f"โหลดคลาส {CLASS_LABELS[class_id]} ({class_id}): {len(image_paths)} รูป")

        for path in image_paths:
            features.append(extract_features(path))
            labels.append(class_id)

    if not features:
        raise RuntimeError("ไม่พบรูปภาพใน dataset กรุณาเก็บข้อมูลเพิ่มก่อนเทรน")

    return np.asarray(features), np.asarray(labels)

# --- ฟังก์ชันหลักสำหรับเทรน KNN ---
def main():
    print("--- เริ่มการเทรนด้วย K-Nearest Neighbors (KNN) ---")
    
    # โหลดข้อมูล
    x, y = load_dataset()
    
    # แบ่งข้อมูล 80/20
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # สร้าง Pipeline สำหรับ KNN
    # n_neighbors=5: ดูเพื่อนบ้านที่ใกล้ที่สุด 5 จุด
    # weights='distance': ให้คะแนนภาพที่ใกล้มากๆ มีน้ำหนักมากกว่าภาพที่อยู่ไกลออกไป
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, weights='distance')),
    ])

    print("กำลังคำนวณระยะห่างและสร้างโมเดล...")
    model.fit(x_train, y_train)

    # วัดผล
    accuracy = model.score(x_test, y_test)
    print(f"\n✅ KNN Accuracy: {accuracy * 100:.2f}%\n")

    y_pred = model.predict(x_test)
    print("Classification Report:")
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=[CLASS_LABELS[class_id] for class_id in CLASS_LABELS]
    ))

    # --- บันทึกไฟล์แยกตามเวลา ---
    MODEL_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_filename = f"model_knn_{timestamp}.joblib"
    save_path = MODEL_DIR / new_model_filename
    
    joblib.dump(model, save_path)
    
    print("-" * 50)
    print(f"เทรนเสร็จสมบูรณ์!")
    print(f"ไฟล์ของคุณคือ: {new_model_filename}")
    print(f"อยู่ที่: {save_path}")
    print("นำไฟล์นี้ไปอัปโหลดหน้า Admin ได้เลยครับ")
    print("-" * 50)

if __name__ == "__main__":
    main()