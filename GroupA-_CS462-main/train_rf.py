import datetime
from pathlib import Path
import joblib
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# --- ตั้งค่า Path และ Labels ---
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

# --- ฟังก์ชันจัดการรูปภาพ (Preprocessing) ---
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
        print(f"โหลดคลาส {CLASS_LABELS[class_id]} ({class_id}): พบ {len(image_paths)} รูป")

        for path in image_paths:
            features.append(extract_features(path))
            labels.append(class_id)

    if not features:
        raise RuntimeError("ไม่พบรูปภาพใน dataset กรุณาไปที่หน้า /collect เพื่อเก็บข้อมูลก่อน")

    return np.asarray(features), np.asarray(labels)

# --- ฟังก์ชันหลักสำหรับเทรน ---
def main():
    print("--- กำลังเริ่มการเทรนด้วย Random Forest ---")
    
    # โหลดข้อมูล
    x, y = load_dataset()
    
    # แบ่งข้อมูล 80% Train, 20% Test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # สร้าง Pipeline (StandardScaler + RandomForest)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, 
            criterion='entropy', 
            random_state=42,
            n_jobs=-1 
        )),
    ])

    # เริ่มเทรน
    print("กำลังประมวลผลข้อมูลและสร้างโมเดล...")
    model.fit(x_train, y_train)

    # วัดความแม่นยำ
    accuracy = model.score(x_test, y_test)
    print(f"\nผลลัพธ์ความแม่นยำ (Accuracy): {accuracy * 100:.2f}%\n")

    # แสดง Report รายละเอียด
    y_pred = model.predict(x_test)
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=[CLASS_LABELS[class_id] for class_id in CLASS_LABELS]
    ))

    # --- บันทึกไฟล์โมเดลใหม่แยกตามเวลา ---
    MODEL_DIR.mkdir(exist_ok=True)
    
    # สร้างชื่อไฟล์ใหม่ เช่น model_rf_20260512_1530.joblib
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_model_filename = f"model_rf_{timestamp}.joblib"
    save_path = MODEL_DIR / new_model_filename
    
    joblib.dump(model, save_path)
    
    print("-" * 30)
    print(f"บันทึกไฟล์สำเร็จ!")
    print(f"ตำแหน่งไฟล์: {save_path}")
    print("คุณสามารถนำไฟล์ชื่อนี้ไปอัปโหลดที่หน้า Admin ของ Web App ได้ทันที")
    print("-" * 30)

if __name__ == "__main__":
    main()