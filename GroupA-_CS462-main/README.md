# Thai Handwriting Classifier 26-30

Web application for collecting Thai handwritten number samples and predicting 5 classes:

- ๒๖
- ๒๗
- ๒๘
- ๒๙
- ๓๐

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Web App

```powershell
python app.py
```

Open:

- User page: http://127.0.0.1:5000/
- Dataset collector: http://127.0.0.1:5000/collect
- Admin page: http://127.0.0.1:5000/admin

## Workflow

### Option A: Generate Starter Dataset

This creates a starter dataset from Thai fonts with rotation, position, blur, and noise augmentation. It is useful for testing the full system quickly.

```powershell
python generate_starter_dataset.py
python train_model.py
```

### Option B: Collect Real Handwriting

1. Open `/collect`.
2. Select a label from ๒๖ to ๓๐.
3. Draw the selected number on the canvas.
4. Click Save Sample.
5. Repeat until each class has at least 50 images.
6. Train the model:

```powershell
python train_model.py
```

7. Use `/` to test prediction.
8. Use `/admin` to upload a new `model.joblib`.

## Dataset Folder

The collector saves images into:

```text
dataset/
  26/
  27/
  28/
  29/
  30/
```

The web page displays Thai labels, while folder names use Arabic digits to avoid encoding problems.
