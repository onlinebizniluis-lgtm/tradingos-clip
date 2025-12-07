import os
import glob
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
import onnxruntime as ort
from typing import Dict

app = FastAPI()

# -----------------------------
# CONFIG
# -----------------------------
REFERENCE_DIR = "reference/market_structure"
MODEL_PATH = "model/mobileclip_s0.onnx"

# -----------------------------
# LOAD MODEL
# -----------------------------
print("Loading MobileCLIP-S0 ONNX model...")
providers = ["CPUExecutionProvider"]
session = ort.InferenceSession(MODEL_PATH, providers=providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# -----------------------------
# IMAGE TO EMBEDDING
# -----------------------------
def load_image(path: str):
    img = Image.open(path).convert("RGB").resize((224, 224))
    return np.array(img).astype(np.float32) / 255.0

def preprocess(img: Image.Image):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = np.expand_dims(arr, 0)
    return arr

def embed(arr: np.ndarray):
    out = session.run([output_name], {input_name: arr})
    vec = out[0][0]
    # Normalize
    return vec / np.linalg.norm(vec)

# -----------------------------
# LOAD REFERENCE DATABASE
# -----------------------------
print("Indexing reference images...")
reference_db = []

for class_name in ["uptrend", "downtrend", "consolidation"]:
    folder = os.path.join(REFERENCE_DIR, class_name)
    files = glob.glob(os.path.join(folder, "*.png"))

    for fpath in files:
        img = Image.open(fpath)
        arr = preprocess(img)
        emb = embed(arr)
        reference_db.append({
            "class": class_name,
            "file": os.path.basename(fpath),
            "embedding": emb
        })

print(f"Loaded {len(reference_db)} reference examples.")

# -----------------------------
# COSINE SIMILARITY
# -----------------------------
def cosine(a, b):
    return float(np.dot(a, b))

# -----------------------------
# API ROUTE
# -----------------------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> Dict:
    img = Image.open(file.file)
    arr = preprocess(img)
    emb = embed(arr)

    # Score against DB
    scores = []
    for ref in reference_db:
        score = cosine(emb, ref["embedding"])
        scores.append((score, ref["class"], ref["file"]))

    scores.sort(reverse=True, key=lambda x: x[0])
    best = scores[0]
    second = scores[1]

    confidence = best[0] - second[0]

    return {
        "pattern": best[1],
        "example": best[2],
        "score": best[0],
        "confidence": round(confidence, 4)
    }

@app.get("/")
def health():
    return {"status": "ok"}
