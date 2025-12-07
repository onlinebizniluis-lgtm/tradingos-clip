import os
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Model Setup (ONNX Runtime)
# -----------------------------
print("Loading ONNX CLIP model...")
session = ort.InferenceSession(
    "model/model.onnx",
    providers=["CPUExecutionProvider"]
)
print("Model loaded.")

# -----------------------------
# Categories
# -----------------------------
categories = ["uptrend", "downtrend", "consolidation"]
base_folder = "reference/market_structure"
class_embeddings = {c: None for c in categories}

# -----------------------------
# Helpers
# -----------------------------
def preprocess(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, 0)

def embed_image(img: Image.Image):
    x = preprocess(img)
    emb = session.run(None, {"image": x})[0]
    emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
    return emb

def load_folder_embeddings(path: str):
    embs = []
    for f in os.listdir(path):
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            img = Image.open(os.path.join(path, f)).convert("RGB")
            embs.append(embed_image(img))
    if not embs:
        return None
    return np.vstack(embs)

def ensure_loaded():
    for c in categories:
        if class_embeddings[c] is None:
            folder = os.path.join(base_folder, c)
            class_embeddings[c] = load_folder_embeddings(folder)

def cosine(a, b):
    return np.dot(b, a.T).flatten()

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/check_structure")
async def check_structure(file: UploadFile = File(...)):
    ensure_loaded()

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    query = embed_image(img)

    scores = {}
    for c in categories:
        ref = class_embeddings[c]
        if ref is None:
            scores[c] = 0.0
        else:
            sims = cosine(query, ref)
            scores[c] = float(np.max(sims))

    best = max(scores, key=scores.get)
    return {
        "structure": best,
        "score": scores[best],
        "details": scores
    }
