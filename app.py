import os
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import Dict

# --------------------------------------------------
# FASTAPI SETUP
# --------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------
device = "cpu"

# --------------------------------------------------
# LOAD CLIP MODEL (smallest version available)
# MobileCLIP-S1 uses tiny built-in weights
# no pretrained tag -> minimal RAM
# --------------------------------------------------
print("Loading CLIP model...")
model, preprocess, _ = open_clip.create_model_and_transforms(
    "MobileCLIP-S1"
)
model.to(device).eval()
print("Model loaded.")

# --------------------------------------------------
# YOUR CATEGORIES
# --------------------------------------------------
base_folder = "reference/market_structure"
categories = ["uptrend", "downtrend", "consolidation"]

# Lazy-loaded embeddings (loads on first request)
class_embeddings: Dict[str, torch.Tensor] = {
    c: None for c in categories
}

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def embed_image(img: Image.Image) -> torch.Tensor:
    """Encode image using the CLIP model."""
    image = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(image)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu()

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity."""
    return (a @ b.T).squeeze(0)

def load_folder_embeddings(folder_path: str) -> torch.Tensor:
    """Load all images in a folder and compute embeddings."""
    embs = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
            embs.append(embed_image(img))
    if not embs:
        return None
    return torch.stack(embs)

# --------------------------------------------------
# PRELOAD ON FIRST REQUEST
# --------------------------------------------------
def ensure_embeddings_loaded():
    """Load embeddings only when needed."""
    for cat in categories:
        if class_embeddings[cat] is None:
            folder = os.path.join(base_folder, cat)
            class_embeddings[cat] = load_folder_embeddings(folder)

# --------------------------------------------------
# API ENDPOINTS
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/check_structure")
async def check_structure(file: UploadFile = File(...)):
    ensure_embeddings_loaded()

    # read image
    contents = await file.read()
    img = Image.open(
        io.BytesIO(contents)
    ).convert("RGB")

    # embed upload
    query_emb = embed_image(img)

    # similarity
    scores = {}
    for cat in categories:
        ref_embs = class_embeddings[cat]
        if ref_embs is None:
            scores[cat] = 0.0
        else:
            sims = cosine_similarity(query_emb, ref_embs)
            scores[cat] = float(torch.max(sims).item())

    # pick best
    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    return {
        "structure": best_cat,
        "score": best_score,
        "details": scores
    }
