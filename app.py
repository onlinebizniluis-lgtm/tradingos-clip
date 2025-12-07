import io
import gc
import time
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import open_clip
import numpy as np

app = FastAPI()

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# PHASE 1 MARKET STRUCTURE
# ---------------------------
MARKET_STRUCTURE_TEXT = [
    "uptrend market structure",
    "downtrend market structure",
    "consolidation market structure"
]

def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16",
        pretrained="laion400m_e32",
    )
    model.eval()
    return model, preprocess


def preprocess_image(preprocess, img: Image.Image):
    return preprocess(img).unsqueeze(0)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    model, preprocess = load_model()

    # Image Load
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_tensor = preprocess_image(preprocess, image)

    # Encode text
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    tokens = tokenizer(MARKET_STRUCTURE_TEXT)

    with torch.no_grad():
        img_embed = model.encode_image(image_tensor)
        text_embed = model.encode_text(tokens)

        # Normalize
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        similarities = (img_embed @ text_embed.T).squeeze().tolist()

    del model
    gc.collect()

    time.sleep(2.0)  # dramatic analysis delay

    best_index = int(np.argmax(similarities))

    return {
        "best_match": MARKET_STRUCTURE_TEXT[best_index],
        "scores": [
            {"structure": MARKET_STRUCTURE_TEXT[i], "score": float(similarities[i])}
            for i in range(len(MARKET_STRUCTURE_TEXT))
        ]
    }


@app.get("/")
def home():
    return {
        "status": "active",
        "phase": "Market Structure Only",
        "structures": MARKET_STRUCTURE_TEXT
    }
