import os
os.environ["OPENCLIP_DISABLE_COCA"] = "1"
os.environ["OPENCLIP_SKIP_CUDA_CHECK"] = "1"

import io
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"

model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
)
model = model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")


concepts = [
    "financial chart",
    "candlestick chart",
    "price action",
    "stock chart",
    "random unrelated photo"
]

text_tokens = tokenizer(concepts)


def get_scores(image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = text_tokens.to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits = (image_features @ text_features.T) * 100
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    return {label: float(p) for label, p in zip(concepts, probs)}


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    scores = get_scores(image)

    chart_score = (
        scores["financial chart"]
        + scores["candlestick chart"]
        + scores["stock chart"]
        + scores["price action"]
    )

    random_score = scores["random unrelated photo"]

    return {
        "scores": scores,
        "is_valid_chart": bool(chart_score > random_score),
        "chart_confidence": float(chart_score),
        "random_confidence": float(random_score),
    }
