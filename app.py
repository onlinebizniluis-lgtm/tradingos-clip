import os
os.environ["OPENCLIP_DISABLE_COCA"] = "1"

import io
import torch
import open_clip
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------
# App setup
# -----------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Load CLIP
# -----------------------------------
device = "cpu"

model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai",
)
model = model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")


# -----------------------------------
# Inference: similarity to chart concepts
# -----------------------------------
concepts = [
    "financial chart",
    "candlestick chart",
    "price action",
    "stock chart",
    "random unrelated photo"
]
text_tokens = tokenizer(concepts)


def get_scores(image: Image.Image):
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = text_tokens.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False):
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logits = (image_features @ text_features.T) * 100
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    result = {}
    for label, p in zip(concepts, probs):
        result[label] = float(p)
    return result


# -----------------------------------
# Routes
# -----------------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "CLIP engine ready"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    scores = get_scores(image)

    # probability it's a chart
    chart_score = (
        scores["financial chart"]
        + scores["candlestick chart"]
        + scores["stock chart"]
        + scores["price action"]
    )

    # random photo detection
    random_score = scores["random unrelated photo"]

    # final logic
    is_chart = chart_score > random_score

    return {
        "scores": scores,
        "is_valid_chart": bool(is_chart),
        "chart_confidence": float(chart_score),
        "random_confidence": float(random_score),
    }
