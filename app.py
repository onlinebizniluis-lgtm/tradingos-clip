import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import open_clip

# ====== FastAPI ======
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== Model Load Once ======
device = "cpu"

model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

CLASSES = ["uptrend", "downtrend", "consolidation"]
class_tokens = tokenizer(CLASSES).to(device)


@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "TradingOS CLIP Market Structure API Live 🔥"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # preprocess
    image_input = preprocess(image).unsqueeze(0).to(device)

    # encode
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(class_tokens)

        # cosine similarity
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    probs = probs.squeeze().tolist()

    result = {
        "prediction": CLASSES[int(torch.argmax(torch.tensor(probs)))],
        "confidence": {
            CLASSES[i]: float(round(p, 4)) for i, p in enumerate(probs)
        }
    }
    return result
