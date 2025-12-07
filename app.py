import os
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import open_clip

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP Model
print("Loading CLIP model...")
model, preprocess, _ = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
model = model.to(device).eval()
print("Model loaded.")


def load_folder_embeddings(folder: str):
        if not os.path.exists(folder):
            print(f"Warning: folder not found → {folder}")
            return None

        tensors = []

        for name in os.listdir(folder):
            path = os.path.join(folder, name)

            if not path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            try:
                img = Image.open(path).convert("RGB")
                img_t = preprocess(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    emb = model.encode_image(img_t)
                    emb = emb / emb.norm(dim=-1, keepdim=True)

                tensors.append(emb)

            except Exception as e:
                print(f"Error loading {path}: {e}")
                pass

        if not tensors:
            print(f"No images in {folder}")
            return None

        return torch.cat(tensors, dim=0)


# Load Reference Data
print("Loading reference datasets...")
base = "reference/market_structure"
categories = ["uptrend", "downtrend", "consolidation"]
class_embeddings = {}

for cat in categories:
    folder = os.path.join(base, cat)
    emb = load_folder_embeddings(folder)
    if emb is not None:
        class_embeddings[cat] = emb

print("Reference loading complete:", list(class_embeddings.keys()))

MIN_VALID_SCORE = 74.0


def compute_scores(upload_emb):
    upload_emb = upload_emb / upload_emb.norm(dim=-1, keepdim=True)

    scores = {}

    for cat, emb in class_embeddings.items():
        score = (emb @ upload_emb.T).max().item()
        scores[cat] = round(score * 100, 2)

    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    best_class, best_score = ordered[0]
    second_class, second_score = ordered[1]

    purity = round(best_score - second_score, 2)

    return scores, best_class, best_score, purity, ordered[1][0]


@app.post("/check_structure")
async def check_structure(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(BytesIO(data)).convert("RGB")
    img_t = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_t)

    scores, best, best_score, purity, confusion = compute_scores(emb)

    if best_score < MIN_VALID_SCORE:
        return {
            "valid_chart": False,
            "reason": "Low similarity to known chart patterns",
            "scores": scores,
            "best": best,
            "best_score": best_score,
            "purity": purity,
            "confusion": confusion
        }

    # Confidence levels
    if purity >= 3:
        confidence = "high"
    elif purity >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "valid_chart": True,
        "chart_type": best,
        "confidence": confidence,
        "scores": scores,
        "best": best,
        "best_score": best_score,
        "purity": purity,
        "confusion": confusion
    }


@app.get("/")
def home():
    return {"status": "ok", "message": "CLIP chart structure API running"}

