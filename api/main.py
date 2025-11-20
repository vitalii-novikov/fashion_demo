from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import io
from PIL import Image
from pydantic import BaseModel
from annoy import AnnoyIndex
import json


app = FastAPI(title="Fashion Style Classifier API")

import os
TORCHSERVE_URL = os.getenv("TORCHSERVE_URL", "http://model-server:8080/predictions/clip")
TORCHSERVE_TOKEN = os.getenv("TORCHSERVE_TOKEN", "")

STYLES = [
    "Casual", "Business Casual", "Formal", "Sport/Activewear",
    "Streetwear", "Minimalist", "Home wear", "Trendy/Fashion-forward"
]

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    try:
        headers = {"Authorization": f"Bearer {TORCHSERVE_TOKEN}"} if TORCHSERVE_TOKEN else {}
        files = {"data": ("image.jpg", image_bytes, file.content_type)}
        response = requests.post(TORCHSERVE_URL, files=files, headers=headers)
        response.raise_for_status()
        output = response.json()
    except requests.HTTPError as e:
        raise HTTPException(status_code=response.status_code, detail=f"Model server error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot contact model server: {e}")

    return output



INDEX_PATH = "/app/indexes/fashion_index.ann"
META_PATH = "/app/indexes/metadata.json"

# CLIP ViT-B/32 --> 512 dimensional embeddings
EMB_DIM = 512

print("Loading Annoy index...")
ann_index = AnnoyIndex(EMB_DIM, "angular")
ann_index.load(INDEX_PATH)
print("Annoy index loaded.")

print("Loading metadata...")
with open(META_PATH, "r") as f:
    metadata = json.load(f)
print(f"Loaded metadata for {len(metadata)} items.")


# --- Annoy index and metadata for recommendations ---
class EmbeddingRequest(BaseModel):
    embedding: list[float]
    k: int = 10


# --- GET RECOMMENDATIONS BY EMBEDDING ---
@app.post("/recommend")
def recommend(req: EmbeddingRequest):
    idxs, dists = ann_index.get_nns_by_vector(
        req.embedding, req.k, include_distances=True
    )

    results = []
    for idx, dist in zip(idxs, dists):
        item = metadata[idx]
        item["distance"] = float(dist)
        results.append(item)

    return {"recommendations": results}

@app.get("/dataset_size")
def dataset_size():
    return {"size": len(metadata)}

