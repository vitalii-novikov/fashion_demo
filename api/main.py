import os
import io
import time
import json
import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from annoy import AnnoyIndex

# --- wandb + weave init ---
import wandb
wandb.login(key="5b4751f4a9800484a36a7d6c7ed7e34c82345c5a")

wandb_run = wandb.init(
    project="fashion-recommender",
    entity="fhtw",
    name="demo_fashion",
    mode="online",
    reinit=True
)

import weave
weave.init("fashion-recommender")


# --- FastAPI ---
app = FastAPI(title="Fashion Style Classifier API")

# --- Model server configuration ---
TORCHSERVE_URL = os.getenv("TORCHSERVE_URL", "http://model-server:8080/predictions/clip")
TORCHSERVE_TOKEN = os.getenv("TORCHSERVE_TOKEN", "")


# --- Load Annoy index + metadata ---
INDEX_PATH = "/app/indexes/fashion_index.ann"
META_PATH = "/app/indexes/metadata.json"
EMB_DIM = 512

print("Loading Annoy index...")
ann_index = AnnoyIndex(EMB_DIM, "angular")
ann_index.load(INDEX_PATH)
print("Annoy index loaded.")

print("Loading metadata...")
with open(META_PATH, "r") as f:
    metadata = json.load(f)
print(f"Loaded metadata for {len(metadata)} items.")


# --- Pydantic models ---
class EmbeddingRequest(BaseModel):
    embedding: list[float]
    k: int
    randomness: float | None = 0.0


# -------------------------------------------------------------------------
#                               CLASSIFY
# -------------------------------------------------------------------------
@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    start_time = time.time()

    # Read image
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
    except Exception as e:
        wandb.log({"classify/error": str(e)})
        raise HTTPException(status_code=400, detail=f"Cannot read image: {e}")

    # Call TorchServe model
    try:
        headers = {"Authorization": f"Bearer {TORCHSERVE_TOKEN}"} if TORCHSERVE_TOKEN else {}
        files = {"data": ("image.jpg", image_bytes, file.content_type)}

        response = requests.post(TORCHSERVE_URL, files=files, headers=headers)
        response.raise_for_status()
        output = response.json()

    except Exception as e:
        wandb.log({"classify/error": str(e)})
        raise HTTPException(status_code=500, detail=f"Model server error: {e}")

    # Log to wandb
    wandb.log({
        "classify/img_width": width,
        "classify/img_height": height,
        "classify/main_style": output.get("main_style"),
        "classify/main_confidence": output.get("main_confidence"),
        "classify/secondary_style": output.get("secondary_style"),
        "classify/secondary_confidence": output.get("secondary_confidence"),
        "classify/embedding_norm": float(sum(x*x for x in output.get("embedding", []))) ** 0.5,
        "classify/duration_ms": int((time.time() - start_time) * 1000)
    })

    return output


# -------------------------------------------------------------------------
#                               RECOMMEND
# -------------------------------------------------------------------------
@app.post("/recommend")
def recommend(req: EmbeddingRequest):
    start_time = time.time()

    # Query Annoy index
    try:
        idxs, dists = ann_index.get_nns_by_vector(
            req.embedding, req.k, include_distances=True
        )
    except Exception as e:
        wandb.log({"recommend/error": str(e)})
        raise HTTPException(status_code=500, detail="Annoy index failure")

    # Build result list
    results = []
    for idx, dist in zip(idxs, dists):
        item = metadata[idx]
        item = dict(item)  # ensure copy
        item["distance"] = float(dist)
        results.append(item)

    # Log wandb tracking
    wandb.log({
        "recommend/k_requested": req.k,
        "recommend/distance_mean": float(sum(dists) / len(dists)) if dists else None,
        "recommend/distance_min": float(min(dists)) if dists else None,
        "recommend/distance_max": float(max(dists)) if dists else None,
        "recommend/duration_ms": int((time.time() - start_time) * 1000)
    })

    return {"recommendations": results}


# -------------------------------------------------------------------------
#                               DATASET INFO
# -------------------------------------------------------------------------
@app.get("/dataset_size")
def dataset_size():
    return {"size": len(metadata)}
