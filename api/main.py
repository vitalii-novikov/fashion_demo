from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import io
from PIL import Image

app = FastAPI(title="Fashion Style Classifier API")

import os
TORCHSERVE_URL = os.getenv("TORCHSERVE_URL", "http://model-server:8080/predictions/clip")
TORCHSERVE_TOKEN = os.getenv("TORCHSERVE_TOKEN", "")

# Список стилей
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
        # Отправляем изображение на TorchServe с токеном
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
