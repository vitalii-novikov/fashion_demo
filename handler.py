import torch
import io
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from ts.torch_handler.base_handler import BaseHandler
import os

STYLES = [
    "Casual", "Business Casual", "Formal", "Sport/Activewear",
    "Streetwear", "Minimalist", "Home wear", "Trendy/Fashion-forward"
]

class ClipHandler(BaseHandler):
    def initialize(self, ctx):
        model_dir = ctx.system_properties.get("model_dir")
        for name in os.listdir(model_dir):
            sub = os.path.join(model_dir, name)
            if os.path.isdir(sub) and "config.json" in os.listdir(sub):
                model_dir = sub
                break

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_dir).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_dir)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        item = data[0]

        if "body" in item and isinstance(item["body"], (bytes, bytearray)):
            image_bytes = item["body"]
        elif "data" in item:
            if isinstance(item["data"], dict) and "blob" in item["data"]:
                image_bytes = item["data"]["blob"]
            else:
                image_bytes = item["data"]
        else:
            raise ValueError("No file found in request")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(
            images=image,
            text=[f"This is {s} clothing" for s in STYLES],
            return_tensors="pt",
            padding=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def inference(self, inputs):
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds

            # Normalization
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            sims = (image_features @ text_features.T).squeeze(0).cpu().numpy()
            sims = sims.astype(float)

            # softmax
            probs = np.exp(sims) / np.sum(np.exp(sims))
            probs = probs.astype(float)

            top_idx = probs.argsort()[-2:][::-1]

            embedding = image_features.squeeze(0).cpu().numpy().astype(float).tolist()

            return {
                "main_style": STYLES[top_idx[0]],
                "main_confidence": float(round(probs[top_idx[0]] * 100, 2)),
                "secondary_style": STYLES[top_idx[1]],
                "secondary_confidence": float(round(probs[top_idx[1]] * 100, 2)),
                "embedding_dim": int(len(embedding)),
                "embedding": embedding
            }

    def postprocess(self, inference_output):
        return [inference_output]
