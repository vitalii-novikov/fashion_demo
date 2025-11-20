# build_annoy.py
import pandas as pd
from annoy import AnnoyIndex
import json
import os

EMB_PATH = "data/embeddings.csv"
INDEX_DIR = "indexes"
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Load embeddings ---
df = pd.read_csv(EMB_PATH)
df["embedding"] = df["embedding"].apply(lambda x: list(map(float, x.split(","))))

emb_dim = len(df["embedding"].iloc[0])
index = AnnoyIndex(emb_dim, "angular")

print(f"Embedding dim = {emb_dim}")
print("Building Annoy index...")

for i, row in df.iterrows():
    index.add_item(i, row["embedding"])

# More trees → better precision, slower build
index.build(50)
index.save(f"{INDEX_DIR}/fashion_index.ann")

print("Saving metadata...")
meta = df.drop(columns=["embedding"]).to_dict(orient="records")

with open(f"{INDEX_DIR}/metadata.json", "w") as f:
    json.dump(meta, f)

print("Done.")
