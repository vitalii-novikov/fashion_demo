# fashion_demo
CLIP model to recommend Fashion items based on embeddings

# Project run instructions

1) install the model and create .mar file using notebook:
    `run-one-time/model-install.ipynb`
2) build and run docker-compose:
    `%%bash 
        docker-compose up --build
    `
3) open `http://0.0.0.0:8501` to test the app

4) check logs at `https://wandb.ai/fhtw/fashion-recommender` with name "demo_fashion"