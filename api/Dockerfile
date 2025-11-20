FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY indexes/ /app/indexes/

EXPOSE 8000
ENV TORCHSERVE_URL=http://model-server:8080/predictions/clip
ENV TORCHSERVE_TOKEN=mysecret123

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
