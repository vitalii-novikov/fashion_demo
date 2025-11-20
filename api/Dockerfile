FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

EXPOSE 8000
ENV TORCHSERVE_URL=http://model-server:8080/predictions/clip
ENV TORCHSERVE_TOKEN=mysecret123

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
