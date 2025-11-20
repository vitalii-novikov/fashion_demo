FROM pytorch/torchserve:latest-cpu

RUN pip install --no-cache-dir \
    transformers pillow numpy safetensors

COPY model-store /home/model-server/model-store
COPY config.properties /home/model-server/config.properties

EXPOSE 8080 8081

CMD ["torchserve", "--start", \
     "--ncs", \
     "--ts-config", "/home/model-server/config.properties", \
     "--model-store", "/home/model-server/model-store", \
     "--models", "clip=clip.mar"]
