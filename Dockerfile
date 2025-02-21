FROM nvcr.io/nvidia/pytorch:25.01-py3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git-lfs \
        libsm6 \
        libxext6 \
        libgl1 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN git lfs install

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install omegaconf pydantic pytorch-lightning clearml pycocotools numpy tqdm pyyaml webcolors matplotlib flash-attn



