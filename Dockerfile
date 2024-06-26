# syntax=docker/dockerfile:1

ARG CUDA_VERSION=11.8.0

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 as deploy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends python3-pip curl
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -U pip
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118

WORKDIR /app

# Remember to regenerate requirements.txt!
COPY --link requirements.txt .env ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install -r requirements.txt
RUN --mount=type=bind,source=./thirdparty,target=/thirdparty,rw \
    pip3 install --no-cache-dir /thirdparty/open_clip /thirdparty/ultralytics

COPY --link models/wiseft.bin ./models/wiseft.bin
COPY --link models/yolov9c-til24ufo-last.pt ./models/yolov9c-til24ufo-last.pt
COPY --link til24_vlm ./til24_vlm

EXPOSE 5004
ENV TORCH_CUDNN_V8_API_ENABLED=1 PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync CUDA_VISIBLE_DEVICES=0 YOLO_AUTOINSTALL=false YOLO_OFFLINE=true
# uvicorn --host=0.0.0.0 --port=5004 --factory til24_vlm:create_app
CMD ["uvicorn", "--log-level=warning", "--host=0.0.0.0", "--port=5004", "--factory", "til24_vlm:create_app"]
