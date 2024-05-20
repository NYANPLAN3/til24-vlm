# syntax=docker/dockerfile:1

ARG CUDA_VERSION=12.1.1

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 as deploy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
  echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y --no-install-recommends \
  cuda-nvtx-12-1 \
  libcusparse-12-1 \
  libcufft-12-1 \
  libcurand-12-1 \
  libcublas-12-1 \
  libnvjitlink-12-1 \
  libnvjpeg-12-1 \
  libjpeg-turbo8 \
  libpng16-16
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y --no-install-recommends python3-pip
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
  pip install -U pip
RUN --mount=type=bind,source=./whl,target=/whl \
  pip install --no-cache-dir /whl/*

WORKDIR /app
COPY --link models ./models

# Remember to regenerate requirements.txt!
COPY --link requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
  pip install -r requirements.txt

COPY --link til24_vlm ./til24_vlm

EXPOSE 5004
# uvicorn --host=0.0.0.0 --port=5004 --factory til24_vlm:create_app
CMD ["uvicorn", "--host=0.0.0.0", "--port=5004", "--factory", "til24_vlm:create_app"]
