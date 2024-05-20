#!/bin/bash
set -e

export VISION_VERSION=0.17.2
export FORCE_CUDA=1

export PYTORCH_BUILD_VERSION=2.2.2
export PYTORCH_BUILD_NUMBER=0
# Cannot be auto-detected during docker build.
export TORCH_CUDA_ARCH_LIST="7.5 8.9"
export CUDA_HOME=/usr/local/cuda

# Need curl to download stuff.
apt-get update
apt-get install -y --no-install-recommends curl

# Install & setup conda.
mkdir -p /tmp/miniconda3
curl -sLo /tmp/miniconda3/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_24.3.0-0-Linux-x86_64.sh
bash /tmp/miniconda3/miniconda.sh -b -u -p /tmp/miniconda3
rm -rf /tmp/miniconda3/miniconda.sh
eval "$(/tmp/miniconda3/bin/conda shell.bash hook)"

# Setup build environment.
conda config --set channel_priority flexible
magma_pkg=$(nvcc --version | sed -ne 's/.*_\([0-9]\+\)\.\([0-9]\+\).*/pytorch::magma-cuda\1\2/p')
# conda install -y --no-update-deps cmake ninja intel::mkl-static intel::mkl-include $magma_pkg
conda install -y --no-update-deps cmake ninja libjpeg-turbo libpng $magma_pkg

# Build & install torch.
mkdir -p /tmp/pytorch
curl -Ls https://github.com/pytorch/pytorch/releases/download/v${PYTORCH_BUILD_VERSION}/pytorch-v${PYTORCH_BUILD_VERSION}.tar.gz | tar --strip-components=1 -xzC /tmp/pytorch
pip install -r /tmp/pytorch/requirements.txt
cd /tmp/pytorch
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
# Temporary bug: https://github.com/pytorch/pytorch/issues/105248
rm -f /tmp/miniconda3/lib/libstdc++.so.6
CFLAGS=" -Os " \
  ONNX_ML=0 USE_CUDNN=0 USE_CUSPARSELT=0 \
  USE_FBGEMM=0 USE_KINETO=0 BUILD_TEST=0 \
  USE_MKLDNN=0 USE_ITT=0 USE_NNPACK=0 \
  USE_QNNPACK=0 USE_DISTRIBUTED=0 USE_TENSORPIPE=0 \
  USE_GLOO=0 USE_MPI=0 USE_XNNPACK=0 \
  USE_PYTORCH_QNNPACK=0 USE_MKL=0 USE_OPENMP=0 \
  python setup.py install
cd /

# Build & install torchvision.
mkdir -p /tmp/torchvision
curl -Ls https://github.com/pytorch/vision/archive/refs/tags/v${VISION_VERSION}.tar.gz | tar --strip-components=1 -xzC /tmp/torchvision
cd /tmp/torchvision
BUILD_VERSION=${VISION_VERSION} python setup.py install
cd /

# Copy all the wheels to /whl and clean up.
mkdir -p /whl
cd /tmp/pytorch
python setup.py bdist_wheel
cp dist/*.whl /whl/
cd /tmp/torchvision
BUILD_VERSION=${VISION_VERSION} python setup.py bdist_wheel
cp dist/*.whl /whl/
cd /
# rm -rf /tmp/miniconda3 /tmp/pytorch /tmp/torchvision
