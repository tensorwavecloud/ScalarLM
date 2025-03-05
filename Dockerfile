ARG BASE_NAME=cpu

###############################################################################
# NVIDIA BASE IMAGE
FROM nvcr.io/nvidia/pytorch:24.11-py3 AS nvidia

RUN apt-get update -y && apt-get install -y python3-venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV
RUN . $VIRTUAL_ENV/bin/activate

ARG MAX_JOBS=8

# Put HPC-X MPI in the PATH, i.e. mpirun
ENV PATH=/opt/hpcx/ompi/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:$LD_LIBRARY_PATH

ARG TORCH_VERSION="2.4.0"

RUN pip install uv
RUN uv pip install torch==${TORCH_VERSION}

###############################################################################
# CPU BASE IMAGE
FROM ubuntu:24.04 AS cpu

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y python3 python3-pip python3-venv \
    openmpi-bin libopenmpi-dev libpmix-dev

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV
RUN . $VIRTUAL_ENV/bin/activate

ARG MAX_JOBS=4
#ENV DNNL_DEFAULT_FPMATH_MODE=F32

ARG TORCH_VERSION="2.4.0"

RUN pip install uv
RUN uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu

###############################################################################
# AMD BASE IMAGE
FROM rocm/dev-ubuntu-22.04:6.3.1-complete AS amd
ARG PYTORCH_ROCM_ARCH=gfx90a;gfx942
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ARG PYTHON_VERSION=3.12
ARG HIPBLASLT_BRANCH="4d40e36"
ARG HIPBLAS_COMMON_BRANCH="7c1566b"
ARG LEGACY_HIPBLASLT_OPTION=
ARG PYTORCH_BRANCH="3a585126"
ARG PYTORCH_VISION_BRANCH="v0.19.1"
ARG PYTORCH_REPO="https://github.com/pytorch/pytorch.git"
ARG PYTORCH_VISION_REPO="https://github.com/pytorch/vision.git"
ARG FA_BRANCH="b7d29fb"
ARG FA_REPO="https://github.com/ROCm/flash-attention.git"

ARG TRITON_BRANCH="e5be006"
ARG TRITON_REPO="https://github.com/triton-lang/triton.git"
ARG MAX_JOBS=128
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app
WORKDIR /app

# Install Python and other dependencies
RUN apt-get update -y \
    && apt-get install -y software-properties-common git curl sudo vim less \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
       python${PYTHON_VERSION}-lib2to3 python-is-python3  \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version \
    && pip install uv \
    && pip install -U packaging cmake ninja wheel setuptools pybind11 Cython \
    && apt-get update && apt-get install -y git && apt-get install -y cmake  && apt-get install -y python3.10-venv \
    && apt-get update && apt-get install -y openmpi-bin libopenmpi-dev

RUN mkdir -p /app/install

RUN git clone https://github.com/ROCm/hipBLAS-common.git \
    && cd hipBLAS-common \
    && git checkout ${HIPBLAS_COMMON_BRANCH} \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make package \
    && dpkg -i ./*.deb

RUN git clone https://github.com/ROCm/hipBLASLt \
    && cd hipBLASLt \
    && git checkout ${HIPBLASLT_BRANCH} \
    && ./install.sh -d --architecture ${PYTORCH_ROCM_ARCH} ${LEGACY_HIPBLASLT_OPTION} \
    && cd build/release \
    && make package -j$(nproc) \
    && echo "Searching for build and .deb files:" \
    && find . -type d -name "build" \
    && find . -name "*.deb" 

RUN cp /app/hipBLASLt/build/release/*.deb /app/hipBLAS-common/build/*.deb /app/install

RUN git clone ${TRITON_REPO}
RUN cd triton \
    && git checkout ${TRITON_BRANCH} \
    && cd python \
    && python3 setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl
RUN cp /app/triton/python/dist/*.whl /app/install

# Set working directory
WORKDIR /pytorch

# Clone and build PyTorch
RUN git clone ${PYTORCH_REPO} pytorch && \
    cd pytorch && \
    git checkout ${PYTORCH_BRANCH} && \
    pip install -r requirements.txt && \
    git submodule update --init --recursive && \
    python3 tools/amd_build/build_amd.py && \
    CMAKE_PREFIX_PATH=$(python3 -c 'import sys; print(sys.prefix)') python3 setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl

# Clone and build torchvision
RUN git clone ${PYTORCH_VISION_REPO} vision && \
    cd vision && \
    git checkout ${PYTORCH_VISION_BRANCH} && \
    python3 setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl

# Clone and build flash-attention
RUN git clone ${FA_REPO} flash-attention && \
    cd flash-attention && \
    git checkout ${FA_BRANCH} && \
    git submodule update --init && \
    MAX_JOBS=${MAX_JOBS} GPU_ARCHS=${PYTORCH_ROCM_ARCH} python3 setup.py bdist_wheel --dist-dir=dist

# Copy all wheel files to installation directory
RUN cp /pytorch/pytorch/dist/*.whl /app/install && \
    cp /pytorch/vision/dist/*.whl /app/install && \
    cp /pytorch/flash-attention/dist/*.whl /app/install


###############################################################################
# VLLM BUILD STAGE
FROM ${BASE_NAME} AS vllm

# Use test command to check if the directory exists
RUN if [ -d "/install" ]; then \
    dpkg -i /install/*.deb \
    && sed -i 's/, hipblaslt-dev \(.*\), hipcub-dev/, hipcub-dev/g' /var/lib/dpkg/status \
    && sed -i 's/, hipblaslt \(.*\), hipfft/, hipfft/g' /var/lib/dpkg/status \
    && pip install /install/*.whl; \
    fi

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y curl ccache git vim numactl gcc-12 g++-12 libomp-dev libnuma-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 libdnnl-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ARG INSTALL_ROOT=/app/cray

COPY ./requirements.txt ${INSTALL_ROOT}/requirements.txt
COPY ./test/requirements-pytest.txt ${INSTALL_ROOT}/requirements-pytest.txt
COPY ./infra/requirements-vllm-build.txt ${INSTALL_ROOT}/requirements-vllm-build.txt

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

RUN pip install /app/install/*.whl

RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm-build.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-pytest.txt

WORKDIR ${INSTALL_ROOT}

COPY ./infra/cray_infra/vllm ${INSTALL_ROOT}/infra/cray_infra/vllm
COPY ./infra/setup.py ${INSTALL_ROOT}/infra/cray_infra/setup.py

COPY ./infra/CMakeLists.txt ${INSTALL_ROOT}/infra/cray_infra/CMakeLists.txt
COPY ./infra/cmake ${INSTALL_ROOT}/infra/cray_infra/cmake
COPY ./infra/csrc ${INSTALL_ROOT}/infra/cray_infra/csrc

COPY ./infra/requirements-vllm.txt ${INSTALL_ROOT}/infra/cray_infra/requirements.txt

WORKDIR ${INSTALL_ROOT}/infra/cray_infra

ARG VLLM_TARGET_DEVICE=cpu
ARG TORCH_CUDA_ARCH_LIST="7.0 8.6"

# Build vllm python package
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST=${PYTORCH_ROCM_ARCH:-${TORCH_CUDA_ARCH_LIST}} \
    VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE} \
    python ${INSTALL_ROOT}/infra/cray_infra/setup.py bdist_wheel && \
    pip install ${INSTALL_ROOT}/infra/cray_infra/dist/*.whl && \
    rm -rf ${INSTALL_ROOT}/infra/cray_infra/dist

WORKDIR ${INSTALL_ROOT}

###############################################################################
# MAIN IMAGE
FROM vllm AS infra

RUN apt-get update -y  \
    && apt-get install -y slurm-wlm libslurm-dev \
    build-essential \
    less curl wget net-tools vim iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Build SLURM
COPY ./infra/slurm_src ${INSTALL_ROOT}/infra/slurm_src
RUN /app/cray/infra/slurm_src/compile.sh

# Copy slurm config templates
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/infra"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/sdk"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/ml"

ENV SLURM_CONF=${INSTALL_ROOT}/infra/slurm_configs/slurm.conf

RUN mkdir -p ${INSTALL_ROOT}/jobs

COPY ./infra ${INSTALL_ROOT}/infra
COPY ./sdk ${INSTALL_ROOT}/sdk
COPY ./test ${INSTALL_ROOT}/test
COPY ./cray ${INSTALL_ROOT}/cray
COPY ./ml ${INSTALL_ROOT}/ml
COPY ./scripts ${INSTALL_ROOT}/scripts

