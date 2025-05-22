ARG BASE_NAME=cpu
ARG TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"

###############################################################################
# NVIDIA BASE IMAGE
FROM nvcr.io/nvidia/pytorch:24.07-py3 AS nvidia

RUN apt-get update -y && apt-get install -y python3-venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV --system-site-packages
RUN . $VIRTUAL_ENV/bin/activate

ARG MAX_JOBS=8

# Put HPC-X MPI in the PATH, i.e. mpirun
ENV PATH=$PATH:/opt/hpcx/ompi/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/hpcx/ompi/lib

ARG TORCH_VERSION="2.4.0"
ARG TORCH_CUDA_ARCH_LIST="7.5"

RUN pip install uv

RUN git clone --branch v0.0.28.post1 https://github.com/facebookresearch/xformers.git
RUN uv pip install ninja
RUN cd xformers && \
    git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} pip install . --no-deps

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=nvidia

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

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=cpu

###############################################################################
# AMD BASE IMAGE
FROM gdiamos/rocm-base:v0.8 AS amd
ARG MAX_JOBS=8

ENV BASE_NAME=amd

RUN pip install amdsmi
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt install -y amd-smi-lib

RUN pip install pyhip>=1.1.0
ENV HIP_FORCE_DEV_KERNARG=1

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

###############################################################################
# VLLM BUILD STAGE
FROM ${BASE_NAME} AS vllm

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y curl ccache git vim numactl gcc-12 g++-12 libomp-dev libnuma-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 libdnnl-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ARG INSTALL_ROOT=/app/cray

COPY ./requirements.txt ${INSTALL_ROOT}/requirements.txt
COPY ./test/requirements-pytest.txt ${INSTALL_ROOT}/requirements-pytest.txt
COPY ./infra/requirements-vllm-build.txt ${INSTALL_ROOT}/requirements-vllm-build.txt

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

# Build vllm python package
RUN \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    MAX_JOBS=${MAX_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE} \
    python ${INSTALL_ROOT}/infra/cray_infra/setup.py bdist_wheel && \
    pip install ${INSTALL_ROOT}/infra/cray_infra/dist/*.whl && \
    rm -rf ${INSTALL_ROOT}/infra/cray_infra/dist

WORKDIR ${INSTALL_ROOT}

###############################################################################
# MAIN IMAGE
FROM vllm AS infra

# Build GPU-aware MPI
COPY ./infra/cray_infra/training/gpu_aware_mpi ${INSTALL_ROOT}/infra/cray_infra/training/gpu_aware_mpi
RUN python3 ${INSTALL_ROOT}/infra/cray_infra/training/gpu_aware_mpi/setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl

RUN apt-get update -y  \
    && apt-get install -y slurm-wlm libslurm-dev \
    build-essential \
    less curl wget net-tools vim iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Setup python path
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/infra"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/sdk"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/ml"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/test"

RUN mkdir -p ${INSTALL_ROOT}/jobs

# Copy the rest of the platform code
COPY ./infra ${INSTALL_ROOT}/infra
COPY ./sdk ${INSTALL_ROOT}/sdk
COPY ./test ${INSTALL_ROOT}/test
COPY ./ml ${INSTALL_ROOT}/ml
COPY ./scripts ${INSTALL_ROOT}/scripts

# Build SLURM plugin
RUN /app/cray/infra/slurm_src/compile.sh

ENV SLURM_CONF=${INSTALL_ROOT}/infra/slurm_configs/slurm.conf

