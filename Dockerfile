ARG BASE_NAME=cpu

###############################################################################
# NVIDIA BASE IMAGE
FROM nvcr.io/nvidia/pytorch:25.05-py3 AS nvidia

RUN apt-get update -y && apt-get install -y python3-venv slurm-wlm libslurm-dev

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV --system-site-packages
RUN . $VIRTUAL_ENV/bin/activate

# Put HPC-X MPI in the PATH, i.e. mpirun
ENV PATH=$PATH:/opt/hpcx/ompi/bin
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/opt/hpcx/ompi/lib

ARG TORCH_VERSION="2.8.0"
ARG TORCH_CUDA_ARCH_LIST="7.5"

RUN pip install uv
RUN uv pip install ninja


ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=nvidia

ENV VLLM_USE_STANDALONE_COMPILE=0

###############################################################################
# CPU BASE IMAGE
FROM ubuntu:24.04 AS cpu

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y python3 python3-pip python3-venv \
    openmpi-bin libopenmpi-dev libpmix-dev slurm-wlm libslurm-dev \
    cmake

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV
RUN . $VIRTUAL_ENV/bin/activate

ARG TORCH_VERSION="2.7.1"

RUN pip install uv
RUN uv pip install torch==${TORCH_VERSION}+cpu --index-url https://download.pytorch.org/whl/cpu

# Put torch on the LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/app/.venv/lib64/python3.12/site-packages/torch/lib

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV BASE_NAME=cpu

###############################################################################
# AMD BASE IMAGE
FROM gdiamos/rocm-base:v0.99 AS amd

ENV BASE_NAME=amd

RUN pip install pyhip>=1.1.0
ENV HIP_FORCE_DEV_KERNARG=1

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/venv/lib/python3.12/site-packages/torch/lib:/usr/local/rdma-lib

# vLLM dependencies
COPY ./infra/requirements-vllm-rocm.txt ${INSTALL_ROOT}/requirements-vllm-rocm.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm-rocm.txt

###############################################################################
# VLLM BUILD STAGE
FROM ${BASE_NAME} AS vllm

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y curl ccache git vim numactl gcc-12 g++-12 libomp-dev libnuma-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 libdnnl-dev \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ARG INSTALL_ROOT=/app/cray

WORKDIR ${INSTALL_ROOT}

# Install build dependencies FIRST
RUN pip install setuptools-scm

# Configure vLLM source - can use either local directory or remote repo
ARG VLLM_SOURCE=remote
ARG VLLM_BRANCH=main
ARG VLLM_REPO=https://github.com/supermassive-intelligence/vllm-fork.git

# Handle vLLM source - support both local and remote modes
COPY scripts/build-copy-vllm.sh ${INSTALL_ROOT}/build-copy-vllm.sh

# Remote mode (default): clone from repository without mounting
RUN if [ "${VLLM_SOURCE}" = "remote" ]; then \
        bash ${INSTALL_ROOT}/build-copy-vllm.sh ${VLLM_SOURCE} ${INSTALL_ROOT}/vllm \
        /dev/null ${VLLM_REPO} ${VLLM_BRANCH}; \
    fi

# Local mode: mount ./vllm directory and copy
RUN --mount=type=bind,source=./vllm,target=/workspace/vllm \
    if [ "${VLLM_SOURCE}" = "local" ]; then \
        bash ${INSTALL_ROOT}/build-copy-vllm.sh ${VLLM_SOURCE} ${INSTALL_ROOT}/vllm \
        /workspace/vllm ${VLLM_REPO} ${VLLM_BRANCH}; \
    fi

WORKDIR ${INSTALL_ROOT}/vllm

# Set build environment variables for CPU compilation
ARG TORCH_CUDA_ARCH_LIST="7.5"
ARG VLLM_TARGET_DEVICE=cpu

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}
ENV CMAKE_BUILD_TYPE=Release

# vLLM dependencies
COPY ./infra/requirements-vllm.txt ${INSTALL_ROOT}/requirements-vllm.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm.txt

# Set fallback version for setuptools-scm in case git metadata is missing
# This handles cases where git history might be incomplete
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.6.3.post1

RUN python ${INSTALL_ROOT}/vllm/use_existing_torch.py

RUN export MAX_JOBS=$(($(nproc) < $(free -g | awk '/^Mem:/ {print int($2/4)}') ? $(nproc) : $(free -g | awk '/^Mem:/ {print int($2/4)}')))

RUN \
    --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    pip install --no-build-isolation -e . --verbose

WORKDIR ${INSTALL_ROOT}

###############################################################################
# MAIN IMAGE
FROM vllm AS infra

# Build GPU-aware MPI
COPY ./infra/cray_infra/training/gpu_aware_mpi ${INSTALL_ROOT}/infra/cray_infra/training/gpu_aware_mpi
RUN python3 ${INSTALL_ROOT}/infra/cray_infra/training/gpu_aware_mpi/setup.py bdist_wheel --dist-dir=dist && \
    pip install dist/*.whl

RUN apt-get update -y  \
    && apt-get install -y build-essential \
    less curl wget net-tools vim iputils-ping strace gdb python3-dbg python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Setup python path
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/infra"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/sdk"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/ml"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/test"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/vllm"

# Megatron dependencies (GPU only)
# note this has to happen after vllm because it overrides some packages installed by vllm
COPY ./infra/requirements-megatron.txt ${INSTALL_ROOT}/requirements-megatron.txt
RUN if [ "$VLLM_TARGET_DEVICE" != "cpu" ]; then \
        uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-megatron.txt; \
    fi

# SDK and Infra dependencies
COPY ./requirements.txt ${INSTALL_ROOT}/requirements.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements.txt

RUN mkdir -p ${INSTALL_ROOT}/jobs
RUN mkdir -p ${INSTALL_ROOT}/nfs

# Copy the rest of the platform code
COPY ./infra ${INSTALL_ROOT}/infra
COPY ./sdk ${INSTALL_ROOT}/sdk
COPY ./test ${INSTALL_ROOT}/test
COPY ./ml ${INSTALL_ROOT}/ml
COPY ./scripts ${INSTALL_ROOT}/scripts

WORKDIR ${INSTALL_ROOT}

# Build SLURM plugin
RUN /app/cray/infra/slurm_src/compile.sh

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PYTHONPATH}:/usr/local/lib/slurm

ENV SLURM_CONF=${INSTALL_ROOT}/nfs/slurm.conf
ENV VLLM_CPU_MOE_PREPACK=0

