ARG BASE_NAME=cpu

###############################################################################
# NVIDIA BASE IMAGE
FROM nvcr.io/nvidia/pytorch:23.05-py3@sha256:d5aa1e516e68afab9cd3ecaaeac3dd2178618bd26cd7ad96762ed53e32e9e0bd AS nvidia

RUN apt-get update -y && apt-get install -y python3-venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python -m venv $VIRTUAL_ENV
RUN . $VIRTUAL_ENV/bin/activate

ARG TORCH_VERSION="2.4.0"

RUN pip install uv
RUN uv pip install torch==${TORCH_VERSION}

###############################################################################
# CPU BASE IMAGE
FROM ubuntu:24.04 AS cpu

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y python3 python3-pip python3-venv

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV
RUN . $VIRTUAL_ENV/bin/activate

ARG TORCH_VERSION="2.4.0"

RUN pip install uv
RUN uv pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu

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
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0 VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE} python3 ${INSTALL_ROOT}/infra/cray_infra/setup.py bdist_wheel && \
    pip install ${INSTALL_ROOT}/infra/cray_infra/dist/*.whl && \
    rm -rf ${INSTALL_ROOT}/infra/cray_infra/dist

WORKDIR ${INSTALL_ROOT}

###############################################################################
# MAIN IMAGE
FROM vllm AS infra

RUN apt-get update -y  \
    && apt-get install -y slurm-wlm \
    mariadb-server build-essential munge libmunge-dev \
    less curl wget net-tools vim iputils-ping \
    openmpi-bin libopenmpi-dev libpmix-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy slurm config templates
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/infra"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/sdk"
ENV PYTHONPATH="${PYTHONPATH}:${INSTALL_ROOT}/ml"

ENV PATH=$PATH:${INSTALL_ROOT}/usr/bin
ENV SLURM_CONF=${INSTALL_ROOT}/infra/slurm_configs/slurm.conf

COPY ./infra ${INSTALL_ROOT}/infra
COPY ./sdk ${INSTALL_ROOT}/sdk
COPY ./test ${INSTALL_ROOT}/test
COPY ./cray ${INSTALL_ROOT}/cray
COPY ./ml ${INSTALL_ROOT}/ml
COPY ./scripts ${INSTALL_ROOT}/scripts
COPY ./infra/slurm_configs/munge.key /etc/munge/munge.key

