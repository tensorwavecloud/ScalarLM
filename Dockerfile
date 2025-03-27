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
RUN uv pip install xformers==0.0.27.post2

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

COPY ./infra/cray_infra/training/gpu_aware_mpi ${INSTALL_ROOT}/infra/cray_infra/training/gpu_aware_mpi
RUN python ${INSTALL_ROOT}/infra/cray_infra/training/gpu_aware_mpi/setup.py install

ENV BASE_NAME=cpu

###############################################################################
# AMD BASE IMAGE
FROM gdiamos/rocm-base AS amd
ARG MAX_JOBS=8

ENV BASE_NAME=amd

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt install -y amd-smi-lib
RUN pip install pyhip>=1.1.0
RUN pip install amdsmi
ENV HIP_FORCE_DEV_KERNARG=1

# Uninstall existing pytorch and dependencies
RUN pip uninstall torch torchvision torchaudio -y

# Set environment variables
ENV ROCM_PATH=/opt/rocm

# Clone and build UCX
RUN git clone https://github.com/openucx/ucx.git -b v1.15.x && \
    cd ucx && \
    ./autogen.sh && \
    ./configure --prefix=/opt/ucx-rocm \
      --with-rocm=$ROCM_PATH \
      --enable-mt && \
    make -j$(nproc) && make install

# Build ROCM-Aware Open MPI
RUN cd / && \
    wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.7.tar.gz && \
    tar -xvf openmpi-5.0.7.tar.gz && \
    cd openmpi-5.0.7 && \
    ./configure --prefix=/opt/ompi-rocm \
      --with-ucx=/opt/ucx-rocm \
      --with-rocm=$ROCM_PATH && \
    make -j$(nproc) && make install

ENV PATH="/opt/ompi-rocm/bin:${PATH}"

# Reinstall pytorch with ROCM and ROCM-aware MPI support
ENV MPI_HOME=/opt/ompi-rocm
ENV PATH=$MPI_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
ENV CMAKE_PREFIX_PATH=$MPI_HOME:$CMAKE_PREFIX_PATH
ENV ROCM_PATH=/opt/rocm
ENV PATH=$ROCM_PATH/bin:$PATH
ENV LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
ENV USE_ROCM=1
ENV USE_MPI=1
ENV MPI_INCLUDE_DIR=$MPI_HOME/include
ENV MPI_LIBRARY=$MPI_HOME/lib/libmpi.so

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install cmake ninja

# Clone PyTorch repository
WORKDIR /opt
RUN git clone --recursive https://github.com/pytorch/pytorch
WORKDIR /opt/pytorch

RUN git submodule sync && git submodule update --init --recursive

# Install PyTorch dependencies
RUN pip3 install -r requirements.txt

# Build PyTorch
RUN python3 setup.py clean
RUN python3 setup.py build
RUN python3 setup.py install

# Verify installation
RUN python3 -c "import torch; print(torch.__version__); print(torch.distributed.is_mpi_available()); print(torch.cuda.is_available()); print(torch.version.hip)"

# Set the default command
CMD ["/bin/bash"]

ARG INSTALL_ROOT=/app/cray
WORKDIR ${INSTALL_ROOT}

COPY ./infra/cray_infra/training/gpu_aware_mpi ${INSTALL_ROOT}/infra/cray_infra/training/gpu_aware_mpi
RUN python ${INSTALL_ROOT}/infra/cray_infra/training/gpu_aware_mpi/setup.py install

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
ARG TORCH_CUDA_ARCH_LIST="7.0 8.6"

# Build vllm python package
RUN \
    --mount=type=cache,target=/root/.cache/pip \
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

