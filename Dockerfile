###############################################################################
# CPU BASE IMAGE
FROM ubuntu:22.04 AS cpu

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
    && apt-get install -y curl ccache git vim numactl gcc-12 g++-12 python3 python3-pip python3-venv libomp-dev libnuma-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN python3 -m venv $VIRTUAL_ENV
RUN . $VIRTUAL_ENV/bin/activate

ARG TORCH_CPU="2.4.0"

RUN pip install uv
RUN uv pip install torch==${TORCH_CPU} --index-url https://download.pytorch.org/whl/cpu

###############################################################################
# VLLM BUILD STAGE
FROM cpu AS vllm

ARG INSTALL_ROOT=/app/cray

COPY ./infra/cray_infra/vllm ${INSTALL_ROOT}/vllm
COPY ./infra/setup.py ${INSTALL_ROOT}/setup.py

COPY ./infra/CMakeLists.txt ${INSTALL_ROOT}/CMakeLists.txt
COPY ./infra/cmake ${INSTALL_ROOT}/cmake
COPY ./infra/csrc ${INSTALL_ROOT}/csrc

COPY ./requirements.txt ${INSTALL_ROOT}/requirements.txt
COPY ./infra/requirements-vllm-build.txt ${INSTALL_ROOT}/requirements-vllm-build.txt
COPY ./infra/requirements-vllm-cpu.txt ${INSTALL_ROOT}/requirements-cpu.txt

RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements.txt
RUN uv pip install --no-compile --no-cache-dir -r ${INSTALL_ROOT}/requirements-vllm-build.txt

WORKDIR ${INSTALL_ROOT}

# Build vllm python package
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=bind,source=.git,target=.git \
    VLLM_TARGET_DEVICE=cpu python3 ${INSTALL_ROOT}/setup.py bdist_wheel && \
    pip install ${INSTALL_ROOT}/dist/*.whl && \
    rm -rf ${INSTALL_ROOT}/dist
