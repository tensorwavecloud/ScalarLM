inspect_args

target=${args[target]}
sm_arch=${args[sm_arch]}

declare -a vllm_target_device
declare -a docker_compose_service
declare -a docker_platform

if [ "$target" == "cpu" ]; then
    vllm_target_device=("cpu")
    docker_compose_service="cray"
    if [ "$(uname -m)" == "x86_64" ]; then
        docker_platform=("linux/amd64")
    else
        docker_platform=("linux/arm64/v8")
    fi
elif [ "$target" == "amd" ]; then
    vllm_target_device=("rocm")
    docker_compose_service="cray-amd"
    docker_platform="linux/amd64"
    sm_arch="gfx942"
else
    vllm_target_device=("cuda")
    docker_compose_service="cray-nvidia"
    docker_platform="linux/amd64"
    if [ "$sm_arch" == "auto" ]; then
        echo "Autodetect sm_arch"
        # Auto-detect the architecture of the GPU using nvidia-smi
        sm_arch=($(nvidia-smi --query-gpu=compute_cap --format=csv,noheader))
    fi
fi

mkdir -p vllm

echo "SM arch is ${sm_arch}"

BASE_NAME=${target} VLLM_TARGET_DEVICE=${vllm_target_device} \
    DOCKER_PLATFORM=${docker_platform} TORCH_CUDA_ARCH_LIST=${sm_arch} \
    docker compose -f docker-compose.yaml up ${docker_compose_service} --build --force-recreate
