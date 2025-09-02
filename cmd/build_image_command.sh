inspect_args

target=${args[target]}
sm_arch=${args[sm_arch]}

declare -a vllm_target_device
declare -a docker_platform

# If target is cpu, build the image with the cpu base image
if [ "$target" == "cpu" ]; then
    vllm_target_device=("cpu")
    if [ "$(uname -m)" == "x86_64" ]; then
        docker_platform=("linux/amd64")
    else
        docker_platform=("linux/arm64/v8")
    fi
elif [ "$target" == "amd" ]; then
    vllm_target_device=("rocm")
    docker_platform=("linux/amd64")
    sm_arch="gfx942"
else
    vllm_target_device=("cuda")
    docker_platform=("linux/amd64")
    if [ "$sm_arch" == "auto" ]; then
        # Auto-detect the architecture of the GPU using nvidia-smi
        sm_arch=($(nvidia-smi --query-gpu=compute_cap --format=csv,noheader))
    fi
fi

docker_build_command="docker build --platform ${docker_platform} --build-arg BASE_NAME=${target} --build-arg TORCH_CUDA_ARCH_LIST=${sm_arch} --build-arg VLLM_TARGET_DEVICE=${vllm_target_device} -t cray:latest --shm-size=8g ."

# Run docker build command
echo $(green_bold Building image with command: ${docker_build_command})
eval $docker_build_command

echo $(green_bold Successfully built image)
