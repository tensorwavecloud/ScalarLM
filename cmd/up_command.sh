inspect_args

target=${args[target]}

declare -a vllm_target_device
declare -a docker_compose_service

if [ "$target" == "cpu" ]; then
    vllm_target_device=("cpu")
    docker_compose_service="cray"
elif [ "$target" == "amd" ]; then
    vllm_target_device=("rocm")
    docker_compose_service="cray-amd"
else
    vllm_target_device=("cuda")
    docker_compose_service="cray-nvidia"
fi

BASE_NAME=${target} VLLM_TARGET_DEVICE=${vllm_target_device} docker compose -f docker-compose.yaml up ${docker_compose_service}
