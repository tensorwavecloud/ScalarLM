inspect_args

target=${args[target]}

declare -a vllm_target_device

if [ "$target" == "cpu" ]; then
    vllm_target_device=("cpu")
elif [ "$target" == "amd" ]; then
    vllm_target_device=("rocm")
else
    vllm_target_device=("cuda")
fi

BASE_NAME=${target} VLLM_TARGET_DEVICE=${vllm_target_device} docker compose -f docker-compose.yaml up --build
