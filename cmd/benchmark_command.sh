inspect_args

target=${args[target]}
visible_gpus=${args[visible-gpus]}

./scalarlm build-image $target

declare -a benchmark_command_parts
benchmark_command_parts=(
      "PYTORCH_TUNABLEOP_ENABLED=1" "PYTORCH_TUNABLEOP_VERBOSE=1" "CUDA_VISIBLE_DEVICES=${visible_gpus}" "python" "/app/cray/test/benchmark/main.py"
)

benchmark_command="${benchmark_command_parts[*]}"

echo $command

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set cwd to the project root directory
ROOT_DIRECTORY=$LOCAL_DIRECTORY/..

declare -a docker_command_parts

# Make sure the data directory exists
mkdir -p $ROOT_DIRECTORY/data

docker_command_parts=("docker" "run" "-it" "--rm" "--network" "host" "-v" "$ROOT_DIRECTORY/data:/app/cray/data")

declare -a gpu_options

# Set the GPU options depending on the target
if [ "$target" == "cpu" ]; then
    gpu_options+=()
elif [ "$target" == "amd" ]; then
    gpu_options+=("--device" "/dev/kfd" "--device" "/dev/dri")
else
    gpu_options+=("--gpus" "all")
fi

docker_command_parts+=("${gpu_options[@]}")
docker_command_parts+=("cray:latest" "sh" "-c" "'$benchmark_command'")

docker_command="${docker_command_parts[*]}"
echo $docker_command
eval $docker_command

