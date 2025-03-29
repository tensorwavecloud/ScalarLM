inspect_args

./cray build-image

declare -a benchmark_command_parts
benchmark_command_parts=(
      "python" "/app/cray/test/benchmark/main.py"
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

docker_command_parts=("docker" "run" "--rm" "--network" "host" "-v" "$ROOT_DIRECTORY/data:/app/cray/data")

docker_command_parts+=("cray:latest" "sh" "-c" "'$benchmark_command'")

docker_command="${docker_command_parts[*]}"
echo $docker_command
eval $docker_command

