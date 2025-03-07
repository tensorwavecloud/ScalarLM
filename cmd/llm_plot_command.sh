inspect_args

model=${args[model]}

if [ -z "$model" ]; then
    model="latest"
fi

./cray build-image

declare -a plot_command_parts
plot_command_parts=(
      "python" "/app/cray/sdk/masint/cli/main.py" "plot" "--model" "$model"
)

plot_command="${plot_command_parts[*]}"

echo $command

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set cwd to the project root directory
ROOT_DIRECTORY=$LOCAL_DIRECTORY/..

declare -a docker_command_parts

# Make sure the data directory exists
mkdir -p $ROOT_DIRECTORY/data

docker_command_parts=("docker" "run" "--rm" "-v" "$ROOT_DIRECTORY/data:/app/cray/data" "--network" "host")

docker_command_parts+=("cray:latest" "sh" "-c" "'$plot_command'")

docker_command="${docker_command_parts[*]}"
echo $docker_command
eval $docker_command

