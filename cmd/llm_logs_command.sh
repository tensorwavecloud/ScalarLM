inspect_args

model=${args[model]}
tail=${args[--tail]}
lines=${args[--lines]}
follow=${args[--follow]}

if [ -z "$model" ]; then
    model="latest"
fi

./cray build-image

declare -a log_command_parts
log_command_parts=(
      "python" "/app/cray/sdk/cli/main.py" "logs" "--model" "$model" "--lines" "$lines"
)

echo $tail

# If tail exists, add it to the command
if [ -n "$tail" ]; then
    log_command_parts+=("--tail")
fi

# If follow exists, add it to the command
if [ -n "$follow" ]; then
    log_command_parts+=("--follow")
fi

log_command="${log_command_parts[*]}"

echo $command

declare -a docker_command_parts

docker_command_parts=("docker" "run" "-it" "--rm" "--network" "host")

docker_command_parts+=("cray:latest" "sh" "-c" "'$log_command'")

docker_command="${docker_command_parts[*]}"
echo $docker_command
eval $docker_command


