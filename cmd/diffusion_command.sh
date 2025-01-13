inspect_args

model=${args[model]}

if [ -z "$model" ]; then
    model="latest"
fi

./cray build-image

declare -a diffusion_command_parts
diffusion_command_parts=(
      "python" "/app/cray/ml/diffusion_forcing/eval.py" "--model" "$model"
)

echo $tail

diffusion_command="${diffusion_command_parts[*]}"

echo $command

declare -a docker_command_parts

docker_command_parts=("docker" "exec" "-it" )

docker_command_parts+=("cray-vllm-1" "sh" "-c" "'$diffusion_command'")

docker_command="${docker_command_parts[*]}"
echo $docker_command
eval $docker_command



