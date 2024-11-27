inspect_args

target=${args[target]}

declare -a vllm_target_device
declare -a docker_platform

# If target is cpu, build the image with the cpu base image
if [ "$target" == "cpu" ]; then
    vllm_target_device=("cpu")
    docker_platform=("linux/amd64")
else
    vllm_target_device=("cuda")
    docker_platform=("linux/amd64")
fi

docker_build_command="depot build --platform ${docker_platform} --build-arg BASE_NAME=${target} --build-arg VLLM_TARGET_DEVICE=${vllm_target_device} -t gdiamos/masint-$target:latest --push ."

# Run docker build command
echo $(green_bold Building image with command: ${docker_build_command})
eval $docker_build_command

echo $(green_bold Successfully built image)

