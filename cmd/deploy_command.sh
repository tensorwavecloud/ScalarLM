inspect_args

target=${args[target]}

declare -a deployment_name
declare -a deployment_script

# If target is cpu, build the image with the cpu base image
if [ "$target" == "cpu" ]; then
    deployment_name=("cray-cpu-llama-3.2-1b-instruct")
    deployment_script=("deployment/modal/staging/cpu/deploy.py")
else
    deployment_name=("cray-nvidia-llama-3.2-3b-instruct")
    deployment_script=("deployment/modal/staging/nvidia/deploy.py")
fi

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set cwd to the project root directory
ROOT_DIRECTORY=$LOCAL_DIRECTORY/..

cd $ROOT_DIRECTORY

modal_deploy_command="modal deploy --name $deployment_name $ROOT_DIRECTORY/$deployment_script"

# Run docker build command
echo $(green_bold Deploying image with command: ${modal_deploy_command})
eval $modal_deploy_command

echo $(green_bold Successfully deployed image)
