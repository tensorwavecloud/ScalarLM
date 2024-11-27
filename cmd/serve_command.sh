inspect_args

target=${args[target]}

declare -a deployment_script

# If target is cpu, build the image with the cpu base image
if [ "$target" == "cpu" ]; then
    deployment_script=("deployment/modal/staging/cpu/deploy.py")
else
    deployment_script=("deployment/modal/staging/nvidia/deploy.py")
fi

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set cwd to the project root directory
ROOT_DIRECTORY=$LOCAL_DIRECTORY/..

cd $ROOT_DIRECTORY

modal_serve_command="modal serve $ROOT_DIRECTORY/$deployment_script"

# Run docker build command
echo $(green_bold Deploying image with command: ${modal_serve_command})
eval $modal_serve_command

echo $(green_bold Successfully served image)
