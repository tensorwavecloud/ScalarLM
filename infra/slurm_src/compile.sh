#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

SOURCE_FILE=$LOCAL_DIRECTORY/cgroup_docker.c
INCLUDE_PATH=/usr/include

# Compile the cgroup_docker.c file into a shared object file
gcc -I$INCLUDE_PATH -Wall -fPIC -shared -o $LOCAL_DIRECTORY/cgroup_docker.so $SOURCE_FILE

# Determine if the target is an x86_64 or aarch64 machine
if [ "$(uname -m)" == "x86_64" ]; then
    TARGET="x86_64-linux-gnu"
elif [ "$(uname -m)" == "aarch64" ]; then
    TARGET="aarch64-linux-gnu"
else
    echo "Unsupported architecture"
    exit 1
fi

# Copy the shared object file to the /usr/lib directory
cp /app/cray/infra/slurm_src/cgroup_docker.so /usr/lib/$TARGET/slurm-wlm/cgroup_docker.so

# Disable the plugin on the AMD target
if [ $BASE_NAME == "amd" ] || [ $BASE_NAME="nvidia" ]; then
    sed -i -e 's/CgroupPlugin=cgroup\/docker/CgroupPlugin=cgroup\/v1/g' /app/cray/infra/slurm_configs/cgroup.conf
fi


