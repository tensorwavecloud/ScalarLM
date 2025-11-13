#!/bin/bash

# Safely execute this bash script
# e exit on first failure
# x all executed commands are printed to the terminal
# u unset variables are errors
# a export all variables to the environment
# E any trap on ERR is inherited by shell functions
# -o pipefail | produces a failure code if any stage fails
set -Eeuoxa pipefail

export CRAY_TRAINING_JOB_CONFIG_PATH=REPLACE_CONFIG_PATH

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${CRAY_TRAINING_JOB_CONFIG_PATH}" )" >/dev/null 2>&1 && pwd )"

# Put the current ml directory in the python path so that the modules can be imported
export PYTHONPATH=$LOCAL_DIRECTORY/ml:$PYTHONPATH

mpirun --allow-run-as-root python $LOCAL_DIRECTORY/ml/cray_megatron/main.py $*
