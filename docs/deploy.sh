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

# Clean up the old deployment directory
rm -rf $LOCAL_DIRECTORY/gh-pages-deployment

# Clone the repository
git clone git@github.com:tensorwavecloud/scalarlm $LOCAL_DIRECTORY/gh-pages-deployment

# Change to the deployment directory
cd $LOCAL_DIRECTORY/gh-pages-deployment

# Change to the git branch
git checkout gh-pages

# Copy the local files from cray-docs to the deployment directory
cp $LOCAL_DIRECTORY/cray-docs/mkdocs.yml $LOCAL_DIRECTORY/gh-pages-deployment
cp -r $LOCAL_DIRECTORY/cray-docs/docs $LOCAL_DIRECTORY/gh-pages-deployment/docs

# Add all the files to the git repository
#git add .

# Commit the changes
#git commit -m "Deploying the latest documentation"

# Run mkdocs gh-deploy
mkdocs gh-deploy



