inspect_args

# Get the directory of this script
LOCAL_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Set cwd to the project sdk directory
ROOT_DIRECTORY=$LOCAL_DIRECTORY/../sdk

cd $ROOT_DIRECTORY

# Build sdk wheel from sdk/pyproject.toml
wheel_build_command="python -m build --sdist --wheel --outdir dist/ ."

# Run sdk wheel build
echo $(green_bold Building wheel with command: ${wheel_build_command})
eval $wheel_build_command

echo $(green_bold Successfully built wheel)

# Upload wheel to pypi
pypi_upload_command="twine upload dist/*"

# Run pypi upload command
echo $(green_bold Uploading wheel to pypi with command: ${pypi_upload_command})
eval $pypi_upload_command

echo $(green_bold Successfully uploaded wheel to pypi)

