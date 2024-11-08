inspect_args

test_path=${args[test-path]}
tag=${args[--tag]}
verbose=${args[--verbose]}
workers=${args[--workers]}

if [ -z "$tag" ]; then
  tag="cray:latest"
fi

./cray build-image

declare -a start_slurm_command_parts

start_slurm_command_parts=(
    "./scripts/start_slurm.sh" ";"
)

declare -a pytest_command_parts
pytest_command_parts=(
      "${start_slurm_command_parts[*]}"
      "python" "-m" "pytest" "-rF" "--dist=loadgroup" "--color=yes"
      "--durations=10" "--durations-min=10.0" "--forked" "--verbose" "-vv"
      "-o" "log_cli=true" "-o" "log_cli_level=DEBUG")

if [ "yes" == "$verbose" ]; then
  pytest_command_parts+=("-rP")
fi

pytest_command_parts+=($test_path)
pytest_command="${pytest_command_parts[*]}"

TTY=-t
if test -t 0; then
  TTY=-it
fi

echo "Test path: $test_path"
echo $command
echo $tag

# Remove the trailing * from the test path, if it exists
base_test_path=${test_path%\*}

if [ ! -e $base_test_path ]; then
  echo $(red_bold "File or does directory exists:  $base_test_path")
  exit 1
fi

declare -a docker_command_parts

docker_command_parts=("docker" "run" "--rm" "--init" )

docker_command_parts+=("-e" "PY_FORCE_COLOR=1"
            "-e" "PY_COLORS=1"
            "-e" "FORCE_COLOR=1"
            "$TTY" "$tag" "sh" "-c" "'$pytest_command'")

docker_command="${docker_command_parts[*]}"
echo $docker_command
eval $docker_command

