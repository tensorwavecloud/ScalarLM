# Command Line Interface

The Craylm CLI is used to inspect and monitor training jobs, as well as to manage the Cray-LM platform.

```console
cray - Craylm CLI

Usage:
  cray COMMAND
  cray [COMMAND] --help | -h
  cray --version | -v

Commands:
  build-image   Build image from dockerfile
  depot-build   Build image from dockerfile and push to depot
  up            Start the container
  test          Run tests in the container
  deploy        Deploy the cray platform to modal
  serve         Serve the cray platform using modal
  llm           Invoke the LLM tool
  diffusion     Evaluate a diffusion model

Options:
  --help, -h
    Show this help

  --version, -v
    Show version number
```

## Installation

Clone the Craylm repository and run the `cray` command to try out the CLI.

```
git clone git@github.com:cray-lm/cray-lm.git
cd cray-lm
./cray
```

## Python package

The Craylm CLI is also available as a Python package. You can install it using pip:

```console
pip install cray-lm
```

After installing the package, you can use the `cray-lm` command in your terminal.

```console
(environment) gregorydiamos@MacBook-Air-Gregory cray % cray-lm
usage: cray-lm [-h] {logs,plot,ls,squeue} ...

The command line interface for MasInt

positional arguments:
  {logs,plot,ls,squeue}
    logs                View logs
    plot                Plot the results of a model
    ls                  List models
    squeue              View the squeue

options:
  -h, --help            show this help message and exit
```

The python package includes is a client for the Craylm platform, and can be
used to interact with the Craylm server.

