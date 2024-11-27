import modal
import os
import subprocess
import asyncio

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

local_config_path = os.path.join(os.path.dirname(__file__), "cray-config.yaml")

try:
    volume = modal.Volume.lookup("models", create_if_missing=True)
except modal.exception.NotFoundError:
    raise Exception("Missing volume 'models'")

try:
    jobs_volume = modal.Volume.lookup("jobs", create_if_missing=True)
except modal.exception.NotFoundError:
    raise Exception("Missing volume 'jobs'")

cray_image = (
    modal.Image.from_registry(
        "gdiamos/masint-cpu:latest",
        secret=modal.Secret.from_name("dockerhub-credentials"),
    )
    .pip_install("fastapi >= 0.107.0", "pydantic >= 2.9")
    .copy_local_file(
        local_path=local_config_path, remote_path="/app/cray/cray-config.yaml"
    )
)

app = modal.App()


with cray_image.imports():
    from cray_infra.api.fastapi.main import app as web_app
    from cray_infra.util.get_config import get_config

    # vLLM imports
    from vllm.entrypoints.openai.api_server import build_app
    from vllm.entrypoints.openai.api_server import build_async_engine_client
    from vllm.entrypoints.openai.api_server import init_app_state
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils import FlexibleArgumentParser

    from vllm.entrypoints.openai import api_server


@app.function(
    image=cray_image,
    container_idle_timeout=5 * 60,
    allow_concurrent_inputs=32,
    secrets=[modal.Secret.from_name("huggingface-credentials")],
    volumes={"/root/.cache/huggingface": volume, "/app/cray/jobs": jobs_volume},
)
@modal.asgi_app()
def fastapi_app():
    run_this_on_container_startup()
    return web_app


@app.function(
    image=cray_image,
    allow_concurrent_inputs=32,
    memory=4 * 1024,
    secrets=[modal.Secret.from_name("huggingface-credentials")],
    volumes={"/root/.cache/huggingface": volume, "/app/cray/jobs": jobs_volume},
)
@modal.asgi_app()
def vllm_app():
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args(args=["--enable-lora"])

    config = get_config()
    args.model = config["model"]
    args.disable_frontend_multiprocessing = True

    logger.info(f"Running vLLM with args: {args}")

    engine_client = get_engine_client(args)

    vllm_app = build_app(args)

    model_config = get_model_config(engine_client)
    init_app_state(engine_client, model_config, vllm_app.state, args)

    return vllm_app


def get_model_config(engine):
    return asyncio.run(engine.get_model_config())


def get_engine_client(args):
    return asyncio.run(get_async_engine_client(args))


async def get_async_engine_client(args):
    async with build_async_engine_client(args) as engine_client:
        return engine_client


def run_this_on_container_startup():
    output = subprocess.Popen(["/app/cray/scripts/start_slurm.sh"])
