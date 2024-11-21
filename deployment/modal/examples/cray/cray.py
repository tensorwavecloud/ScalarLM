import modal
import os

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

cray_image = (modal.Image.from_registry(
    "gdiamos/masint-cpu:latest",
    secret=modal.Secret.from_dict(
        {
            "REGISTRY_USERNAME": "gdiamos",
            "REGISTRY_PASSWORD": "dckr_pat_q2mAHptoFmIW43E1_d6STXF65t0",
        }
    ),
)
.pip_install("fastapi >= 0.107.0", "pydantic >= 2.9")
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

@app.function(image=cray_image)
@modal.asgi_app()
def fastapi_app():
    return web_app

@app.function(image=cray_image)
@modal.asgi_app()
def vllm_app():
    os.environ["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "true"
    os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server.")
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
    import asyncio

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    return model_config

def get_engine_client(args):
    import asyncio

    try:  # adapted from vLLM source -- https://github.com/vllm-project/vllm/blob/507ef787d85dec24490069ffceacbd6b161f4f72/vllm/entrypoints/openai/api_server.py#L235C1-L247C1
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        engine_client = event_loop.run_until_complete(get_async_engine_client(args))
    else:
        # When using single vLLM without engine_use_ray
        engine_client = asyncio.run(get_async_engine_client(args))

    return engine_client

async def get_async_engine_client(args):
    async with build_async_engine_client(args) as engine_client:
        return engine_client
