from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

from cray_infra.training.vllm_model_manager import get_vllm_model_manager

from cray_infra.one_server.wait_for_vllm import get_vllm_health

from cray_infra.util.get_config import get_config

import os

import logging

from pathlib import Path

logger = logging.getLogger(__name__)


async def register_megatron_models():
    logger.info("Registering Megatron models")

    if not await is_vllm_ready():
        logger.info("VLLM is not ready. Skipping model registration")
        return

    # Get all the models that are in the model directory
    models = get_models()

    # Get all the models that are already registered
    registered_models = await get_registered_models()

    # Register all the models that are not already registered
    async for model in models:
        if model not in registered_models:
            await register_model(model)


async def get_models():
    config = get_config()
    logger.info(f"Getting models from {config['training_job_directory']}")

    if not os.path.exists(config["training_job_directory"]):
        return

    for path in os.listdir(config["training_job_directory"]):
        root = os.path.join(config["training_job_directory"], path)
        logger.info(f"Checking {root}")
        # Look for any file matching *.pt* in this directory
        pt_files = list(Path(root).glob("*.pt"))
        if pt_files:
            logger.info(f"Found model {path}")
            yield path


async def is_vllm_ready():
    health_status = await get_vllm_health()
    if health_status == 200:
        return True
    return False


async def get_registered_models():
    config = get_config()
    session = get_global_session()
    async with session.get(config["vllm_api_url"] + "/v1/models") as resp:
        request_data = await resp.json()

    logger.info(f"Registered models: {request_data['data']}")

    models = [model["id"] for model in request_data["data"]]

    vllm_model_manager = get_vllm_model_manager()
    vllm_model_manager.set_registered_models(models)

    return models


async def register_model(model):
    session = get_global_session()
    config = get_config()
    path = os.path.join(config["training_job_directory"], model)
    logger.info(f"Registering model {model} at {path}")
    async with session.post(
        config["vllm_api_url"] + "/v1/load_lora_adapter",
        json={"lora_name": model, "lora_path": path},
    ) as resp:
        if resp.status != 200:
            raise Exception(f"Failed to register model {model} at {path}")
