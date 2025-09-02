from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

from cray_infra.training.vllm_model_manager import get_vllm_model_manager

from cray_infra.util.get_config import get_config

import os

import logging

from pathlib import Path

logger = logging.getLogger(__name__)


async def register_megatron_models():
    logger.info("Registering Megatron models")

    # Get all the models that are in the model directory
    models = get_models()

    # Get all the models that are already registered
    registered_models = await get_registered_models()

    # Register all the models that are not already registered
    async for model in models:
        if model not in registered_models:
            await register_model(model)

    logger.info(f"Finished registering Megatron models, there are {len(registered_models)} registered models.")


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


async def get_registered_models():
    vllm_model_manager = get_vllm_model_manager()

    return set(vllm_model_manager.get_registered_models())


async def register_model(model):
    vllm_model_manager = get_vllm_model_manager()

    vllm_model_manager.register_model(model)


