from cray_infra.api.fastapi.aiohttp.get_global_session import get_global_session

from cray_infra.training.vllm_model_manager import get_vllm_model_manager

from cray_infra.util.get_config import get_config

import os

import logging

import yaml

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

    served_base = config.get("model")

    for path in os.listdir(config["training_job_directory"]):
        root = os.path.join(config["training_job_directory"], path)
        logger.info(f"Checking {root}")
        # Look for any file matching *.pt* in this directory
        pt_files = list(Path(root).glob("*.pt"))
        if not pt_files:
            continue
        # Only register adapters trained for the model this server is serving.
        # The serve worker loads every registered adapter onto the single served
        # base with no compatibility check; a LoRA trained for a different
        # architecture has mismatched tensor shapes and crashes set_lora
        # (IndexError in column_parallel_linear). In the multi-model finetune
        # sweep the shared jobs/ dir accumulates adapters from many archs, so
        # scope to the served base here. A missing/unreadable llm_name is kept
        # (fail-open) so the ordinary single-model case is unaffected.
        adaptor_base = _read_adaptor_base_model(root)
        if served_base and adaptor_base and adaptor_base != served_base:
            logger.info(
                f"Skipping adaptor {path}: trained for {adaptor_base}, "
                f"serving {served_base}"
            )
            continue
        logger.info(f"Found model {path}")
        yield path


def _read_adaptor_base_model(job_directory_path):
    """The base model (`llm_name`) an adapter was trained for, read from the
    job's config.yaml, or None if unreadable."""
    config_filepath = os.path.join(job_directory_path, "config.yaml")
    try:
        with open(config_filepath, "r") as file:
            job_config = yaml.safe_load(file)
    except (FileNotFoundError, yaml.YAMLError):
        logger.warning(f"Could not read config.yaml in {job_directory_path}")
        return None
    if not job_config:
        return None
    return job_config.get("llm_name")


async def get_registered_models():
    vllm_model_manager = get_vllm_model_manager()

    return set(vllm_model_manager.get_registered_models())


async def register_model(model):
    vllm_model_manager = get_vllm_model_manager()

    vllm_model_manager.register_model(model)


