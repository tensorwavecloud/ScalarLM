from cray_infra.api.fastapi.routers.request_types.list_models_response import (
    ListModelsResponse,
)

from cray_infra.training.get_training_job_info import get_training_job_status

from cray_infra.training.vllm_model_manager import get_vllm_model_manager

from cray_infra.util.get_config import get_config

import os

import logging

logger = logging.getLogger(__name__)


async def list_models():
    logger.info("Listing models")

    config = get_config()

    models = []

    if not os.path.exists(config["training_job_directory"]):
        return ListModelsResponse(models=models)

    registered_models = set(get_vllm_model_manager().get_registered_models())

    for model_name in os.listdir(config["training_job_directory"]):
        job_status = get_training_job_status(model_name)

        status = {}

        if job_status is not None:
            status = job_status

        models.append(
            {
                "name": model_name,
                "deployed": model_name in registered_models,
                "status": status.get("status", "UNKNOWN"),
                "start_time": status.get("start_time", 0),
            }
        )

    return ListModelsResponse(models=models)
