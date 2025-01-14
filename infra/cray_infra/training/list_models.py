from cray_infra.api.fastapi.routers.request_types.list_models_response import (
    ListModelsResponse,
)

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

    for model_name in os.listdir(config["training_job_directory"]):
        models.append(model_name)

    return ListModelsResponse(models=models)




