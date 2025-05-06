from cray_infra.api.fastapi.routers.request_types.train_request import (
    TrainResponse
)

from cray_infra.training.vllm_model_manager import get_vllm_model_manager
from cray_infra.training.get_latest_model import get_latest_model

from cray_infra.util.get_config import get_config

from fastapi import HTTPException, status

import yaml
import json
import os

import logging

logger = logging.getLogger(__name__)


async def get_training_job_info(job_hash: str):
    try:
        if job_hash == "latest":
            job_hash = get_latest_model()

        job_status, job_directory_path = get_training_job_status(job_hash)

        if job_status is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job was not found at {job_directory_path}",
            )
        # Get job config
        job_config = None

        config_filepath = os.path.join(job_directory_path, "config.yaml")
        try:
            with open(config_filepath, "r") as file:
                job_config = yaml.safe_load(file)

        except FileNotFoundError:
            logger.error(f"{config_filepath} file not found")
        except json.JSONDecodeError:
            logger.error("Invalid YAML")

        if job_config is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job config was not found at {job_directory_path}",
            )

        registered_models = set(get_vllm_model_manager().get_registered_models())

        return TrainResponse(
            job_status=job_status, job_config=job_config, deployed=job_hash in registered_models
        )
    except Exception as e:
        logger.exception(
            f"Error retrieving training job {job_hash} "
            "Exception: {type(e).__name__}, Message: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve training job information: {str(e)}",
        )


def get_job_directory_for_hash(hash_id: str):
    config = get_config()
    perfect_match = os.path.join(config["training_job_directory"], hash_id)

    if os.path.exists(perfect_match):
        return perfect_match

    # Check for partial match
    job_directory_path = None

    for job_directory in os.listdir(config["training_job_directory"]):
        if hash_id in job_directory:
            job_directory_path = os.path.join(
                config["training_job_directory"], job_directory
            )
            break

    if job_directory_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job was not found at {config['training_job_directory']}",
        )

    return job_directory_path


def get_training_job_status(job_hash: str):
    job_status = None

    job_directory_path = get_job_directory_for_hash(job_hash)
    status_filepath = os.path.join(job_directory_path, "status.json")

    # Get job status
    try:
        with open(status_filepath, "r") as file:
            job_status = json.loads(file.readline().strip())
    except FileNotFoundError:
        logger.error("File not found")
    except json.JSONDecodeError:
        logger.error("Invalid JSON in first line")

    return job_status, job_directory_path
