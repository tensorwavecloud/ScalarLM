from cray_infra.api.fastapi.routers.request_types.train_request import (
    TrainResponse
)

from cray_infra.training.get_latest_model import get_latest_model
from cray_infra.training.get_training_job_info import get_training_job_status


import subprocess
import os
import yaml
import json

from fastapi import HTTPException, status

import logging

logger = logging.getLogger(__name__)


async def cancel(job_hash: str):
    logger.info(f"Cancel request received for job hash: {job_hash}")

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

        logger.info(f"Job status before cancellation: {job_status}")

        try:
            # Cancel SLURM job
            slurm_job_id = job_status['job_id']

            scancel_output = subprocess.check_output(
                ["scancel", str(slurm_job_id)]
            )

            job_status['status'] = 'CANCELLED'
            write_job_status('CANCELLED', job_directory_path)

        except subprocess.CalledProcessError:
            logger.error(f"scancel command failed for job id {slurm_job_id}")

        logger.info(f"Job status after cancellation: {job_status}")

        return TrainResponse(
            job_status=job_status,
            job_config=job_config,
            deployed=False
        )

    except Exception as e:
        logger.exception(
            f"Error cancelling training job {job_hash} "
            "Exception: {type(e).__name__}, Message: {str(e)}"
        )

def write_job_status(status: str, job_directory_path: str):
    status_filepath = os.path.join(job_directory_path, "status.json")
    try:
        with open(status_filepath, "r") as file:
            job_status = json.load(file)

        job_status['status'] = status

        with open(status_filepath, "w") as file:
            json.dump(job_status, file)

    except FileNotFoundError:
        logger.error(f"{status_filepath} file not found")
    except json.JSONDecodeError:
        logger.error("Invalid JSON")
