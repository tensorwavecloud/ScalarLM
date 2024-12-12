from cray_infra.util.get_config import get_config
from cray_infra.util.get_job_config import get_job_config

import torch

import os
import json

import logging

logger = logging.getLogger(__name__)


class TrainingHarness:
    def update_status(self, status, metadata={}):

        current_status = get_status()

        current_status["status"] = status
        for key, value in metadata.items():
            current_status[key] = value

        save_status(current_status)

    def checkpoint(self, model, checkpoint_state, checkpoint_name):
        job_config = get_job_config()

        checkpoint_path = os.path.join(job_config["job_directory"], checkpoint_name)

        torch.save(checkpoint_state, checkpoint_path)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

        saved_model_path = os.path.join(job_config["job_directory"], "saved_model")

        model.save_pretrained(saved_model_path)

        logger.info(f"Model saved to {saved_model_path}")


def get_status():
    try:
        with open(os.path.join(get_training_job_directory(), "status.json"), "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading job status: {e}")
        return {"status": "unknown"}


def get_training_job_directory():
    job_config = get_job_config()

    return job_config["job_directory"]


def save_status(job_status):
    try:
        contents = json.dumps(job_status)
    except Exception as e:
        logger.error(f"Error serializing job status: {e}")
        return

    with open(os.path.join(get_training_job_directory(), "status.json"), "w") as f:
        f.write(contents)
