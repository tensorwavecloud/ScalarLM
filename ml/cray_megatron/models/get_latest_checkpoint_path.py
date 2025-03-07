from cray_infra.util.get_job_config import get_job_config

import os

import logging

logger = logging.getLogger(__name__)


def get_latest_checkpoint_path():
    job_config = get_job_config()
    job_path = job_config["job_directory"]

    # checkpoints start with checkpoint_, followed by an int step count, and end with .pt
    # the latest checkpoint is the one with the highest step count
    checkpoint_files = [
        f
        for f in os.listdir(job_path)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]

    if len(checkpoint_files) == 0:
        return None

    latest_checkpoint = max(
        checkpoint_files, key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    return os.path.join(job_path, latest_checkpoint)


def delete_old_checkpoints():
    job_config = get_job_config()
    job_path = job_config["job_directory"]

    # checkpoints start with checkpoint_, followed by an int step count, and end with .pt
    # the latest checkpoint is the one with the highest step count
    checkpoint_files = [
        f
        for f in os.listdir(job_path)
        if f.startswith("checkpoint_") and f.endswith(".pt")
    ]

    if len(checkpoint_files) == 0:
        return

    sorted_checkpoints = list(
        sorted(checkpoint_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    )

    max_checkpoints_to_keep = job_config["max_checkpoints_to_keep"]

    if len(sorted_checkpoints) > max_checkpoints_to_keep:
        for checkpoint in sorted_checkpoints[:-max_checkpoints_to_keep]:
            logger.info(f"Deleting old checkpoint: {checkpoint}")
            os.remove(os.path.join(job_path, checkpoint))
