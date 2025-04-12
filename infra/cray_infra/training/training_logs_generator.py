from cray_infra.training.get_latest_model import get_latest_model
from cray_infra.training.get_training_job_info import get_job_directory_for_hash

from cray_infra.util.get_config import get_config

import aiofiles
import os
import json
import logging


logger = logging.getLogger(__name__)


def training_logs_generator(model_name: str, starting_line_number: int):
    config = get_config()

    if model_name == "latest":
        latest_model_name = get_latest_model()
        if latest_model_name is None:
            raise FileNotFoundError("Could not find any models")
        model_name = latest_model_name

    job_directory = get_job_directory_for_hash(model_name)

    # Find the log file inside the job directory, it will be named "slurm-<job_id>.out, but we don't know the job_id yet
    log_files = []

    for file in os.listdir(job_directory):
        if file.startswith("slurm-") and file.endswith(".out"):
            log_file = os.path.join(job_directory, file)
            log_files.append(log_file)

    # sort the log files by name
    log_files.sort()

    logger.info(f"Found log files: {log_files}")

    if len(log_files) == 0:
        raise FileNotFoundError(f"Could not find log file in {job_directory}")

    async def generate():
        line_number = 0
        for log_file in log_files:
            async with aiofiles.open(log_file, mode="r") as f:
                async for line in f:
                    if line_number < starting_line_number:
                        line_number += 1
                        continue

                    yield json.dumps(
                        {"line": line.rstrip(), "line_number": line_number}
                    ) + "\n"
                    line_number += 1

    return generate()


