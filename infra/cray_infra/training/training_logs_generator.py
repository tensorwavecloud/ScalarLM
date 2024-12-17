from cray_infra.util.get_config import get_config

import aiofiles
import os
import json


def training_logs_generator(model_name: str, starting_line_number: int):
    config = get_config()

    if model_name == "latest":
        model_name = get_latest_model()

    job_directory = os.path.join(config["training_job_directory"], model_name)

    # Find the log file inside the job directory, it will be named "slurm-<job_id>.out, but we don't know the job_id yet
    log_file = None

    for file in os.listdir(job_directory):
        if file.startswith("slurm-") and file.endswith(".out"):
            log_file = os.path.join(job_directory, file)
            break

    if log_file is None:
        raise FileNotFoundError(f"Could not find log file in {job_directory}")

    async def generate():
        async with aiofiles.open(log_file, mode="r") as f:
            line_number = 0
            async for line in f:
                if line_number < starting_line_number:
                    line_number += 1
                    continue

                yield json.dumps({"line": line.rstrip(), "line_number": line_number}) + "\n"
                line_number += 1

    return generate()


def get_latest_model():
    config = get_config()

    if not os.path.exists(config["training_job_directory"]):
        raise FileNotFoundError("No training jobs found")

    # Get the latest model by timestamp
    models = os.listdir(config["training_job_directory"])

    if len(models) == 0:
        raise FileNotFoundError("No training jobs found")

    models.sort(
        key=lambda x: os.path.getmtime(
            os.path.join(config["training_job_directory"], x)
        ),
        reverse=True,
    )

    model_name = models[0]

    return model_name
