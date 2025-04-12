from cray_infra.util.get_config import get_config

import os
import json


def get_latest_model():
    config = get_config()

    if not os.path.exists(config["training_job_directory"]):
        raise FileNotFoundError("No training jobs found")

    # Get the latest model by timestamp
    models = os.listdir(config["training_job_directory"])

    if len(models) == 0:
        raise FileNotFoundError("No training jobs found")

    models.sort(
        key=lambda x: get_start_time(os.path.join(config["training_job_directory"], x)),
        reverse=True,
    )

    model_name = models[0]

    return model_name


def get_start_time(path):
    with open(os.path.join(path, "status.json")) as f:
        status = json.load(f)

    if "history" not in status:
        return 0

    return status.get("start_time", 0)
