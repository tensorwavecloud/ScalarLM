from cray_infra.util.default_job_config import JobConfig

import yaml
import os


def get_job_config():
    job_config_path = get_job_config_path()

    with open(job_config_path, "r") as stream:
        job_config = yaml.safe_load(stream)

    # fill in missing values with defaults
    job_config = JobConfig(**job_config).dict()

    return job_config


def get_job_config_path():
    assert (
        "CRAY_TRAINING_JOB_CONFIG_PATH" in os.environ
    ), "CRAY_TRAINING_JOB_CONFIG_PATH not set"
    return os.environ["CRAY_TRAINING_JOB_CONFIG_PATH"]
