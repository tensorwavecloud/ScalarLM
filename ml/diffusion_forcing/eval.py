from cray_infra.training.get_latest_model import get_latest_model
from cray_infra.util.get_config import get_config

from cray_megatron.models.diffusion_forcing.diffusion_forcing_eval import (
    diffusion_forcing_eval,
)

from argparse import ArgumentParser

import os
import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging()

    path = get_path()

    diffusion_forcing_eval(path)

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Ignore urllib and filelock
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)


def get_path():
    parser = ArgumentParser()

    parser.add_argument(
        "--model", type=str, required=True, help="Path to the model job directory"
    )
    args = parser.parse_args()

    if args.model == "latest":
        return find_latest_model()

    return args.model


def find_latest_model():
    latest_model = get_latest_model()

    config = get_config()

    job_directory = os.path.join(config["training_job_directory"], latest_model)

    logger.info(f"Using latest model: {job_directory}")

    return job_directory


main()
