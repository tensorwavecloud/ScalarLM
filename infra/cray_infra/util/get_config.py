from .default_config import Config

import os
import yaml

import logging

logger = logging.getLogger(__name__)


def get_config():
    loaded_config = {}

    config_path = "/app/cray/cray-config.yaml"

    if os.path.exists(config_path):
        with open(config_path, "r") as stream:
            loaded_config = yaml.safe_load(stream)

    config = Config(**loaded_config).dict()

    for key, value in config.items():
        corresponding_env_var = f"SCALARLM_{key.upper()}"

        if corresponding_env_var in os.environ:
            env_value = os.environ[corresponding_env_var]

            logger.info(f"Overriding config '{key}' with environment variable '{corresponding_env_var}' - value: {env_value}")

            if isinstance(value, bool):
                config[key] = env_value.lower() in ("true", "1", "yes")
            elif isinstance(value, int):
                config[key] = int(env_value)
            elif isinstance(value, float):
                config[key] = float(env_value)
            else:
                config[key] = env_value

    return config
