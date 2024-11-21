from cray_infra.util.default_config import Config

import os
import yaml

def get_config():
    loaded_config = {}

    config_path = "/app/cray/cray-config.yaml"

    if os.path.exists(config_path):
        with open(config_path, "r") as stream:
            loaded_config = yaml.safe_load(stream)

    return Config(**loaded_config).dict()
