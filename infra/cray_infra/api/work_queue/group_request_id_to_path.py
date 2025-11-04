from cray_infra.util.get_config import get_config

import os

def group_request_id_to_path(group_request_id: str):
    config = get_config()
    base_path = config["upload_base_path"]

    request_path = os.path.join(base_path, f"{group_request_id}.json")

    return request_path
