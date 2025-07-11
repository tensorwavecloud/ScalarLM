
from cray_infra.util.get_config import get_config

from cryptography.fernet import Fernet


import os

def get_hf_token():
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]

    config = get_config()

    if config["hf_token"] != "":
        return config["hf_token"]

    # Fuck you huggingface
    encrypted_token = config["hf_encrypted_token"]
    key = config["encryption_key"]

    cipher = Fernet(key)

    token = cipher.decrypt(encrypted_token).decode()

    return token



