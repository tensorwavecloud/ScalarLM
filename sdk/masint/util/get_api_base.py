import masint

import os

def get_api_base():
    if masint.api_url is not None:
        return masint.api_url

    if "MASINT_API_URL" in os.environ:
        return os.environ["MASINT_API_URL"]

    return "http://localhost:8000"

