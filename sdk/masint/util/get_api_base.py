import masint
import scalarlm

import os

def get_api_base():
    if hasattr(scalarlm, "api_url") and scalarlm.api_url is not None:
        return scalarlm.api_url

    if hasattr(masint, "api_url") and masint.api_url is not None:
        return masint.api_url

    if "SCALARLM_API_URL" in os.environ:
        return os.environ["SCALARLM_API_URL"]

    if "MASINT_API_URL" in os.environ:
        return os.environ["MASINT_API_URL"]

    return "http://localhost:8000"

