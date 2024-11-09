from masint.util.get_api_base import get_api_base


def make_api_url(endpoint):
    api_base = get_api_base()
    return f"{api_base}/{endpoint}"
