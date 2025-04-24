from masint.util.get_api_base import get_api_base


def make_api_url(endpoint, api_url=None):
    if api_url is not None:
        api_base = api_url
    else:
        api_base = get_api_base()
    return f"{api_base}/{endpoint}"
