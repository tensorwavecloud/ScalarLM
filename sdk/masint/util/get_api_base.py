
import masint

def get_api_base():
    if masint.api_url is None:
        return "http://localhost:8000"

    return masint.api_url

